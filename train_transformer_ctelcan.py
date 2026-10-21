"""
Αναπαραγωγή του CT-ELCAN (Li et al., Bioengineering 2025, 12, 528) στο DEAP
με EEG + GSR/EDA + PPG, σε subject-wise k-fold ή LOSO.

Ό,τι ορίζει το paper (Table 1 + κείμενο): Adam, lr 1e-4, batch 128,
50 epochs, dropout 0.1, c-SAGAN lr 1e-4, 4 Transformer layers, 4 heads,
hidden 128, gradient clipping στο norm 10, LOSO, accuracy ανά subject
(mean/std), binary valence/arousal (+ quaternary). Δεν ορίζει preprocessing,
window, normalization, μέγεθος ConvBlock, βάρη α/β/γ/φ -- βλ. τα defaults
και τα --flags παρακάτω. Δεν υπάρχει validation/early stopping: το
αποτέλεσμα είναι του ΤΕΛΕΥΤΑΙΟΥ epoch (καμία επιλογή πάνω στο test).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from src.config.config import Config
from src.models.ct_elcan import (
    CTELCAN, ConditionalDiscriminator, ConditionalGenerator, grl_schedule,
)
from src.preprocessing.baseline_correction import remove_trial_baseline
from train_cross_attention_transformer_cv import build_folds

EEG_CH = slice(0, 32)
GSR_CH = 36
PPG_CH = 38
MODALITY_CHANNELS = (32, 1, 1)
MODALITY_NAMES = ("EEG", "GSR", "PPG")


def parse_args():
    p = argparse.ArgumentParser(description="CT-ELCAN (Transformer) subject-wise CV")
    p.add_argument("--protocol", choices=["kfold", "loso"], default="kfold")
    p.add_argument("--folds", type=int, default=8)
    p.add_argument("--target", choices=["valence", "arousal", "quaternary"], default="valence")
    p.add_argument("--cache-path", type=str, default="outputs/ctelcan_windows_w128.npz")
    p.add_argument("--window", type=int, default=128, help="samples @128Hz (1 s)")
    p.add_argument("--step", type=int, default=128)
    p.add_argument("--norm", choices=["subject", "trial", "none"], default="subject")
    p.add_argument("--max-subjects", type=int, default=None)
    # Paper Table 1
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gan-lr", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--weight-decay", type=float, default=0.0, help="L2 στον Adam (εκτός paper)")
    p.add_argument("--noise-std", type=float, default=0.0, help="Gaussian noise στην είσοδο (εκτός paper)")
    p.add_argument("--select-val", action="store_true",
                   help="κράτα τα validation subjects χωριστά και επίλεξε το καλύτερο epoch σε αυτά")
    # Δεν ορίζονται στο paper (εξ. 18: L = a*Lmod + b*Lsub + g*Lemo + phi*Lgan)
    p.add_argument("--alpha", type=float, default=0.1, help="modality adversarial weight")
    p.add_argument("--beta", type=float, default=0.3, help="subject adversarial weight")
    p.add_argument("--gamma", type=float, default=1.0, help="emotion CE weight")
    p.add_argument("--phi", type=float, default=0.1, help="c-SAGAN loss weight")
    p.add_argument("--grl-max", type=float, default=1.0)
    p.add_argument("--gan-warmup-epochs", type=int, default=0,
                   help="epochs πριν χρησιμοποιηθούν τα synthetic samples στο LCAN (0 = όπως το paper)")
    p.add_argument("--no-gan", action="store_true", help="ablation: χωρίς c-SAGAN")
    p.add_argument("--no-adv", action="store_true", help="ablation: χωρίς adversarial (LCAN)")
    p.add_argument("--log-test-every", type=int, default=0,
                   help="μόνο logging του test acc ανά N epochs (δεν επηρεάζει την επιλογή)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--tag", type=str, default="ctelcan")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="deap-ct-elcan")
    p.add_argument("--wandb-run-name", type=str, default=None)
    # compat με build_folds
    p.add_argument("--loso-val-size", type=int, default=4)
    return p.parse_args()


##########################################################################
# Data
##########################################################################

def prepare_windows(args, config) -> dict:
    cache = Path(args.cache_path)
    meta = {"window": args.window, "step": args.step, "norm": args.norm,
            "max_subjects": args.max_subjects}
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if json.loads(str(z["meta"])) == meta:
            print(f"Loaded window cache {cache}", flush=True)
            return {k: z[k] for k in z.files if k != "meta"}
        raise RuntimeError(
            f"Cache {cache} has different settings ({str(z['meta'])} vs {meta}); "
            "use another --cache-path."
        )

    import pickle
    files = sorted(Path(config.DEAP_PATH).glob("s*.dat"))
    if args.max_subjects:
        files = files[:args.max_subjects]
    xs, ys, subj, trial = [], [], [], []
    for sid, f in enumerate(files):
        with open(f, "rb") as fh:
            d = pickle.load(fh, encoding="latin1")
        data = np.asarray(d["data"], dtype=np.float32)
        labels = np.asarray(d["labels"], dtype=np.float32)
        sig = np.concatenate(
            [data[:, EEG_CH], data[:, GSR_CH:GSR_CH + 1], data[:, PPG_CH:PPG_CH + 1]], axis=1
        )
        sig = remove_trial_baseline(sig, 384).astype(np.float32)  # (40, 34, 7680)
        if args.norm == "subject":
            m = sig.mean(axis=(0, 2), keepdims=True)
            s = sig.std(axis=(0, 2), keepdims=True)
            sig = (sig - m) / (s + 1e-6)
        elif args.norm == "trial":
            m = sig.mean(axis=2, keepdims=True)
            s = sig.std(axis=2, keepdims=True)
            sig = (sig - m) / (s + 1e-6)
        n_samples = sig.shape[-1]
        starts = list(range(0, n_samples - args.window + 1, args.step))
        for t in range(sig.shape[0]):
            for st in starts:
                xs.append(sig[t, :, st:st + args.window])
                ys.append(labels[t, :2])
                subj.append(sid)
                trial.append(t)
        print(f"  {f.stem}: {len(starts) * sig.shape[0]} windows", flush=True)

    out = {
        "x": np.stack(xs).astype(np.float32),
        "ratings": np.asarray(ys, dtype=np.float32),
        "subject": np.asarray(subj, dtype=np.int64),
        "trial": np.asarray(trial, dtype=np.int64),
        "subject_names": np.asarray([f.stem for f in files]),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, meta=json.dumps(meta), **out)
    print(f"Saved window cache -> {cache}", flush=True)
    return out


def make_labels(ratings: np.ndarray, target: str) -> np.ndarray:
    v = (ratings[:, 0] > 5.0).astype(np.int64)
    a = (ratings[:, 1] > 5.0).astype(np.int64)
    if target == "valence":
        return v
    if target == "arousal":
        return a
    return 2 * v + a


##########################################################################
# Train / eval
##########################################################################

def split_modalities(x: torch.Tensor):
    return [x[:, :32], x[:, 32:33], x[:, 33:34]]


@torch.no_grad()
def predict(model, x_cpu: np.ndarray, device, batch=512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x_cpu), batch):
        xb = torch.from_numpy(x_cpu[i:i + batch]).to(device)
        out.append(torch.softmax(model(split_modalities(xb)), dim=1).cpu().numpy())
    return np.concatenate(out)


def train_fold(fold, data, labels, subject_names, args, device, wandb_run):
    name_to_idx = {n: i for i, n in enumerate(subject_names)}
    if args.select_val and fold["validation"]:
        train_names = sorted(fold["train"])
        val_ids = [name_to_idx[s] for s in fold["validation"]]
    else:
        train_names = sorted(set(fold["train"]) | set(fold["validation"]))
        val_ids = []
    train_ids = [name_to_idx[s] for s in train_names]
    test_ids = [name_to_idx[s] for s in fold["test"]]

    subj = data["subject"]
    tr_mask = np.isin(subj, train_ids)
    te_mask = np.isin(subj, test_ids)
    va_mask = np.isin(subj, val_ids) if val_ids else None
    best = {"acc": -1.0, "state": None, "epoch": 0}
    n_classes = 4 if args.target == "quaternary" else 2

    x_tr = torch.from_numpy(data["x"][tr_mask]).to(device)
    y_tr = torch.from_numpy(labels[tr_mask]).to(device)
    remap = {sid: i for i, sid in enumerate(train_ids)}
    s_tr = torch.from_numpy(np.asarray([remap[s] for s in subj[tr_mask]])).to(device)

    model = CTELCAN(
        MODALITY_CHANNELS, args.window, n_classes, len(train_ids),
        dim=args.dim, layers=args.layers, heads=args.heads, dropout=args.dropout,
    ).to(device)
    gen = ConditionalGenerator(args.dim, n_classes).to(device)
    disc = ConditionalDiscriminator(args.dim, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_g = torch.optim.Adam(gen.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))

    n = len(x_tr)
    steps_per_epoch = max(1, n // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    step = 0
    m = len(MODALITY_CHANNELS)
    modality_ids = torch.arange(m, device=device)

    for epoch in range(1, args.epochs + 1):
        model.train(); gen.train(); disc.train()
        perm = torch.randperm(n, device=device)
        agg = {"emo": 0.0, "mod": 0.0, "sub": 0.0, "d": 0.0, "g": 0.0, "acc": 0.0}
        use_fake = (not args.no_gan) and epoch > args.gan_warmup_epochs
        t_ep = time.time()
        for b in range(steps_per_epoch):
            idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
            xb, yb, sb = x_tr[idx], y_tr[idx], s_tr[idx]
            bsz = len(idx)
            if args.noise_std > 0:
                xb = xb + args.noise_std * torch.randn_like(xb)
            lam = 0.0 if args.no_adv else grl_schedule(step / total_steps, args.grl_max)
            model.set_grl_lambda(lam)
            step += 1

            feats = model.encode(split_modalities(xb))       # m x (B, D, L)

            fake = None
            if not args.no_gan:
                real = torch.cat([f.detach() for f in feats], dim=0)   # (mB, D, L)
                y_rep = yb.repeat(m)
                fake_all = gen(real, y_rep)
                d_loss = (
                    F.relu(1.0 - disc(real, y_rep)).mean()
                    + F.relu(1.0 + disc(fake_all.detach(), y_rep)).mean()
                )
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()
                g_loss = -disc(fake_all, y_rep).mean()
                opt_g.zero_grad(set_to_none=True)
                g_loss.backward()
                opt_g.step()
                agg["d"] += d_loss.item(); agg["g"] += g_loss.item()
                if use_fake:
                    fake = [c.detach() for c in fake_all.chunk(m, dim=0)]

            if fake is not None:
                inp = [torch.cat([r, f], dim=0) for r, f in zip(feats, fake)]
                y_all = torch.cat([yb, yb]); s_all = torch.cat([sb, sb])
            else:
                inp, y_all, s_all = feats, yb, sb
            out = model.lcan(inp)
            b_all = len(y_all)
            l_emo = F.cross_entropy(out["emotion"], y_all)
            mod_labels = modality_ids.repeat_interleave(b_all)
            l_mod = F.cross_entropy(out["modality"], mod_labels)
            l_sub = F.cross_entropy(out["subject"], s_all)
            loss = args.gamma * l_emo
            if not args.no_adv:
                loss = loss + args.alpha * l_mod + args.beta * l_sub
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            agg["emo"] += float(l_emo); agg["mod"] += float(l_mod); agg["sub"] += float(l_sub)
            agg["acc"] += float((out["emotion"][:bsz].argmax(1) == yb).float().mean())

        stats = {k: v / steps_per_epoch for k, v in agg.items()}
        msg = (
            f"  ep {epoch:02d}/{args.epochs} emo={stats['emo']:.4f} train_acc={stats['acc']:.4f} "
            f"mod={stats['mod']:.3f} sub={stats['sub']:.3f} D={stats['d']:.3f} G={stats['g']:.3f} "
            f"grl={lam:.2f} ({time.time() - t_ep:.0f}s)"
        )
        log = {f"train/{k}": v for k, v in stats.items()}
        if args.log_test_every and epoch % args.log_test_every == 0:
            pr = predict(model, data["x"][te_mask], device)
            acc = float((pr.argmax(1) == labels[te_mask]).mean())
            msg += f" | test_win_acc={acc:.4f} (monitor only)"
            log["monitor/test_window_acc"] = acc
        if va_mask is not None:
            pv = predict(model, data["x"][va_mask], device)
            vacc = float((pv.argmax(1) == labels[va_mask]).mean())
            msg += f" | val_acc={vacc:.4f}"
            log["val/window_acc"] = vacc
            if vacc > best["acc"]:
                best = {"acc": vacc, "epoch": epoch,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        print(msg, flush=True)
        if wandb_run is not None:
            wandb_run.log({"fold": fold["fold"], "epoch": epoch, **log})

    if best["state"] is not None:
        model.load_state_dict(best["state"])
        print(f"  best val epoch = {best['epoch']} (val_acc={best['acc']:.4f})", flush=True)
    probs = predict(model, data["x"][te_mask], device)
    return {
        "probabilities": probs,
        "labels": labels[te_mask],
        "subjects": subj[te_mask],
        "trials": data["trial"][te_mask],
    }


def summarize(results, n_classes, subject_names):
    probs = np.concatenate([r["probabilities"] for r in results])
    labels = np.concatenate([r["labels"] for r in results])
    subjects = np.concatenate([r["subjects"] for r in results])
    trials = np.concatenate([r["trials"] for r in results])
    pred = probs.argmax(1)

    per_subject = []
    for sid in np.unique(subjects):
        mk = subjects == sid
        win_acc = float((pred[mk] == labels[mk]).mean())
        tr_p, tr_l = [], []
        for t in np.unique(trials[mk]):
            mt = mk & (trials == t)
            tr_p.append(int(probs[mt].mean(0).argmax()))
            tr_l.append(int(labels[mt][0]))
        tr_p, tr_l = np.asarray(tr_p), np.asarray(tr_l)
        per_subject.append({
            "subject": str(subject_names[sid]),
            "window_accuracy": win_acc,
            "trial_accuracy": float((tr_p == tr_l).mean()),
            "trial_f1": float(f1_score(tr_l, tr_p, average="macro", zero_division=0)),
        })

    win_acc = np.asarray([s["window_accuracy"] for s in per_subject])
    trial_acc = np.asarray([s["trial_accuracy"] for s in per_subject])
    trial_f1 = np.asarray([s["trial_f1"] for s in per_subject])

    keys = sorted({(int(s), int(t)) for s, t in zip(subjects, trials)})
    all_p, all_l = [], []
    for s, t in keys:
        mt = (subjects == s) & (trials == t)
        all_p.append(int(probs[mt].mean(0).argmax()))
        all_l.append(int(labels[mt][0]))
    all_p, all_l = np.asarray(all_p), np.asarray(all_l)

    return {
        "window_accuracy_pooled": float((pred == labels).mean()),
        "window_f1_pooled": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "trial_accuracy_pooled": float((all_p == all_l).mean()),
        "trial_f1_pooled": float(f1_score(all_l, all_p, average="macro", zero_division=0)),
        "subject_window_acc_mean": float(win_acc.mean()),
        "subject_window_acc_std": float(win_acc.std()),
        "subject_trial_acc_mean": float(trial_acc.mean()),
        "subject_trial_acc_std": float(trial_acc.std()),
        "subject_trial_f1_mean": float(trial_f1.mean()),
        "subject_trial_f1_std": float(trial_f1.std()),
        "worst_subject_trial_f1": float(trial_f1.min()),
        "per_subject": per_subject,
    }


def main():
    args = parse_args()
    config = Config()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    print("=" * 80)
    print(f"CT-ELCAN -- {args.protocol.upper()} ({args.target}, seed {args.seed}, device {device})")
    print("=" * 80, flush=True)

    t0 = time.time()
    data = prepare_windows(args, config)
    subject_names = [str(s) for s in data["subject_names"]]
    labels = make_labels(data["ratings"], args.target)
    n_classes = 4 if args.target == "quaternary" else 2
    print(f"{len(data['x'])} windows, {len(subject_names)} subjects, "
          f"class balance={np.bincount(labels) / len(labels)} ({time.time() - t0:.0f}s)", flush=True)

    folds = build_folds(subject_names, args)
    results = []
    for fold in folds:
        print(f"\n{'=' * 80}\nFOLD {fold['fold']} | test={fold['test']}\n{'=' * 80}", flush=True)
        t_f = time.time()
        res = train_fold(fold, data, labels, subject_names, args, device, wandb_run)
        acc = float((res["probabilities"].argmax(1) == res["labels"]).mean())
        print(f"FOLD {fold['fold']} test window acc (last epoch) = {acc:.4f} "
              f"({time.time() - t_f:.0f}s)", flush=True)
        if wandb_run is not None:
            wandb_run.log({"fold_done": fold["fold"], "fold/test_window_acc": acc})
        results.append(res)

    s = summarize(results, n_classes, subject_names)
    print("\n" + "=" * 80)
    print(f"CT-ELCAN OUT-OF-FOLD RESULTS ({args.protocol}, {args.target}, last epoch)")
    print("=" * 80)
    print(f"Window Accuracy (pooled)  : {s['window_accuracy_pooled']:.4f}")
    print(f"Window F1 macro (pooled)  : {s['window_f1_pooled']:.4f}")
    print(f"Trial Accuracy (pooled)   : {s['trial_accuracy_pooled']:.4f}")
    print(f"Trial F1 macro (pooled)   : {s['trial_f1_pooled']:.4f}")
    print(f"Subject Accuracy (%) window: {100 * s['subject_window_acc_mean']:.2f}/"
          f"{100 * s['subject_window_acc_std']:.2f} (mean/std, paper-style)")
    print(f"Subject Accuracy (%) trial : {100 * s['subject_trial_acc_mean']:.2f}/"
          f"{100 * s['subject_trial_acc_std']:.2f}")
    print(f"Subject Trial-F1          : {s['subject_trial_f1_mean']:.4f} ± {s['subject_trial_f1_std']:.4f}")
    print(f"Worst Subject Trial-F1    : {s['worst_subject_trial_f1']:.4f}")
    print("=" * 80, flush=True)

    if wandb_run is not None:
        wandb_run.summary.update({k: v for k, v in s.items() if k != "per_subject"})
        wandb_run.finish()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.tag}_{args.protocol}_{args.target}_seed{args.seed}_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": s}, f, indent=2, ensure_ascii=False)
    print(f"Report saved -> {path}", flush=True)


if __name__ == "__main__":
    main()
