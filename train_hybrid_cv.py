"""
Subject-wise Cross-Validation (k-fold / LOSO) για το Hybrid CNN-MLP v2.

Ίδιο πρωτόκολλο με train_cross_attention_transformer_cv.py: το
preprocessing γίνεται ΜΙΑ φορά για όλα τα subjects (--cache-path), κάθε
fold εκπαιδεύει από την αρχή, το validation χρησιμοποιείται μόνο για
checkpoint selection και τα out-of-fold windows συγκεντρώνονται σε ένα
ενιαίο trial-level evaluation (κάθε subject μετράει μόνο από το fold
όπου ήταν test).
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.config.config import Config
from src.models import build_model
from src.models.supervised_contrastive_loss import SupervisedContrastiveLoss
from src.pipeline.data_pipeline import DataPipeline
from src.utils.evaluation import evaluate_hierarchical_predictions

from train_hybrid_baseline import build_subject_to_idx, run_epoch
from train_cross_attention_transformer_cv import build_folds


def parse_args():
    p = argparse.ArgumentParser(description="Hybrid CNN-MLP v2 subject-wise CV")
    p.add_argument("--protocol", choices=["loso", "kfold"], default="kfold")
    p.add_argument("--folds", type=int, default=8)
    p.add_argument("--loso-val-size", type=int, default=4)
    p.add_argument("--max-subjects", type=int, default=None)
    p.add_argument("--trials-per-subject", type=int, default=None)
    p.add_argument("--cache-path", type=str, default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--embedding-dim", type=int, default=96)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--gate-balance-weight", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--eeg-norm", choices=["trial", "subject"], default="trial",
                   help="Raw-EEG z-score scope. 'subject' needs its own --cache-path.")
    p.add_argument("--target", choices=["valence", "arousal"], default="valence",
                   help="Binary target (rating > 5). Re-labels the cached windows.")
    p.add_argument("--subject-norm", action="store_true",
                   help="Label-free per-subject z-score of handcrafted features.")
    p.add_argument("--enable-adversarial", action="store_true")
    p.add_argument("--adversarial-weight", type=float, default=0.30)
    p.add_argument("--enable-contrastive", action="store_true")
    p.add_argument("--contrastive-weight", type=float, default=0.20)
    p.add_argument("--contrastive-temperature", type=float, default=0.07)
    p.add_argument("--val-smooth-window", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--tag", type=str, default="hybrid_v2_cv")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="deap-hybrid-cnn-mlp")
    p.add_argument("--wandb-run-name", type=str, default=None)
    return p.parse_args()


def train_one_fold(fold, config, args, device, wandb_run):

    pipeline = fold["pipeline"]
    split = {k: fold[k] for k in ("train", "validation", "test")}

    (
        train_dataset, validation_dataset, test_dataset,
        train_loader, validation_loader, test_loader,
        _, _,
    ) = pipeline.build_fold_loaders(fold["all_windows"], split, model_name=config.MODEL_NAME)

    subject_to_idx = None
    num_subjects = None
    if args.enable_adversarial:
        subject_to_idx = build_subject_to_idx(train_dataset)
        num_subjects = len(subject_to_idx)

    model = build_model(config.MODEL_NAME, config, num_subjects=num_subjects).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )
    contrastive_loss_fn = (
        SupervisedContrastiveLoss(temperature=args.contrastive_temperature)
        if args.enable_contrastive else None
    )

    best_val, best_state, stale, history = -1.0, None, 0, []

    for epoch in range(1, args.epochs + 1):

        progress = min(1.0, (epoch - 1) / max(1, args.epochs / 2))
        grl_lambda = config.ADVERSARIAL_LAMBDA_MAX * progress if subject_to_idx else 0.0

        tr = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True,
            grad_clip=args.grad_clip, augment=not args.no_augment,
            subject_to_idx=subject_to_idx, grl_lambda=grl_lambda,
            adversarial_weight=args.adversarial_weight,
            contrastive_loss_fn=contrastive_loss_fn,
            contrastive_weight=args.contrastive_weight,
            mixup_alpha=args.mixup_alpha,
            gate_balance_weight=args.gate_balance_weight,
            heartbeat_every=0,
        )
        va = run_epoch(
            model, validation_loader, criterion, optimizer, device,
            train=False, heartbeat_every=0,
        )

        scheduler.step(va["trial_f1"])
        history.append(va["trial_f1"])
        smooth = float(np.mean(history[-args.val_smooth_window:]))
        gc.collect()

        print(
            f"  [fold {fold['fold']:>2}][epoch {epoch:>3}/{args.epochs}] "
            f"train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} "
            f"val_trial_f1={va['trial_f1']:.4f} (smooth={smooth:.4f})",
            flush=True,
        )

        if smooth > best_val:
            best_val = smooth
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            print(f"  [fold {fold['fold']}] early stopping at epoch {epoch}.", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    te = run_epoch(model, test_loader, criterion, optimizer, device, train=False, heartbeat_every=0)

    print(
        f"  [fold {fold['fold']:>2}] TEST subjects={fold['test']} "
        f"trial_f1={te['trial_f1']:.4f} trial_acc={te['trial_accuracy']:.4f} "
        f"(best val smooth={best_val:.4f})",
        flush=True,
    )

    if wandb_run is not None:
        wandb_run.log({
            "fold": fold["fold"],
            "fold/test_trial_f1": te["trial_f1"],
            "fold/test_trial_accuracy": te["trial_accuracy"],
            "fold/best_val_trial_f1": best_val,
        })

    result = {
        "fold": fold["fold"],
        "train": fold["train"],
        "validation": fold["validation"],
        "test": fold["test"],
        "best_val_trial_f1": best_val,
        "test_trial_f1": te["trial_f1"],
        "test_trial_accuracy": te["trial_accuracy"],
        "probabilities": te["probabilities"],
        "labels": te["labels"],
        "subjects": te["subjects"],
        "trials": te["trials"],
    }

    del model, optimizer, scheduler, train_loader, validation_loader, test_loader
    del train_dataset, validation_dataset, test_dataset
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main():
    args = parse_args()

    config = Config()
    config.MODEL_NAME = "hybrid"
    config.PREPROCESSING_VERSION = "v2"
    config.HYBRID_VERSION = "v2"
    config.EEG_HANDCRAFTED_DIM = 256
    config.EMBEDDING_DIM = args.embedding_dim
    config.DROPOUT = args.dropout
    config.LEARNING_RATE = args.lr
    config.WEIGHT_DECAY = args.weight_decay
    config.EARLY_STOPPING_PATIENCE = args.patience
    config.NUM_WORKERS = args.num_workers
    config.SUBJECT_FEATURE_NORM = args.subject_norm
    config.EEG_NORM = args.eeg_norm
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(
        "cuda" if (config.DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    config.PIN_MEMORY = device.type == "cuda"

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project, name=args.wandb_run_name, config=vars(args)
        )

    print("=" * 80)
    print(f"HYBRID CNN-MLP v2 -- {args.protocol.upper()} CROSS-VALIDATION (seed {args.seed})")
    print("=" * 80, flush=True)

    t0 = time.time()
    pipeline = DataPipeline(config)
    all_windows = pipeline.prepare_all_subject_windows(
        cache_path=args.cache_path,
        max_subjects=args.max_subjects,
        trials_per_subject=args.trials_per_subject,
    )
    print(f"All-subject preprocessing: {time.time() - t0:.1f}s, {len(all_windows)} windows", flush=True)

    subject_ids = sorted({str(w["subject_id"]) for w in all_windows})
    if args.target == "arousal":
        for w in all_windows:
            w["label"] = int(w["raw_labels"][1] > 5.0)
    labels_all = np.asarray([w["label"] for w in all_windows])
    print(f"Target: {args.target} | positive fraction = {labels_all.mean():.3f}", flush=True)
    folds = build_folds(subject_ids, args)
    print(f"{len(subject_ids)} subjects, {len(folds)} folds", flush=True)

    results = []
    for fold in folds:
        fold["pipeline"] = pipeline
        fold["all_windows"] = all_windows
        print(
            f"\n{'=' * 80}\nFOLD {fold['fold']} | train={len(fold['train'])} "
            f"val={len(fold['validation'])} test={fold['test']}\n{'=' * 80}",
            flush=True,
        )
        t_fold = time.time()
        result = train_one_fold(fold, config, args, device, wandb_run)
        result["fold_seconds"] = time.time() - t_fold
        results.append(result)

    metrics = evaluate_hierarchical_predictions(
        probabilities=np.concatenate([r["probabilities"] for r in results]),
        labels=np.concatenate([r["labels"] for r in results]),
        subjects=np.concatenate([r["subjects"] for r in results]),
        trials=np.concatenate([r["trials"] for r in results]),
    )

    print("\n" + "=" * 80)
    print("OUT-OF-FOLD CROSS-SUBJECT EVALUATION (all folds pooled)")
    print("=" * 80)
    print(f"OOF Trial Accuracy    : {metrics['trial']['accuracy']:.4f}")
    print(f"OOF Trial F1 (macro)  : {metrics['trial']['f1']:.4f}")
    print(f"Subject Trial-F1      : {metrics['subject_f1_mean']:.4f} ± {metrics['subject_f1_std']:.4f}")
    print(f"Worst Subject F1      : {metrics['worst_subject_f1']:.4f}")
    subj_acc = np.asarray([m["accuracy"] for m in metrics["subject_metrics"]], dtype=float)
    print(f"Subject Accuracy (%)  : {100 * subj_acc.mean():.2f}/{100 * subj_acc.std(ddof=0):.2f} (mean/std)")
    fold_f1 =  [r["test_trial_f1"] for r in results]
    print(f"Per-fold test Trial-F1: mean={np.mean(fold_f1):.4f} std={np.std(fold_f1):.4f}")

    # Label-free per-subject centering: shift each subject's probabilities
    # so the median trial probability is 0.5 (assumes roughly balanced
    # valence per subject; uses no test labels).
    all_probs = np.concatenate([r["probabilities"] for r in results]).copy()
    all_subjects = np.concatenate([r["subjects"] for r in results])
    all_trials = np.concatenate([r["trials"] for r in results])
    for sid in np.unique(all_subjects):
        mask = all_subjects == sid
        trial_means = [
            all_probs[mask & (all_trials == t), 1].mean()
            for t in np.unique(all_trials[mask])
        ]
        shift = 0.5 - float(np.median(trial_means))
        all_probs[mask, 1] = np.clip(all_probs[mask, 1] + shift, 1e-6, 1 - 1e-6)
        all_probs[mask, 0] = 1.0 - all_probs[mask, 1]
    centered = evaluate_hierarchical_predictions(
        probabilities=all_probs,
        labels=np.concatenate([r["labels"] for r in results]),
        subjects=all_subjects,
        trials=all_trials,
    )
    print("-" * 80)
    print("WITH per-subject median centering (label-free post-processing):")
    print(f"OOF Trial Accuracy    : {centered['trial']['accuracy']:.4f}")
    print(f"OOF Trial F1 (macro)  : {centered['trial']['f1']:.4f}")
    print(f"Subject Trial-F1      : {centered['subject_f1_mean']:.4f} ± {centered['subject_f1_std']:.4f}")
    print(f"Worst Subject F1      : {centered['worst_subject_f1']:.4f}")
    print("=" * 80, flush=True)

    if wandb_run is not None:
        wandb_run.summary.update({
            "oof_centered_trial_f1": centered["trial"]["f1"],
            "oof_centered_trial_accuracy": centered["trial"]["accuracy"],
            "oof_centered_subject_f1_mean": centered["subject_f1_mean"],
            "oof_trial_f1": metrics["trial"]["f1"],
            "oof_trial_accuracy": metrics["trial"]["accuracy"],
            "oof_subject_f1_mean": metrics["subject_f1_mean"],
            "oof_subject_f1_std": metrics["subject_f1_std"],
            "oof_subject_acc_mean": float(subj_acc.mean()),
            "oof_subject_acc_std": float(subj_acc.std(ddof=0)),
            "oof_worst_subject_f1": metrics["worst_subject_f1"],
            "fold_f1_mean": float(np.mean(fold_f1)),
            "fold_f1_std": float(np.std(fold_f1)),
        })
        wandb_run.finish()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "args": vars(args),
        "oof": {
            "trial_accuracy": metrics["trial"]["accuracy"],
            "trial_f1": metrics["trial"]["f1"],
            "subject_f1_mean": metrics["subject_f1_mean"],
            "subject_f1_std": metrics["subject_f1_std"],
            "worst_subject_f1": metrics["worst_subject_f1"],
            "subject_metrics": metrics["subject_metrics"],
        },
        "per_fold": [
            {k: r[k] for k in ("fold", "train", "validation", "test",
                               "best_val_trial_f1", "test_trial_f1",
                               "test_trial_accuracy", "fold_seconds")}
            for r in results
        ],
    }
    path = out / f"{args.tag}_{args.protocol}_seed{args.seed}_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=float)
    print(f"Report saved -> {path}", flush=True)


if __name__ == "__main__":
    main()
