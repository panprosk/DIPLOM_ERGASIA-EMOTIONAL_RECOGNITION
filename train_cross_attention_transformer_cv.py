"""
train_cross_attention_transformer_cv.py

Proper subject-wise Cross-Validation harness for the Cross-Attention
Transformer (2ο μοντέλο της διπλωματικής), αντίστοιχο του
`train_saffaryazdi_cv.py` για το Saffaryazdi LSTM baseline.

ΓΙΑΤΙ ΥΠΑΡΧΕΙ ΑΥΤΟ ΤΟ SCRIPT
----------------------------
`train_cross_attention_transformer.py` τρέχει ΕΝΑ σταθερό 70/15/15
subject split (`DataPipeline.run()` -> `SubjectSplitter.train_val_test_split`).
Αυτό σημαίνει ότι κάθε αναφερόμενο Test Trial-F1 είναι η απόδοση σε
ΜΟΛΙΣ 6 άγνωστα subjects, πάνω σε ΕΝΑ τυχαίο partition -- πολύ
θορυβώδης και δυνητικά παραπλανητική εκτίμηση της πραγματικής
cross-subject generalization ικανότητας.

Αυτό εδώ το script κάνει σωστό πρωτόκολλο:

  --protocol loso   : κάθε ΕΝΑ από τα subjects γίνεται test ακριβώς
                       μία φορά (Leave-One-Subject-Out, όπως και ο
                       `SubjectSplitter.loso()` / το design doc).
  --protocol kfold  : k ομάδες από subjects, κάθε ομάδα γίνεται test
                       ακριβώς μία φορά (φθηνότερο από πλήρες LOSO,
                       καλή επιλογή για πρώτο feasibility check).
  --protocol single : το ίδιο ΕΝΑ 70/15/15 split με πριν (baseline
                       σύγκρισης, ίδιο seed/πρωτόκολλο).

Σε κάθε fold, 3-4 validation subjects (ξεχωριστά από τα test subjects
του fold) χρησιμοποιούνται ΜΟΝΟ για checkpoint selection/early
stopping -- ο test subject ΔΕΝ βλέπεται ποτέ κατά το training.

Οι out-of-fold (OOF) trial-level προβλέψεις όλων των folds
συγκεντρώνονται στο τέλος σε ΕΝΑ ενιαίο evaluation πάνω σε ΟΛΟ το
dataset (κάθε subject contribution προέρχεται αποκλειστικά από το
fold όπου ήταν test) -- αυτό δίνει μια αμερόληπτη εκτίμηση cross-
subject generalization, με πλήρη mean/std/worst-subject στατιστικά.

ΑΠΟΔΟΤΙΚΟΤΗΤΑ (CPU-only, 8GB RAM)
----------------------------------
Preprocessing/windowing/feature-extraction ΔΕΝ εξαρτάται από το
split (βλ. DataPipeline.prepare_all_subject_windows docstring) -- γι'
αυτό γίνεται ΜΙΑ φορά για όλα τα subjects (`--cache-path`), και κάθε
fold ξαναχρησιμοποιεί αυτά τα ήδη επεξεργασμένα windows, κάνοντας
μόνο tα φθηνά, fold-specific βήματα (feature scaling fit-on-train +
dataset/loader building) ξανά. Χωρίς αυτό, πλήρες 32-fold LOSO θα
σήμαινε ~32x επανάληψη μιας πολύωρης preprocessing διαδικασίας.

Χρήση
-----
Πλήρες 32-fold LOSO (αργό, πολλές ώρες/μέρες σε CPU):

    python train_cross_attention_transformer_cv.py --protocol loso \\
        --cache-path outputs/cat_cv_all_windows.pkl --epochs 15

Γρήγορο feasibility check (λίγα subjects, λίγα trials, λίγα epochs):

    python train_cross_attention_transformer_cv.py --protocol loso \\
        --max-subjects 8 --trials-per-subject 10 --epochs 3 \\
        --loso-val-size 2

k-fold εναλλακτική (φθηνότερη από πλήρες LOSO):

    python train_cross_attention_transformer_cv.py --protocol kfold \\
        --folds 4 --cache-path outputs/cat_cv_all_windows.pkl --epochs 15
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.config.config import Config
from src.pipeline.data_pipeline import DataPipeline
from src.models import build_model
from src.models.supervised_contrastive_loss import SupervisedContrastiveLoss
from src.utils.evaluation import evaluate_hierarchical_predictions
from src.split.subject_split import SubjectSplitter

from train_cross_attention_transformer import (
    augment_eeg,
    build_subject_to_idx,
    run_epoch,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser(
        description="Subject-wise Cross-Validation (LOSO / k-fold) για "
                    "το Cross-Attention Transformer."
    )

    parser.add_argument(
        "--protocol", choices=["loso", "kfold", "single"], default="kfold",
        help="loso = Leave-One-Subject-Out (ένα subject/fold), "
             "kfold = ομάδες από subjects/fold, "
             "single = ίδιο 70/15/15 split με το train_cross_attention_transformer.py."
    )
    parser.add_argument("--folds", type=int, default=5,
                        help="Αριθμός folds για --protocol kfold.")
    parser.add_argument("--loso-val-size", type=int, default=4,
                        help="Validation subjects ανά LOSO fold (checkpoint "
                             "selection/early stopping -- ποτέ ο test subject).")
    parser.add_argument("--max-subjects", type=int, default=None,
                        help="Χρησιμοποιεί μόνο τα πρώτα N subjects (sanity check).")
    parser.add_argument("--trials-per-subject", type=int, default=None,
                        help="Κόβει κάθε subject στα πρώτα N trials (sanity check).")
    parser.add_argument("--cache-path", type=str, default=None,
                        help="Pickle cache των ΟΛΩΝ των subjects, ήδη "
                             "preprocessed/windowed/feature-extracted (split-"
                             "independent). Αν υπάρχει, φορτώνεται αντί να "
                             "ξανατρέξει το (πολύωρο) preprocessing.")

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-adversarial", action="store_true")
    parser.add_argument("--adversarial-weight", type=float, default=0.30)
    parser.add_argument("--adversarial-lambda-max", type=float, default=None)
    parser.add_argument("--no-contrastive", action="store_true")
    parser.add_argument("--contrastive-weight", type=float, default=0.20)
    parser.add_argument("--contrastive-temperature", type=float, default=0.07)
    parser.add_argument("--attn-entropy-weight", type=float, default=0.05)
    parser.add_argument("--val-smooth-window", type=int, default=3)

    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--self-attn-layers", type=int, default=None)
    parser.add_argument("--cross-attn-layers", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--dim-feedforward", type=int, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--tag", type=str, default="cat_cv")
    parser.add_argument("--heartbeat-every", type=int, default=0,
                        help="0 = σιωπηλό training (χωρίς per-batch ETA logs, "
                             "πιο σύντομο output για πολλαπλά folds).")

    return parser.parse_args()


def build_folds(subject_ids: list[str], args) -> list[dict]:
    """
    Χτίζει τη λίστα των CV folds, καθένα με δικά του train/validation/test
    subject-id splits. Ίδιο βασικό rationale με
    `train_saffaryazdi_cv.py::build_folds`.
    """

    ids = sorted(subject_ids)

    if args.protocol == "loso":
        val_size = max(1, min(args.loso_val_size, len(ids) - 2))
        folds = []
        for index, test_subject in enumerate(ids):
            rest = [s for s in ids if s != test_subject]
            offset = index % len(rest)
            rotated = rest[offset:] + rest[:offset]
            validation = sorted(rotated[:val_size])
            train = sorted(rotated[val_size:])
            folds.append({
                "fold": index,
                "train": train,
                "validation": validation,
                "test": [test_subject],
            })
        return folds

    if args.protocol == "single":
        splitter = SubjectSplitter(random_seed=args.seed)
        split = splitter.train_val_test_split(ids)
        return [{"fold": 0, **split}]

    if args.folds < 3:
        raise ValueError("--protocol kfold requires --folds >= 3.")

    rng = random.Random(args.seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    groups = [shuffled[i::args.folds] for i in range(args.folds)]

    folds = []
    for index in range(args.folds):
        test = sorted(groups[index])
        validation = sorted(groups[(index + 1) % args.folds])
        train = sorted(
            subject
            for group_index, group in enumerate(groups)
            if group_index not in (index, (index + 1) % args.folds)
            for subject in group
        )
        folds.append({
            "fold": index,
            "train": train,
            "validation": validation,
            "test": test,
        })
    return folds


def train_one_fold(fold, config, args, device):
    """
    Εκπαιδεύει ένα πλήρες μοντέλο πάνω στο δοσμένο fold (ίδια λογική
    training loop με το main() του train_cross_attention_transformer.py)
    και επιστρέφει τις test-set out-of-fold trial-level προβλέψεις.
    """

    pipeline = fold["pipeline"]
    split = {"train": fold["train"], "validation": fold["validation"], "test": fold["test"]}

    (
        train_dataset, validation_dataset, test_dataset,
        train_loader, validation_loader, test_loader,
        train_labels, train_subject_ids,
    ) = pipeline.build_fold_loaders(fold["all_windows"], split, model_name=config.MODEL_NAME)

    subject_to_idx = None
    num_subjects = None
    if not args.no_adversarial:
        subject_to_idx = build_subject_to_idx(train_dataset)
        num_subjects = len(subject_to_idx)

    model = build_model(config.MODEL_NAME, config, num_subjects=num_subjects).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )

    contrastive_loss_fn = (
        None if args.no_contrastive
        else SupervisedContrastiveLoss(temperature=args.contrastive_temperature)
    )

    best_val_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    val_trial_f1_history = []

    for epoch in range(1, args.epochs + 1):

        progress = min(1.0, (epoch - 1) / max(1, args.epochs / 2))
        grl_lambda = config.ADVERSARIAL_LAMBDA_MAX * progress if subject_to_idx else 0.0

        train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, device,
            train=True, grad_clip=args.grad_clip, augment=not args.no_augment,
            subject_to_idx=subject_to_idx, grl_lambda=grl_lambda,
            adversarial_weight=args.adversarial_weight,
            contrastive_loss_fn=contrastive_loss_fn,
            contrastive_weight=args.contrastive_weight,
            attn_entropy_weight=args.attn_entropy_weight,
            heartbeat_every=args.heartbeat_every,
            heartbeat_label=f"fold {fold['fold']} train ",
        )

        val_metrics = run_epoch(
            model, validation_loader, criterion, optimizer, device, train=False,
            heartbeat_every=args.heartbeat_every, heartbeat_label=f"fold {fold['fold']} val ",
        )

        scheduler.step(val_metrics["trial_f1"])

        val_trial_f1_history.append(val_metrics["trial_f1"])
        smoothed_val_f1 = float(np.mean(val_trial_f1_history[-args.val_smooth_window:]))

        gc.collect()

        logger.info(
            f"  [fold {fold['fold']:>2}][epoch {epoch:>3}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_trial_f1={val_metrics['trial_f1']:.4f} "
            f"(smooth={smoothed_val_f1:.4f}) "
            f"val_worst_subj_f1={val_metrics['worst_subject_f1']:.4f}"
        )

        if smoothed_val_f1 > best_val_f1:
            best_val_f1 = smoothed_val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
            logger.info(f"  [fold {fold['fold']}] early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = run_epoch(
        model, test_loader, criterion, optimizer, device, train=False,
    )

    logger.info(
        f"  [fold {fold['fold']:>2}] TEST subjects={fold['test']} "
        f"trial_f1={test_metrics['trial_f1']:.4f} "
        f"trial_acc={test_metrics['trial_accuracy']:.4f} "
        f"(best val smoothed trial_f1={best_val_f1:.4f})"
    )

    del model, optimizer, scheduler, train_loader, validation_loader, test_loader
    del train_dataset, validation_dataset, test_dataset
    gc.collect()

    return {
        "fold": fold["fold"],
        "train": fold["train"],
        "validation": fold["validation"],
        "test": fold["test"],
        "best_val_trial_f1": best_val_f1,
        "test_trial_f1": test_metrics["trial_f1"],
        "test_trial_accuracy": test_metrics["trial_accuracy"],
        "trial_probabilities": test_metrics["trial_probabilities"],
        "trial_labels": test_metrics["trial_labels"],
        "trial_subjects": test_metrics["trial_subjects"],
        "trial_ids": test_metrics["trial_ids"],
    }


def main():

    args = parse_args()

    config = Config()
    config.MODEL_NAME = "cross_attention_transformer"

    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.num_workers is not None:
        config.NUM_WORKERS = args.num_workers
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
    if args.weight_decay is not None:
        config.WEIGHT_DECAY = args.weight_decay
    if args.dropout is not None:
        config.TRANSFORMER_DROPOUT = args.dropout
        config.DROPOUT = args.dropout
    if args.patience is not None:
        config.EARLY_STOPPING_PATIENCE = args.patience
    if args.adversarial_lambda_max is not None:
        config.ADVERSARIAL_LAMBDA_MAX = args.adversarial_lambda_max
    if args.d_model is not None:
        config.TRANSFORMER_D_MODEL = args.d_model
    if args.num_heads is not None:
        config.TRANSFORMER_NUM_HEADS = args.num_heads
    if args.self_attn_layers is not None:
        config.TRANSFORMER_SELF_ATTN_LAYERS = args.self_attn_layers
    if args.cross_attn_layers is not None:
        config.TRANSFORMER_CROSS_ATTN_LAYERS = args.cross_attn_layers
    if args.patch_size is not None:
        config.TRANSFORMER_PATCH_SIZE = args.patch_size
    if args.dim_feedforward is not None:
        config.TRANSFORMER_FF_DIM = args.dim_feedforward

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(
        "cuda" if (config.DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    config.PIN_MEMORY = (device.type == "cuda")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("CROSS-ATTENTION TRANSFORMER -- SUBJECT-WISE CROSS-VALIDATION", flush=True)
    print("=" * 80, flush=True)
    print(f"Protocol          : {args.protocol}", flush=True)
    print(f"Device            : {device}", flush=True)
    print(f"Epochs/fold       : {args.epochs}", flush=True)
    print(f"Max Subjects      : {args.max_subjects or 'ALL (32)'}", flush=True)
    print(f"Trials/Subject    : {args.trials_per_subject or 'ALL (40)'}", flush=True)
    print("=" * 80, flush=True)

    t0 = time.time()
    pipeline = DataPipeline(config)
    all_windows = pipeline.prepare_all_subject_windows(
        cache_path=args.cache_path,
        max_subjects=args.max_subjects,
        trials_per_subject=args.trials_per_subject,
    )
    print(f"\nAll-subject preprocessing finished in {time.time() - t0:.1f}s "
          f"({len(all_windows)} windows total)\n", flush=True)

    subject_ids = sorted({str(w["subject_id"]) for w in all_windows})
    print(f"Subjects available: {len(subject_ids)} -> {subject_ids}", flush=True)

    folds = build_folds(subject_ids, args)
    print(f"\nBuilt {len(folds)} fold(s) (protocol={args.protocol}).\n", flush=True)

    fold_results = []

    for fold in folds:

        fold["pipeline"] = pipeline
        fold["all_windows"] = all_windows

        print(
            f"\n{'=' * 80}\nFOLD {fold['fold']} | "
            f"train={len(fold['train'])} val={len(fold['validation'])} "
            f"test={fold['test']}\n{'=' * 80}",
            flush=True,
        )

        fold_t0 = time.time()
        result = train_one_fold(fold, config, args, device)
        result["fold_seconds"] = time.time() - fold_t0

        fold_results.append(result)

        print(
            f"Fold {fold['fold']} finished in {result['fold_seconds']/60:.1f} min -- "
            f"test_trial_f1={result['test_trial_f1']:.4f}",
            flush=True,
        )

    # ------------------------------------------------------------
    # Out-of-fold (OOF) aggregation: πλήρες, αμερόληπτο cross-subject
    # evaluation πάνω σε ΟΛΑ τα subjects μαζί (κάθε subject έχει
    # προβλέψεις αποκλειστικά από το fold όπου ήταν test).
    # ------------------------------------------------------------

    oof_probabilities = np.concatenate([r["trial_probabilities"] for r in fold_results])
    oof_labels = np.concatenate([r["trial_labels"] for r in fold_results])
    oof_subjects = np.concatenate([r["trial_subjects"] for r in fold_results])
    oof_trials = np.concatenate([r["trial_ids"] for r in fold_results])

    oof_metrics = evaluate_hierarchical_predictions(
        probabilities=oof_probabilities,
        labels=oof_labels,
        subjects=oof_subjects,
        trials=oof_trials,
    )

    print("\n" + "=" * 80, flush=True)
    print("OUT-OF-FOLD (OOF) CROSS-SUBJECT EVALUATION -- ALL FOLDS POOLED", flush=True)
    print("=" * 80, flush=True)
    print(f"Protocol              : {args.protocol} ({len(folds)} folds)", flush=True)
    print(f"Subjects evaluated    : {len(np.unique(oof_subjects))}", flush=True)
    print(f"OOF Trial Accuracy    : {oof_metrics['trial']['accuracy']:.4f}", flush=True)
    print(f"OOF Trial F1 (macro)  : {oof_metrics['trial']['f1']:.4f}", flush=True)
    print(f"Subject Trial-F1      : {oof_metrics['subject_f1_mean']:.4f} ± {oof_metrics['subject_f1_std']:.4f}", flush=True)
    print(f"Worst Subject F1      : {oof_metrics['worst_subject_f1']:.4f}", flush=True)
    print("Subject-wise Trial Metrics:", flush=True)
    for metrics in sorted(oof_metrics["subject_metrics"], key=lambda m: m["subject"]):
        print(
            f"  {metrics['subject']}: trials={metrics['n_trials']}, "
            f"accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['f1']:.4f}",
            flush=True,
        )
    print("=" * 80, flush=True)

    report = {
        "protocol": args.protocol,
        "num_folds": len(folds),
        "subjects": subject_ids,
        "args": vars(args),
        "per_fold": [
            {
                "fold": r["fold"],
                "train": r["train"],
                "validation": r["validation"],
                "test": r["test"],
                "best_val_trial_f1": r["best_val_trial_f1"],
                "test_trial_f1": r["test_trial_f1"],
                "test_trial_accuracy": r["test_trial_accuracy"],
                "fold_seconds": r["fold_seconds"],
            }
            for r in fold_results
        ],
        "oof": {
            "trial_accuracy": oof_metrics["trial"]["accuracy"],
            "trial_f1": oof_metrics["trial"]["f1"],
            "subject_f1_mean": oof_metrics["subject_f1_mean"],
            "subject_f1_std": oof_metrics["subject_f1_std"],
            "worst_subject_f1": oof_metrics["worst_subject_f1"],
            "subject_metrics": oof_metrics["subject_metrics"],
        },
    }

    report_path = output_dir / f"{args.tag}_{args.protocol}_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nFull CV report saved -> {report_path}", flush=True)


if __name__ == "__main__":
    main()
