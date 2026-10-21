"""Paper-faithful Saffaryazdi et al. (2022) DEAP reproduction.

This is deliberately separate from ``train_paper_lstm.py``.  It implements
the paper's signal-only adaptation: eight-second removal, 1-second
non-overlapping windows, EEG FFT band powers, EDA/PPG handcrafted features,
80->30 stacked LSTM branches, class-weighted training, and validation-fitted
late probability fusion.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config.config import Config
from src.models.saffaryazdi_lstm import SaffaryazdiTaskModel
from src.pipeline.saffaryazdi_pipeline import SaffaryazdiPipeline

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Saffaryazdi DEAP LSTM reproduction.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many non-improving epochs; 0 disables early stopping.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Cross-entropy label smoothing factor.",
    )
    parser.add_argument(
        "--scheduler-metric",
        choices=["loss", "selection"],
        default="loss",
        help="Monitor validation loss (min) or the selection score (max).",
    )
    parser.add_argument(
        "--pooling",
        choices=["last", "mean", "attention", "attention_residual"],
        default="last",
    )
    parser.add_argument(
        "--fusion",
        choices=["late", "learned", "gated", "cross_modal"],
        default="late",
    )
    parser.add_argument("--fusion-loss-weight", type=float, default=0.50)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--threshold-tuning", action="store_true")
    parser.add_argument(
        "--fusion-selection",
        choices=["subject_g", "macro_f1", "coarse_macro_f1", "fixed"],
        default="coarse_macro_f1",
        help="Validation criterion for late-fusion EEG probability weighting.",
    )
    parser.add_argument(
        "--threshold-selection",
        choices=["subject_g", "macro_f1"],
        default="macro_f1",
        help="Validation criterion used when --threshold-tuning is enabled.",
    )
    parser.add_argument(
        "--task-specific-modality-selection",
        action="store_true",
        help="Select EEG, physiology, or fused predictions per task using validation macro-F1.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["absolute", "log", "absolute_relative", "hann_absolute"],
        default="absolute",
    )
    parser.add_argument(
        "--eeg-normalization",
        choices=["global", "subject", "feature_global"],
        default="global",
        help="EEG normalization fitted globally on train subjects or per subject.",
    )
    parser.add_argument("--phys-mode", choices=["all", "eda", "ppg"], default="all")
    parser.add_argument("--modality", choices=["all", "eeg", "phys"], default="all")
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--eeg-band-mode", choices=["five", "four"], default="five")
    parser.add_argument(
        "--eeg-filter",
        choices=["none", "1-45"],
        default="none",
        help="Optional EEG band-pass filter applied after trial trimming.",
    )
    parser.add_argument("--drop-seconds", type=int, default=8)
    parser.add_argument("--filter-before-trim", action="store_true")
    parser.add_argument(
        "--temporal-encoder",
        choices=["lstm", "cnn_lstm", "spatial_cnn_lstm"],
        default="lstm",
    )
    parser.add_argument("--subject-adversarial-lambda", type=float, default=0.0)
    parser.add_argument(
        "--checkpoint-selection",
        choices=["subject_g", "macro_f1", "loss"],
        default="subject_g",
        help="Validation criterion used to select the task checkpoint.",
    )
    parser.add_argument(
        "--unweighted-loss",
        action="store_true",
        help="Use ordinary cross-entropy instead of training-label class weights.",
    )
    parser.add_argument("--hidden1", type=int, default=80)
    parser.add_argument("--hidden2", type=int, default=30)
    parser.add_argument("--cache-path", type=str, default=None)
    parser.add_argument("--output", type=str, default="outputs/saffaryazdi_lstm.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--wandb", action="store_true", help="Log this run to Weights & Biases.")
    parser.add_argument(
        "--wandb-project", type=str, default="thesis-emotion-recognition"
    )
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64), minlength=2).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError(f"Training labels contain an empty class: {counts.tolist()}")
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_loader(arrays: dict, task: str, batch_size: int, shuffle: bool, subject_map=None):
    labels = torch.from_numpy(arrays[task]).long()
    dataset = TensorDataset(
        torch.from_numpy(arrays["eeg"]).float(),
        torch.from_numpy(arrays["phys"]).float(),
        labels,
    )
    if subject_map is not None:
        subject_targets = torch.tensor(
            [subject_map.get(str(subject), -1) for subject in arrays["subjects"]],
            dtype=torch.long,
        )
        dataset = TensorDataset(*dataset.tensors, subject_targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def predict(model, loader, device, fusion_mode):
    model.eval()
    eeg_probs, phys_probs, fused_probs, labels = [], [], [], []
    for batch in loader:
        eeg, phys, target = batch[:3]
        if fusion_mode == "learned":
            eeg_logits, phys_logits, fused_logits = model.forward_learned(
                eeg.to(device), phys.to(device)
            )
            fused_probs.append(torch.softmax(fused_logits, dim=1).cpu().numpy())
        elif fusion_mode == "gated":
            eeg_logits, phys_logits, fused_logits = model.forward_gated(
                eeg.to(device), phys.to(device)
            )
            fused_probs.append(torch.softmax(fused_logits, dim=1).cpu().numpy())
        elif fusion_mode == "cross_modal":
            eeg_logits, phys_logits, fused_logits = model.forward_cross_modal(
                eeg.to(device), phys.to(device)
            )
            fused_probs.append(torch.softmax(fused_logits, dim=1).cpu().numpy())
        else:
            eeg_logits, phys_logits = model(eeg.to(device), phys.to(device))
        eeg_probs.append(torch.softmax(eeg_logits, dim=1).cpu().numpy())
        phys_probs.append(torch.softmax(phys_logits, dim=1).cpu().numpy())
        labels.append(target.numpy())
    return (np.concatenate(eeg_probs), np.concatenate(phys_probs),
            np.concatenate(fused_probs) if fused_probs else None,
            np.concatenate(labels))


def metrics(predictions, labels, subjects):
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    result = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "precision": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
    }
    subject_scores = []
    per_subject = {}
    for subject in sorted(set(subjects)):
        mask = np.asarray(subjects) == subject
        score = float(
            f1_score(labels[mask], predictions[mask], average="macro", zero_division=0)
        )
        per_subject[str(subject)] = score
        subject_scores.append(score)
    result.update(
        subject_f1_mean=float(np.mean(subject_scores)),
        subject_f1_std=float(np.std(subject_scores)),
        worst_subject_f1=float(np.min(subject_scores)),
        per_subject_f1=per_subject,
    )
    return result


def choose_fusion_weight(
    eeg_probs, phys_probs, labels, subjects, selection="subject_g"
):
    if selection == "fixed":
        alpha = 0.5
        fused = alpha * eeg_probs + (1.0 - alpha) * phys_probs
        return alpha, metrics(fused.argmax(axis=1), labels, subjects), None
    grid = (
        np.linspace(0.0, 1.0, 11)
        if selection == "coarse_macro_f1"
        else np.linspace(0.0, 1.0, 101)
    )
    best = None
    for alpha in grid:
        fused = alpha * eeg_probs + (1.0 - alpha) * phys_probs
        current = metrics(fused.argmax(axis=1), labels, subjects)
        subject_score = (
            0.5 * current["subject_f1_mean"]
            + 0.3 * current["worst_subject_f1"]
            - 0.2 * current["subject_f1_std"]
        )
        score = subject_score if selection == "subject_g" else current["macro_f1"]
        candidate = (score, -abs(alpha - 0.5), alpha, current, subject_score)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return best[2], best[3], best[4]


def choose_decision_threshold(
    probabilities, labels, subjects, selection="subject_g"
):
    best = None
    for threshold in np.linspace(0.40, 0.60, 21):
        predictions = (probabilities[:, 1] >= threshold).astype(np.int64)
        current = metrics(predictions, labels, subjects)
        subject_score = (
            0.5 * current["subject_f1_mean"]
            + 0.3 * current["worst_subject_f1"]
            - 0.2 * current["subject_f1_std"]
        )
        score = subject_score if selection == "subject_g" else current["macro_f1"]
        candidate = (score, -abs(threshold - 0.5), threshold, current, subject_score)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return best[2], best[3], best[4]


def choose_task_modality(
    val_eeg,
    val_phys,
    val_fused,
    val_labels,
    val_subjects,
    test_eeg,
    test_phys,
    test_fused,
):
    candidates = {
        "eeg": (
            metrics(val_eeg.argmax(axis=1), val_labels, val_subjects),
            val_eeg,
            test_eeg,
        ),
        "phys": (
            metrics(val_phys.argmax(axis=1), val_labels, val_subjects),
            val_phys,
            test_phys,
        ),
        "all": (
            metrics(val_fused.argmax(axis=1), val_labels, val_subjects),
            val_fused,
            test_fused,
        ),
    }
    selected = max(
        candidates,
        key=lambda name: (
            candidates[name][0]["macro_f1"],
            candidates[name][0]["subject_f1_mean"],
            {"all": 2, "eeg": 1, "phys": 0}[name],
        ),
    )
    return selected, candidates


def train_task(
    model, train_loader, val_loader, labels, val_subjects, task, args, device,
    wandb_run=None,
):
    if not 0.0 <= args.fusion_loss_weight <= 1.0:
        raise ValueError("--fusion-loss-weight must be between 0 and 1.")
    weights = None if args.unweighted_loss else class_weights(labels, device)
    criterion = nn.CrossEntropyLoss(
        weight=weights, label_smoothing=args.label_smoothing
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if args.scheduler_metric == "selection" else "min",
        factor=0.5,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    best_state = None
    best_val_loss = float("inf")
    best_selection_score = -float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for batch in train_loader:
            eeg, phys, target = batch[:3]
            subject_target = batch[3].to(device) if len(batch) > 3 else None
            eeg, phys, target = eeg.to(device), phys.to(device), target.to(device)
            optimizer.zero_grad()
            if args.subject_adversarial_lambda > 0:
                if subject_target is None or torch.any(subject_target < 0):
                    raise RuntimeError("Missing training subject labels for adversarial loss.")
                eeg_logits, phys_logits, subject_logits = model.forward_adversarial(
                    eeg, phys, args.subject_adversarial_lambda
                )
                emotion_loss = 0.5 * (
                    criterion(eeg_logits, target) + criterion(phys_logits, target)
                )
                subject_loss = nn.CrossEntropyLoss()(subject_logits, subject_target)
                loss = emotion_loss + args.subject_adversarial_lambda * subject_loss
            elif args.fusion in {"learned", "gated", "cross_modal"}:
                forward_fused = (
                    model.forward_learned
                    if args.fusion == "learned"
                    else (
                        model.forward_gated
                        if args.fusion == "gated"
                        else model.forward_cross_modal
                    )
                )
                eeg_logits, phys_logits, fused_logits = forward_fused(eeg, phys)
                branch_weight = (1.0 - args.fusion_loss_weight) / 2.0
                loss = (
                    branch_weight * criterion(eeg_logits, target)
                    + branch_weight * criterion(phys_logits, target)
                    + args.fusion_loss_weight * criterion(fused_logits, target)
                )
            else:
                eeg_logits, phys_logits = model(eeg, phys)
                loss = 0.5 * (
                    criterion(eeg_logits, target) + criterion(phys_logits, target)
                )
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item() * len(target)
            total_items += len(target)

        model.eval()
        val_loss = 0.0
        val_items = 0
        with torch.no_grad():
            for batch in val_loader:
                eeg, phys, target = batch[:3]
                eeg, phys, target = eeg.to(device), phys.to(device), target.to(device)
                if args.fusion in {"learned", "gated", "cross_modal"}:
                    forward_fused = (
                        model.forward_learned
                        if args.fusion == "learned"
                        else (
                            model.forward_gated
                            if args.fusion == "gated"
                            else model.forward_cross_modal
                        )
                    )
                    eeg_logits, phys_logits, fused_logits = forward_fused(eeg, phys)
                    branch_weight = (1.0 - args.fusion_loss_weight) / 2.0
                    loss = (
                        branch_weight * criterion(eeg_logits, target)
                        + branch_weight * criterion(phys_logits, target)
                        + args.fusion_loss_weight * criterion(fused_logits, target)
                    )
                else:
                    eeg_logits, phys_logits = model(eeg, phys)
                    loss = 0.5 * (
                        criterion(eeg_logits, target) + criterion(phys_logits, target)
                    )
                val_loss += loss.item() * len(target)
                val_items += len(target)
        val_loss /= max(1, val_items)
        if args.checkpoint_selection in {"subject_g", "macro_f1"}:
            val_eeg, val_phys, val_learned, val_labels = predict(
                model, val_loader, device, args.fusion
            )
            if args.fusion in {"learned", "gated", "cross_modal"}:
                val_predictions = val_learned.argmax(axis=1)
                selection_metrics = metrics(
                    val_predictions, val_labels, val_subjects
                )
                subject_score = (
                    0.5 * selection_metrics["subject_f1_mean"]
                    + 0.3 * selection_metrics["worst_subject_f1"]
                    - 0.2 * selection_metrics["subject_f1_std"]
                )
            elif args.modality == "eeg":
                selection_metrics = metrics(
                    val_eeg.argmax(axis=1), val_labels, val_subjects
                )
                subject_score = (
                    0.5 * selection_metrics["subject_f1_mean"]
                    + 0.3 * selection_metrics["worst_subject_f1"]
                    - 0.2 * selection_metrics["subject_f1_std"]
                )
            elif args.modality == "phys":
                selection_metrics = metrics(
                    val_phys.argmax(axis=1), val_labels, val_subjects
                )
                subject_score = (
                    0.5 * selection_metrics["subject_f1_mean"]
                    + 0.3 * selection_metrics["worst_subject_f1"]
                    - 0.2 * selection_metrics["subject_f1_std"]
                )
            else:
                _, selection_metrics, subject_score = choose_fusion_weight(
                    val_eeg,
                    val_phys,
                    val_labels,
                    val_subjects,
                    args.fusion_selection,
                )
            selection_score = (
                subject_score
                if args.checkpoint_selection == "subject_g"
                else selection_metrics["macro_f1"]
            )
        else:
            selection_score = -val_loss
        scheduler.step(
            selection_score if args.scheduler_metric == "selection" else val_loss
        )
        logger.info(
            "[%s][%03d/%03d] train_loss=%.4f val_loss=%.4f "
            "selection_score=%.4f lr=%.6f",
            task, epoch, args.epochs, total_loss / max(1, total_items),
            val_loss, selection_score, optimizer.param_groups[0]["lr"],
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    f"{task}/train_loss": total_loss / max(1, total_items),
                    f"{task}/val_loss": val_loss,
                    f"{task}/selection_score": selection_score,
                    f"{task}/learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                }
            )
        improved = selection_score > best_selection_score
        if not improved and selection_score == best_selection_score:
            improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_selection_score = selection_score
            epochs_without_improvement = 0
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if (
                args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                logger.info(
                    "[%s] early stopping at epoch %d; best_val_loss=%.4f",
                    task, epoch, best_val_loss,
                )
                break

    if best_state is None:
        raise RuntimeError(f"No checkpoint was produced for {task}.")
    model.load_state_dict(best_state)
    return model, [1.0, 1.0] if weights is None else weights.detach().cpu().tolist()


def main():
    args = parse_args()
    set_seed(args.seed)
    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging requested but wandb is not installed. "
                "Install it with: python -m pip install wandb"
            ) from exc
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
    config = Config()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    logger.info("Device: %s", device)
    pipeline = SaffaryazdiPipeline(
        config,
        seed=args.seed,
        feature_mode=args.feature_mode,
        eeg_normalization=args.eeg_normalization,
        phys_mode=args.phys_mode,
        modality=args.modality,
        window_seconds=args.window_seconds,
        eeg_band_mode=args.eeg_band_mode,
        eeg_filter=args.eeg_filter,
        drop_seconds=args.drop_seconds,
        filter_before_trim=args.filter_before_trim,
    )
    arrays = pipeline.run(cache_path=args.cache_path, max_subjects=args.max_subjects)

    subjects = {
        split: arrays[split]["subjects"].astype(str)
        for split in ("train", "validation", "test")
    }
    results = {"protocol": {
        "drop_seconds": args.drop_seconds,
        "window_seconds": args.window_seconds,
        "overlap": 0,
        "feature_mode": args.feature_mode,
        "eeg_normalization": args.eeg_normalization,
        "eeg_normalization_protocol": (
            "training-subject statistics; global training fallback for unseen subjects"
            if args.eeg_normalization == "subject"
            else "FFT band-power features normalized with training-feature statistics"
            if args.eeg_normalization == "feature_global"
            else "global training statistics"
        ),
        "phys_mode": args.phys_mode,
        "modality": args.modality,
        "checkpoint_selection": args.checkpoint_selection,
        "fusion_selection": args.fusion_selection,
        "threshold_selection": args.threshold_selection,
        "window_seconds": args.window_seconds,
        "eeg_band_mode": args.eeg_band_mode,
        "eeg_filter": args.eeg_filter,
        "filter_before_trim": args.filter_before_trim,
        "eeg_shape": list(arrays["train"]["eeg"].shape[1:]),
        "phys_shape": list(arrays["train"]["phys"].shape[1:]),
        "class_weight_source": (
            "disabled (--unweighted-loss)"
            if args.unweighted_loss
            else "training subjects only"
        ),
    }}
    checkpoints = {}
    subject_map = {
        str(subject): index
        for index, subject in enumerate(sorted(set(subjects["train"])))
    }

    for task in ("valence", "arousal"):
        train_loader = make_loader(
            arrays["train"], task, args.batch_size, True, subject_map
        )
        val_loader = make_loader(
            arrays["validation"], task, args.batch_size, False, subject_map
        )
        test_loader = make_loader(
            arrays["test"], task, args.batch_size, False, subject_map
        )
        model = SaffaryazdiTaskModel(
            input_dim=arrays["train"]["eeg"].shape[-1],
            phys_dim=arrays["train"]["phys"].shape[-1],
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            dropout=args.dropout,
            pooling=args.pooling,
            fusion_mode=args.fusion,
            bidirectional=args.bidirectional,
            temporal_encoder=args.temporal_encoder,
            subject_classes=(
                len(subject_map) if args.subject_adversarial_lambda > 0 else 0
            ),
        ).to(device)
        model, weights = train_task(
            model,
            train_loader,
            val_loader,
            arrays["train"][task],
            subjects["validation"],
            task,
            args,
            device,
            wandb_run,
        )
        val_eeg, val_phys, val_learned, val_labels = predict(
            model, val_loader, device, args.fusion
        )
        test_eeg, test_phys, test_learned, test_labels = predict(
            model, test_loader, device, args.fusion
        )
        if args.fusion in {"learned", "gated", "cross_modal"}:
            alpha, val_result, val_g = None, metrics(
                val_learned.argmax(axis=1), val_labels, subjects["validation"]
            ), None
            val_fused = val_learned
            test_fused = test_learned
        elif args.modality == "eeg":
            alpha = 1.0
            val_fused = val_eeg
            test_fused = test_eeg
            val_result = metrics(
                val_fused.argmax(axis=1), val_labels, subjects["validation"]
            )
            val_g = None
        elif args.modality == "phys":
            alpha = 0.0
            val_fused = val_phys
            test_fused = test_phys
            val_result = metrics(
                val_fused.argmax(axis=1), val_labels, subjects["validation"]
            )
            val_g = None
        else:
            alpha, val_result, val_g = choose_fusion_weight(
                val_eeg,
                val_phys,
                val_labels,
                subjects["validation"],
                args.fusion_selection,
            )
            val_fused = alpha * val_eeg + (1.0 - alpha) * val_phys
            test_fused = alpha * test_eeg + (1.0 - alpha) * test_phys
        modality_candidates = None
        selected_modality = args.modality
        if (
            args.task_specific_modality_selection
            and args.fusion == "late"
            and args.modality == "all"
        ):
            selected_modality, modality_candidates = choose_task_modality(
                val_eeg,
                val_phys,
                val_fused,
                val_labels,
                subjects["validation"],
                test_eeg,
                test_phys,
                test_fused,
            )
            val_result, val_fused, test_fused = modality_candidates[selected_modality]
            alpha = (
                1.0
                if selected_modality == "eeg"
                else 0.0
                if selected_modality == "phys"
                else alpha
            )
        if args.threshold_tuning:
            threshold, threshold_result, threshold_g = choose_decision_threshold(
                val_fused,
                val_labels,
                subjects["validation"],
                args.threshold_selection,
            )
            test_fusion_predictions = (
                test_fused[:, 1] >= threshold
            ).astype(np.int64)
            fusion_result = metrics(
                test_fusion_predictions, test_labels, subjects["test"]
            )
            validation_result = threshold_result
        else:
            threshold, threshold_g = 0.50, None
            fusion_result = metrics(
                test_fused.argmax(axis=1), test_labels, subjects["test"]
            )
            validation_result = val_result
        results[task] = {
            "class_weights": weights,
            "fusion_eeg_weight": alpha,
            "fusion_physiology_weight": None if alpha is None else 1.0 - alpha,
            "selected_modality": selected_modality,
            "validation_modality_candidates": (
                None
                if modality_candidates is None
                else {
                    name: candidate[0]
                    for name, candidate in modality_candidates.items()
                }
            ),
            "decision_threshold": threshold,
            "validation": validation_result,
            "validation_G": threshold_g if args.threshold_tuning else val_g,
            "test_eeg": metrics(test_eeg.argmax(axis=1), test_labels, subjects["test"]),
            "test_physiology": metrics(test_phys.argmax(axis=1), test_labels, subjects["test"]),
            "test_fusion": fusion_result,
        }
        checkpoints[task] = model.state_dict()
        fusion_label = "learned" if alpha is None else f"{alpha:.2f}"
        logger.info(
            "%s fusion=%s | validation F1=%.4f | test F1=%.4f",
            task, fusion_label, validation_result["macro_f1"],
            results[task]["test_fusion"]["macro_f1"],
        )

        test_metrics = results[task]["test_fusion"]
        if wandb_run is not None:
            wandb_run.summary[f"{task}/test_accuracy"] = test_metrics["accuracy"]
            wandb_run.summary[f"{task}/test_macro_f1"] = test_metrics["macro_f1"]
            wandb_run.summary[f"{task}/test_subject_f1_mean"] = (
                test_metrics["subject_f1_mean"]
            )
        logger.info(
            "%s TEST | accuracy=%.4f | precision=%.4f | recall=%.4f | "
            "macro_f1=%.4f | subject_f1_mean=%.4f | subject_f1_std=%.4f | "
            "worst_subject_f1=%.4f",
            task,
            test_metrics["accuracy"],
            test_metrics["precision"],
            test_metrics["recall"],
            test_metrics["macro_f1"],
            test_metrics["subject_f1_mean"],
            test_metrics["subject_f1_std"],
            test_metrics["worst_subject_f1"],
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dicts": checkpoints, "args": vars(args), "results": results},
        output_path,
    )
    metrics_path = output_path.with_suffix(".json")
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    mean_f1 = np.mean(
        [results[task]["test_fusion"]["macro_f1"] for task in ("valence", "arousal")]
    )
    mean_accuracy = np.mean(
        [results[task]["test_fusion"]["accuracy"] for task in ("valence", "arousal")]
    )
    logger.info(
        "FINAL SUMMARY | mean_accuracy=%.4f | mean_f1=%.4f",
        mean_accuracy,
        mean_f1,
    )
    logger.info("Checkpoint saved to: %s", output_path)
    logger.info("Metrics saved to: %s", metrics_path)
    if wandb_run is not None:
        wandb_run.summary["mean_accuracy"] = mean_accuracy
        wandb_run.summary["mean_f1"] = mean_f1
        wandb_run.finish()


if __name__ == "__main__":
    main()
