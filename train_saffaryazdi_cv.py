"""Cross-subject cross-validation harness for the Saffaryazdi LSTM baseline.

This drives the paper-faithful Saffaryazdi reproduction
(``src/models/saffaryazdi_lstm.py`` + ``src/pipeline/saffaryazdi_pipeline.py``)
through a proper subject-wise cross-validation protocol instead of the single
70/15/15 split used by ``train_saffaryazdi_lstm.py``.  On top of the original
recipe it adds:

* regularization (dropout, weight decay, gradient clipping),
* label smoothing,
* learning-rate scheduling driven by the subject-level generalization score,
* early stopping on the generalization score,
* optional subject-adversarial (DANN / gradient reversal) alignment,
* optional CORAL subject alignment on the branch embeddings,
* seed ensembling inside every fold,
* out-of-fold (OOF) aggregation across all subjects.

Every subject is held out exactly once, so the reported subject-level F1 is
estimated on genuinely unseen subjects and does not depend on a single
4-subject validation set.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.config.config import Config
from src.models.saffaryazdi_lstm import SaffaryazdiTaskModel, gradient_reverse
from src.pipeline.saffaryazdi_pipeline import SaffaryazdiPipeline
from src.split.subject_split import SubjectSplitter
from train_saffaryazdi_lstm import (
    choose_decision_threshold,
    choose_fusion_weight,
    class_weights,
    make_loader,
    metrics,
    predict,
    set_seed,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Saffaryazdi cross-subject CV (k-fold / LOSO)."
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=15,
        help="Stop after this many non-improving epochs; 0 disables.",
    )
    parser.add_argument(
        "--scheduler-metric",
        choices=["loss", "selection"],
        default="selection",
        help="Monitor validation loss (min) or the subject-G selection score (max).",
    )
    parser.add_argument(
        "--pooling", choices=["last", "mean", "attention", "attention_residual"],
        default="last",
    )
    parser.add_argument(
        "--fusion", choices=["late", "learned", "gated", "cross_modal"], default="late"
    )
    parser.add_argument(
        "--fusion-selection",
        choices=["subject_g", "macro_f1", "coarse_macro_f1", "fixed"],
        default="subject_g",
    )
    parser.add_argument("--fusion-loss-weight", type=float, default=0.50)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument(
        "--feature-mode",
        choices=["absolute", "log", "absolute_relative", "hann_absolute"],
        default="absolute",
    )
    parser.add_argument(
        "--eeg-normalization",
        choices=["global", "subject", "feature_global"],
        default="global",
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
        "--temporal-encoder", choices=["lstm", "cnn_lstm", "spatial_cnn_lstm"],
        default="lstm",
    )
    parser.add_argument("--subject-adversarial-lambda", type=float, default=0.0)
    parser.add_argument(
        "--coral-weight",
        type=float,
        default=0.0,
        help="CORAL alignment weight on EEG/phys branch embeddings (train only).",
    )
    parser.add_argument(
        "--checkpoint-selection",
        choices=["subject_g", "macro_f1", "loss"],
        default="subject_g",
    )
    parser.add_argument("--threshold-tuning", action="store_true")
    parser.add_argument(
        "--temperature-scaling",
        action="store_true",
        help="Fit a validation-only temperature before applying test probabilities.",
    )
    parser.add_argument(
        "--focal-loss",
        action="store_true",
        help="Use focal loss instead of cross-entropy for the task objective.",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--unweighted-loss",
        action="store_true",
        help="Use ordinary cross-entropy instead of class-weighted loss.",
    )
    parser.add_argument("--hidden1", type=int, default=80)
    parser.add_argument("--hidden2", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ensemble-seeds",
        type=str,
        default="",
        help="Comma-separated extra seeds averaged inside each fold (e.g. 42,7,2026).",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--protocol",
        choices=["kfold", "loso", "single"],
        default="kfold",
        help="k-fold subject CV, leave-one-subject-out, or the original 70/15/15 split.",
    )
    parser.add_argument(
        "--loso-val-size",
        type=int,
        default=4,
        help="Held-out validation subjects per LOSO fold.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="valence,arousal",
        help="Comma-separated tasks to run.",
    )
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--tag", type=str, default="cv")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--quiet-epochs", action="store_true")
    parser.add_argument(
        "--adaptation",
        choices=["none", "supervised", "unsupervised"],
        default="none",
        help="Optional test-subject adaptation after zero-shot training.",
    )
    parser.add_argument("--adaptation-seconds", type=int, default=30)
    parser.add_argument("--adaptation-epochs", type=int, default=5)
    parser.add_argument("--adaptation-lr", type=float, default=1e-4)
    parser.add_argument(
        "--adaptation-weight-decay",
        type=float,
        default=0.0,
        help="L2 regularization during target-subject adaptation.",
    )
    parser.add_argument("--wandb", action="store_true", help="Log the CV run to W&B.")
    parser.add_argument(
        "--wandb-project", type=str, default="thesis-emotion-recognition"
    )
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--adaptation-scope",
        choices=["classifier", "full"],
        default="classifier",
        help="Parameters updated during test-subject adaptation.",
    )
    return parser.parse_args()


def generalization_score(result: dict) -> float:
    return float(
        0.5 * result["subject_f1_mean"]
        + 0.3 * result["worst_subject_f1"]
        - 0.2 * result["subject_f1_std"]
    )


def build_folds(subject_ids, args):
    ids = sorted(subject_ids)
    if args.protocol == "loso":
        val_size = max(1, min(args.loso_val_size, len(ids) - 2))
        for index, test_subject in enumerate(ids):
            rest = [s for s in ids if s != test_subject]
            offset = index % len(rest)
            rotated = rest[offset:] + rest[:offset]
            validation = sorted(rotated[:val_size])
            train = sorted(rotated[val_size:])
            yield {
                "fold": index,
                "train": train,
                "validation": validation,
                "test": [test_subject],
            }
        return
    if args.protocol == "single":
        splitter = SubjectSplitter(random_seed=args.seed)
        split = splitter.train_val_test_split(ids)
        yield {"fold": 0, **split}
        return
    rng = random.Random(args.seed)
    if args.folds < 3:
        raise ValueError("k-fold CV requires at least 3 folds.")
    shuffled = ids[:]
    rng.shuffle(shuffled)
    groups = [shuffled[i::args.folds] for i in range(args.folds)]
    for index in range(args.folds):
        test = sorted(groups[index])
        validation = sorted(groups[(index + 1) % args.folds])
        train = sorted(
            subject
            for group_index, group in enumerate(groups)
            if group_index not in (index, (index + 1) % args.folds)
            for subject in group
        )
        yield {
            "fold": index,
            "train": train,
            "validation": validation,
            "test": test,
        }


def _pairwise_coral(representations: torch.Tensor, subjects: torch.Tensor):
    groups: dict[int, list[int]] = {}
    for position, subject in enumerate(subjects.tolist()):
        groups.setdefault(int(subject), []).append(position)
    covariances = []
    for positions in groups.values():
        if len(positions) < 2:
            continue
        index = torch.tensor(positions, device=representations.device, dtype=torch.long)
        samples = representations.index_select(0, index)
        centered = samples - samples.mean(dim=0, keepdim=True)
        covariances.append(centered.transpose(0, 1).matmul(centered) / (samples.shape[0] - 1))
    if len(covariances) < 2:
        return None
    losses = []
    for first in range(len(covariances)):
        for second in range(first + 1, len(covariances)):
            losses.append(((covariances[first] - covariances[second]) ** 2).mean())
    return torch.stack(losses).mean()


def _task_criterion(args, train_labels, device):
    weights = None if args.unweighted_loss else class_weights(train_labels, device)
    if args.focal_loss and args.label_smoothing:
        raise ValueError("--focal-loss cannot be combined with --label-smoothing.")
    if args.focal_loss:
        class FocalLoss(nn.Module):
            def __init__(self, class_weights, gamma):
                super().__init__()
                self.class_weights = class_weights
                self.gamma = gamma

            def forward(self, logits, target):
                log_prob = torch.log_softmax(logits, dim=1)
                prob = log_prob.exp()
                target_log_prob = log_prob.gather(1, target.unsqueeze(1)).squeeze(1)
                target_prob = prob.gather(1, target.unsqueeze(1)).squeeze(1)
                loss = -(1.0 - target_prob).pow(self.gamma) * target_log_prob
                if self.class_weights is not None:
                    loss = loss * self.class_weights[target]
                return loss.mean()

        criterion = FocalLoss(weights, args.focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(
            weight=weights, label_smoothing=args.label_smoothing
        )
    return criterion, weights


def _forward_components(model, eeg, phys, include_fusion):
    eeg_representation = model.eeg.encode(eeg)
    phys_representation = model.phys.encode(phys)
    eeg_logits = model.eeg.classifier(eeg_representation)
    phys_logits = model.phys.classifier(phys_representation)
    fused_logits = None
    if include_fusion:
        fused_logits = model.fusion_classifier(
            torch.cat([eeg_representation, phys_representation], dim=1)
        )
    return eeg_representation, phys_representation, eeg_logits, phys_logits, fused_logits


def _forward_logits(model, eeg, phys, fusion):
    if fusion == "learned":
        return model.forward_learned(eeg, phys)
    if fusion == "gated":
        return model.forward_gated(eeg, phys)
    if fusion == "cross_modal":
        return model.forward_cross_modal(eeg, phys)
    eeg_logits, phys_logits = model(eeg, phys)
    return eeg_logits, phys_logits, None


def _task_loss(criterion, eeg_logits, phys_logits, fused_logits, target, fusion_loss_weight):
    if fused_logits is not None:
        branch_weight = (1.0 - fusion_loss_weight) / 2.0
        return (
            branch_weight * criterion(eeg_logits, target)
            + branch_weight * criterion(phys_logits, target)
            + fusion_loss_weight * criterion(fused_logits, target)
        )
    return 0.5 * (criterion(eeg_logits, target) + criterion(phys_logits, target))


def train_fold_task(
    model,
    train_loader,
    val_loader,
    train_labels,
    val_subjects,
    args,
    device,
):
    criterion, weights = _task_criterion(args, train_labels, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    if args.scheduler_metric == "selection":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=args.lr_patience, min_lr=args.min_lr
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=args.lr_patience, min_lr=args.min_lr
        )

    use_alignment = args.subject_adversarial_lambda > 0 or args.coral_weight > 0
    if use_alignment and args.fusion not in {"late", "learned"}:
        raise ValueError(
            "Domain alignment is only supported for late/learned fusion; "
            f"got fusion={args.fusion}."
        )
    include_fusion = args.fusion == "learned"

    best_state = None
    best_val_loss = float("inf")
    best_selection = -float("inf")
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
            if use_alignment:
                eeg_rep, phys_rep, eeg_logits, phys_logits, fused_logits = (
                    _forward_components(model, eeg, phys, include_fusion)
                )
            else:
                eeg_logits, phys_logits, fused_logits = _forward_logits(
                    model, eeg, phys, args.fusion
                )
            loss = _task_loss(
                criterion,
                eeg_logits,
                phys_logits,
                fused_logits,
                target,
                args.fusion_loss_weight,
            )
            if use_alignment:
                if args.coral_weight > 0 and subject_target is not None:
                    coral_terms = [
                        _pairwise_coral(rep, subject_target)
                        for rep in (eeg_rep, phys_rep)
                    ]
                    coral_terms = [term for term in coral_terms if term is not None]
                    if coral_terms:
                        loss = loss + args.coral_weight * torch.stack(coral_terms).mean()
                if args.subject_adversarial_lambda > 0:
                    if subject_target is None or torch.any(subject_target < 0):
                        raise RuntimeError("Missing subject labels for adversarial loss.")
                    subject_input = torch.cat([eeg_rep, phys_rep], dim=1)
                    subject_logits = model.subject_classifier(
                        gradient_reverse(subject_input, args.subject_adversarial_lambda)
                    )
                    subject_loss = nn.CrossEntropyLoss()(subject_logits, subject_target)
                    loss = loss + args.subject_adversarial_lambda * subject_loss
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
                eeg_logits, phys_logits, fused_logits = _forward_logits(
                    model, eeg, phys, args.fusion
                )
                val_loss += _task_loss(
                    criterion,
                    eeg_logits,
                    phys_logits,
                    fused_logits,
                    target,
                    args.fusion_loss_weight,
                ).item() * len(target)
                val_items += len(target)
        val_loss /= max(1, val_items)

        val_eeg, val_phys, val_fused, val_labels = predict(
            model, val_loader, device, args.fusion
        )
        if val_fused is not None:
            selection_metrics = metrics(
                val_fused.argmax(axis=1), val_labels, val_subjects
            )
            subject_score = generalization_score(selection_metrics)
        elif args.modality == "eeg":
            selection_metrics = metrics(val_eeg.argmax(axis=1), val_labels, val_subjects)
            subject_score = generalization_score(selection_metrics)
        elif args.modality == "phys":
            selection_metrics = metrics(val_phys.argmax(axis=1), val_labels, val_subjects)
            subject_score = generalization_score(selection_metrics)
        else:
            _, selection_metrics, subject_score = choose_fusion_weight(
                val_eeg,
                val_phys,
                val_labels,
                val_subjects,
                args.fusion_selection,
            )

        if args.checkpoint_selection == "subject_g":
            selection_score = subject_score
        elif args.checkpoint_selection == "macro_f1":
            selection_score = selection_metrics["macro_f1"]
        else:
            selection_score = -val_loss

        scheduler.step(selection_score if args.scheduler_metric == "selection" else val_loss)

        if not args.quiet_epochs:
            logger.info(
                "[%03d/%03d] train_loss=%.4f val_loss=%.4f selection=%.4f lr=%.6f",
                epoch,
                args.epochs,
                total_loss / max(1, total_items),
                val_loss,
                selection_score,
                optimizer.param_groups[0]["lr"],
            )

        improved = selection_score > best_selection
        if not improved and selection_score == best_selection:
            improved = val_loss < best_val_loss
        if improved:
            best_selection = selection_score
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1
            if (
                args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                if not args.quiet_epochs:
                    logger.info("[early stop] epoch=%d best=%.4f", epoch, best_selection)
                break

    if best_state is None:
        raise RuntimeError("No checkpoint was produced.")
    model.load_state_dict(best_state)
    return model, weights


def fuse_test_probabilities(
    model,
    val_loader,
    test_loader,
    val_subjects,
    args,
    device,
):
    val_eeg, val_phys, val_fused, val_labels = predict(
        model, val_loader, device, args.fusion
    )
    test_eeg, test_phys, test_fused, test_labels = predict(
        model, test_loader, device, args.fusion
    )
    if val_fused is not None:
        alpha = None
        val_result = metrics(val_fused.argmax(axis=1), val_labels, val_subjects)
        test_probs = test_fused
    elif args.modality == "eeg":
        alpha = 1.0
        test_probs = test_eeg
        val_result = metrics(val_eeg.argmax(axis=1), val_labels, val_subjects)
    elif args.modality == "phys":
        alpha = 0.0
        test_probs = test_phys
        val_result = metrics(val_phys.argmax(axis=1), val_labels, val_subjects)
    else:
        alpha, val_result, _ = choose_fusion_weight(
            val_eeg,
            val_phys,
            val_labels,
            val_subjects,
            args.fusion_selection,
        )
        test_probs = alpha * test_eeg + (1.0 - alpha) * test_phys
    val_probs = (
        val_fused
        if val_fused is not None
        else alpha * val_eeg + (1.0 - alpha) * val_phys
    )
    return test_probs, test_labels, val_result, alpha, val_probs, val_labels


def fit_temperature(probabilities, labels):
    """Fit temperature by validation negative log likelihood only."""
    logits = np.log(np.clip(probabilities, 1e-8, 1.0))
    best = (float("inf"), 1.0)
    for temperature in np.exp(np.linspace(np.log(0.5), np.log(5.0), 61)):
        scaled = logits / temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        log_probs = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
        nll = float(-log_probs[np.arange(len(labels)), labels].mean())
        if nll < best[0]:
            best = (nll, float(temperature))
    return best[1]


def apply_temperature(probabilities, temperature):
    logits = np.log(np.clip(probabilities, 1e-8, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def create_model(arrays, subject_map, args, device):
    return SaffaryazdiTaskModel(
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


def adapt_model(model, arrays, task, args, device):
    """Adapt on one held-out trial, then evaluate the remaining trials."""
    if args.adaptation == "none":
        return model, arrays["test"]
    steps = max(1, min(int(args.adaptation_seconds / args.window_seconds), arrays["test"]["eeg"].shape[1]))
    if len(arrays["test"]["eeg"]) < 2:
        raise ValueError("Adaptation requires at least two held-out trials.")
    calibration = {
        key: value[:1].copy() if isinstance(value, np.ndarray) else value
        for key, value in arrays["test"].items()
    }
    calibration["eeg"] = calibration["eeg"][:, :steps]
    calibration["phys"] = calibration["phys"][:, :steps]
    evaluation = {
        key: value[1:].copy() if isinstance(value, np.ndarray) else value
        for key, value in arrays["test"].items()
    }
    calibration_loader = make_loader(calibration, task, 1, False)
    if args.adaptation_scope == "classifier":
        parameters = list(model.eeg.classifier.parameters()) + list(
            model.phys.classifier.parameters()
        )
        for module in (model.eeg, model.phys):
            for parameter in module.parameters():
                parameter.requires_grad = False
        for parameter in parameters:
            parameter.requires_grad = True
        if model.fusion_classifier is not None:
            parameters += list(model.fusion_classifier.parameters())
        if model.gated_classifier is not None:
            parameters += list(model.gated_classifier.parameters())
        if model.cross_classifier is not None:
            parameters += list(model.cross_classifier.parameters())
    else:
        parameters = list(model.parameters())
    initial_parameters = {
        id(parameter): parameter.detach().clone()
        for parameter in parameters
    }
    optimizer = torch.optim.Adam(parameters, lr=args.adaptation_lr)
    model.train()
    for _ in range(args.adaptation_epochs):
        for eeg, phys, target in calibration_loader:
            eeg, phys = eeg.to(device), phys.to(device)
            optimizer.zero_grad()
            eeg_logits, phys_logits, fused_logits = _forward_logits(
                model, eeg, phys, args.fusion
            )
            if args.adaptation == "supervised":
                target = target.to(device)
                logits = fused_logits if fused_logits is not None else (
                    eeg_logits + phys_logits
                ) / 2.0
                loss = nn.CrossEntropyLoss()(logits, target)
            else:
                logits = fused_logits if fused_logits is not None else (
                    eeg_logits + phys_logits
                ) / 2.0
                probabilities = torch.softmax(logits, dim=1)
                loss = -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum(dim=1).mean()
            if args.adaptation_weight_decay > 0:
                anchor_penalty = torch.stack(
                    [
                        (parameter - initial_parameters[id(parameter)]).pow(2).mean()
                        for parameter in parameters
                    ]
                ).mean()
                loss = loss + args.adaptation_weight_decay * anchor_penalty
            loss.backward()
            optimizer.step()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = True
    return model, evaluation


def main():
    args = parse_args()
    config = Config()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    logger.info("Device: %s", device)
    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging requested but wandb is not installed."
            ) from exc
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    seeds = [args.seed]
    if args.ensemble_seeds.strip():
        seeds = sorted({int(part) for part in args.ensemble_seeds.split(",") if part.strip()})

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
    subject_ids = pipeline.subject_ids()
    if args.max_subjects is not None:
        subject_ids = subject_ids[: args.max_subjects]
    folds = list(build_folds(subject_ids, args))
    logger.info(
        "Protocol=%s folds=%d subjects=%d seeds=%s tasks=%s",
        args.protocol,
        len(folds),
        len(subject_ids),
        seeds,
        tasks,
    )

    oof = {
        task: {
            "probs": [],
            "labels": [],
            "subjects": [],
            "trials": [],
            "thresholds": [],
        }
        for task in tasks
    }
    adaptation_oof = {
        task: {
            "probs": [],
            "labels": [],
            "subjects": [],
            "trials": [],
            "thresholds": [],
        }
        for task in tasks
    }
    fold_records = []
    started = time.time()

    for fold in folds:
        split = {
            "train": fold["train"],
            "validation": fold["validation"],
            "test": fold["test"],
        }
        arrays = pipeline.run(splits=split)
        train_subjects = sorted(set(arrays["train"]["subjects"].astype(str)))
        subject_map = {subject: index for index, subject in enumerate(train_subjects)}
        fold_summary = {"fold": fold["fold"], "tasks": {}}
        logger.info(
            "=== fold %d | train=%d val=%d test=%d ===",
            fold["fold"],
            len(fold["train"]),
            len(fold["validation"]),
            len(fold["test"]),
        )

        for task in tasks:
            train_loader = make_loader(
                arrays["train"], task, args.batch_size, True, subject_map
            )
            val_loader = make_loader(
                arrays["validation"], task, args.batch_size, False, subject_map
            )
            test_loader = make_loader(
                arrays["test"], task, args.batch_size, False, subject_map
            )
            seed_probabilities = []
            seed_labels = None
            seed_subjects = arrays["test"]["subjects"]
            seed_trials = arrays["test"]["trials"]
            fold_alphas = []
            for seed in seeds:
                set_seed(seed)
                model = create_model(arrays, subject_map, args, device)
                model, _ = train_fold_task(
                    model,
                    train_loader,
                    val_loader,
                    arrays["train"][task],
                    arrays["validation"]["subjects"],
                    args,
                    device,
                )
                adapted_model, adapted_test = adapt_model(
                    model, arrays, task, args, device
                )
                if args.adaptation != "none":
                    seed_subjects = adapted_test["subjects"]
                    seed_trials = adapted_test["trials"]
                adapted_test_loader = make_loader(
                    adapted_test, task, args.batch_size, False, subject_map
                )
                (
                    test_probs,
                    test_labels,
                    val_result,
                    alpha,
                    val_probs,
                    val_labels,
                ) = fuse_test_probabilities(
                    adapted_model,
                    val_loader,
                    adapted_test_loader,
                    arrays["validation"]["subjects"],
                    args,
                    device,
                )
                temperature = 1.0
                if args.temperature_scaling:
                    temperature = fit_temperature(val_probs, val_labels)
                    test_probs = apply_temperature(test_probs, temperature)
                threshold = 0.5
                if args.threshold_tuning:
                    threshold = choose_decision_threshold(
                        val_probs,
                        val_labels,
                        arrays["validation"]["subjects"],
                    )[0]
                seed_probabilities.append(test_probs)
                seed_labels = test_labels
                fold_alphas.append(alpha)
                logger.info(
                    "[fold %d][%s][seed %d] val_macro_f1=%.4f val_subject_f1=%.4f alpha=%s",
                    fold["fold"],
                    task,
                    seed,
                    val_result["macro_f1"],
                    val_result["subject_f1_mean"],
                    "learned" if alpha is None else f"{alpha:.2f}",
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            f"{task}/fold": fold["fold"],
                            f"{task}/seed": seed,
                            f"{task}/validation_macro_f1": val_result["macro_f1"],
                            f"{task}/validation_subject_f1": val_result[
                                "subject_f1_mean"
                            ],
                            f"{task}/temperature": temperature,
                            f"{task}/threshold": threshold,
                        }
                    )
            ensemble_probs = np.mean(seed_probabilities, axis=0)
            target_oof = adaptation_oof if args.adaptation != "none" else oof
            target_oof[task]["probs"].append(ensemble_probs)
            target_oof[task]["labels"].append(seed_labels)
            target_oof[task]["subjects"].append(
                adapted_test["subjects"] if args.adaptation != "none" else seed_subjects
            )
            target_oof[task]["trials"].append(
                adapted_test["trials"] if args.adaptation != "none" else seed_trials
            )
            target_oof[task]["thresholds"].append(
                np.full(len(seed_labels), threshold, dtype=np.float32)
            )
            fold_metrics = metrics(
                ensemble_probs.argmax(axis=1), seed_labels, seed_subjects
            )
            fold_summary["tasks"][task] = {
                "macro_f1": fold_metrics["macro_f1"],
                "subject_f1_mean": fold_metrics["subject_f1_mean"],
                "worst_subject_f1": fold_metrics["worst_subject_f1"],
                "alphas": fold_alphas,
            }
            logger.info(
                "[fold %d][%s] OOF macro_f1=%.4f subject_f1=%.4f worst=%.4f",
                fold["fold"],
                task,
                fold_metrics["macro_f1"],
                fold_metrics["subject_f1_mean"],
                fold_metrics["worst_subject_f1"],
            )
        fold_records.append(fold_summary)

    results = {"args": vars(args), "folds": fold_records, "tasks": {}}
    if args.adaptation != "none":
        results["adaptation_protocol"] = (
            "The first held-out trial supplies the first adaptation_seconds of signal; "
            "metrics are computed on all remaining held-out trials."
        )
    for task in tasks:
        source_oof = adaptation_oof if args.adaptation != "none" else oof
        probs = np.concatenate(source_oof[task]["probs"])
        labels = np.concatenate(source_oof[task]["labels"])
        subjects = np.concatenate(source_oof[task]["subjects"])
        trials = np.concatenate(source_oof[task]["trials"])
        thresholds = np.concatenate(source_oof[task]["thresholds"])
        predictions = (
            (probs[:, 1] >= thresholds).astype(np.int64)
            if args.threshold_tuning
            else probs.argmax(axis=1)
        )
        threshold = float(np.mean(thresholds)) if args.threshold_tuning else 0.50
        task_metrics = metrics(predictions, labels, subjects)
        task_metrics["generalization_score"] = generalization_score(task_metrics)
        task_metrics["threshold"] = threshold
        results["tasks"][task] = task_metrics
        logger.info(
            "OOF %s | macro_f1=%.4f subject_f1_mean=%.4f std=%.4f worst=%.4f G=%.4f",
            task,
            task_metrics["macro_f1"],
            task_metrics["subject_f1_mean"],
            task_metrics["subject_f1_std"],
            task_metrics["worst_subject_f1"],
            task_metrics["generalization_score"],
        )
        np.savez_compressed(
            Path(args.output_dir) / f"saffaryazdi_{args.tag}_oof_{task}.npz",
            probs=probs,
            labels=labels,
            subjects=subjects,
            trials=trials,
            thresholds=thresholds,
        )
        if wandb_run is not None:
            wandb_run.summary[f"{task}/accuracy"] = task_metrics["accuracy"]
            wandb_run.summary[f"{task}/macro_f1"] = task_metrics["macro_f1"]
            wandb_run.summary[f"{task}/subject_f1_mean"] = task_metrics[
                "subject_f1_mean"
            ]

    mean_subject = float(
        np.mean([results["tasks"][task]["subject_f1_mean"] for task in tasks])
    )
    mean_macro = float(np.mean([results["tasks"][task]["macro_f1"] for task in tasks]))
    results["summary"] = {
        "mean_subject_f1": mean_subject,
        "mean_macro_f1": mean_macro,
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    output_path = Path(args.output_dir) / f"saffaryazdi_{args.tag}_cv.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(
        "SUMMARY | mean_subject_f1=%.4f mean_macro_f1=%.4f | elapsed=%.1f min",
        mean_subject,
        mean_macro,
        results["summary"]["elapsed_minutes"],
    )
    logger.info("Saved CV results -> %s", output_path)
    if wandb_run is not None:
        wandb_run.summary["mean_macro_f1"] = mean_macro
        wandb_run.summary["mean_subject_f1"] = mean_subject
        wandb_run.finish()


if __name__ == "__main__":
    main()
