"""
train_paper_lstm.py

Training script για το PaperLSTMFusionModel -- reproduction /
physiological adaptation του:

    Saffaryazdi et al. (2022), "Using Facial Micro-Expressions in
    Combination With EEG and Physiological Signals for Emotion
    Recognition", Frontiers in Psychology, 13:864047.

Αρχιτεκτονική: ξεχωριστό 2-layer stacked LSTM (80->30 units) ανά
modality (EEG/EDA/PPG) πάνω σε 1-δευτερόλεπτο, μη-επικαλυπτόμενα
window-features (FFT band powers για EEG, απλά στατιστικά για
EDA/PPG), με learnable weighted-probability fusion και ΔΥΟ binary
tasks (valence, arousal) -- βλ. src/models/paper_lstm_fusion.py και
src/pipeline/paper_lstm_pipeline.py για πλήρη τεκμηρίωση.

Evaluation: subject-independent train/validation/test split
(ίδιο με τα υπόλοιπα μοντέλα του project), trial-level + subject-level
metrics (reuse του src/utils/evaluation.py::
evaluate_hierarchical_predictions).

Χρήση
-----

Πλήρες dataset:

    python train_paper_lstm.py --epochs 60

Γρήγορο δοκιμαστικό run σε λίγα subjects:

    python train_paper_lstm.py --epochs 5 --max-subjects 8

Παράδειγμα με ενισχυμένο Arousal task:

    python train_paper_lstm.py --epochs 100 --aux-weight 0.5 --arousal-weight 1.25
"""

from __future__ import annotations

import argparse
import logging
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from src.config.config import Config
from src.pipeline.paper_lstm_pipeline import PaperLSTMDataPipeline
from src.datasets.paper_lstm_dataset import PaperLSTMDataset
from src.models.paper_lstm_fusion import PaperLSTMFusionModel
from src.utils.evaluation import evaluate_hierarchical_predictions
from src.utils.generalization_score import compute_generalization_score

from torch.utils.data import DataLoader


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser(
        description="Training run για το paper-reproduction LSTM fusion model "
                    "(Saffaryazdi et al. 2022)."
    )

    parser.add_argument("--epochs", type=int, default=60)

    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--dropout", type=float, default=0.30)

    parser.add_argument(
        "--hidden1",
        type=int,
        default=80,
        help="Units του 1ου LSTM layer (paper: 80)."
    )

    parser.add_argument(
        "--hidden2",
        type=int,
        default=30,
        help="Units του 2ου LSTM layer (paper: 30)."
    )

    parser.add_argument(
        "--aux-weight",
        type=float,
        default=0.30,
        help="Βάρος του auxiliary per-branch (EEG/EDA/PPG) "
             "classification loss πάνω στο συνολικό loss."
    )

    # ============================================================
    # ΝΕΟ:
    # Ελέγχει το συνολικό βάρος του Arousal task.
    # 1.0 = κανονικό βάρος
    # 1.25 = 25% ισχυρότερο Arousal loss
    # 1.50 = 50% ισχυρότερο Arousal loss
    # ============================================================
    parser.add_argument(
        "--arousal-weight",
        type=float,
        default=1.0,
        help="Πολλαπλασιαστής του συνολικού Arousal loss."
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=12,
        help="Early stopping patience (epochs χωρίς βελτίωση "
             "στο combined valence+arousal generalization score)."
    )

    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Περιορίζει τον αριθμό subjects (γρήγορο sanity check)."
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/paper_lstm_fusion.pt"
    )

    parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Αν δοθεί: φορτώνει τα προ-υπολογισμένα "
             "features από εδώ αν υπάρχουν, αλλιώς τρέχει "
             "το pipeline και τα αποθηκεύει εκεί για "
             "μελλοντική χρήση (π.χ. μεταφορά σε άλλο "
             "μηχάνημα ώστε να μην ξαναγίνει preprocessing)."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed για αναπαραγωγιμότητα (torch/numpy/random)."
    )

    parser.add_argument(
        "--subject-aware-normalization",
        action="store_true",
        help=(
            "Χρησιμοποιεί ξεχωριστό z-score ανά subject και modality, "
            "με statistics από τα unlabeled signals του ίδιου subject. "
            "Τα feature scalers εξακολουθούν να γίνονται fit μόνο στο train."
        ),
    )

    # ============================================================
    # ΝΕΑ: CHECKPOINT SELECTION
    # ============================================================
    parser.add_argument(
        "--selection-metric",
        type=str,
        choices=["combined", "arousal", "valence", "mean"],
        default="combined",
        help=(
            "Κριτήριο επιλογής best checkpoint. "
            "'combined' = min(valence_gen, arousal_gen), "
            "'arousal' = arousal generalization, "
            "'valence' = valence generalization, "
            "'mean' = μέσος όρος των δύο."
        )
    )

    # ============================================================
    # ΝΕΑ: AROUSAL THRESHOLD TUNING
    # ============================================================
    parser.add_argument(
        "--tune-arousal-threshold",
        action="store_true",
        help=(
            "Βρίσκει το καλύτερο binary Arousal threshold "
            "στο validation set και το εφαρμόζει ΜΟΝΟ στο test."
        )
    )

    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.30,
        help="Ελάχιστο Arousal threshold για validation tuning."
    )

    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.70,
        help="Μέγιστο Arousal threshold για validation tuning."
    )

    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="Βήμα threshold tuning."
    )

    return parser.parse_args()


def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights_from_labels(
    labels: np.ndarray,
    num_classes: int = 2
) -> torch.Tensor:

    counts = np.bincount(
        labels,
        minlength=num_classes
    ).astype(np.float32)

    counts[counts == 0] = 1.0

    weights = counts.sum() / (num_classes * counts)

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


def make_loader(
    arrays: dict,
    batch_size: int,
    shuffle: bool
) -> DataLoader:

    dataset = PaperLSTMDataset(arrays)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    valence_criterion=None,
    arousal_criterion=None,
    aux_weight: float = 0.30,
    arousal_weight: float = 1.0,
    grad_clip: float = 1.0
):

    is_train = optimizer is not None

    model.train() if is_train else model.eval()

    total_loss = 0.0
    n_batches = 0

    all_valence_probs = []
    all_arousal_probs = []

    all_valence_labels = []
    all_arousal_labels = []

    all_subjects = []
    all_trials = []

    context = (
        torch.enable_grad()
        if is_train
        else torch.no_grad()
    )

    with context:

        for batch in loader:

            eeg = batch["eeg"].to(device)
            eda = batch["eda"].to(device)
            ppg = batch["ppg"].to(device)

            valence = batch["valence"].to(device)
            arousal = batch["arousal"].to(device)

            if is_train:
                optimizer.zero_grad()

            output = model(
                eeg,
                eda,
                ppg
            )

            # ====================================================
            # FUSED VALENCE LOSS
            # ====================================================

            fused_valence_loss = torch.nn.functional.nll_loss(
                torch.log(
                    output["valence_probs"].clamp_min(1e-8)
                ),
                valence,
                weight=(
                    valence_criterion.weight
                    if valence_criterion is not None
                    else None
                ),
            )

            # ====================================================
            # FUSED AROUSAL LOSS
            # ====================================================

            fused_arousal_loss = torch.nn.functional.nll_loss(
                torch.log(
                    output["arousal_probs"].clamp_min(1e-8)
                ),
                arousal,
                weight=(
                    arousal_criterion.weight
                    if arousal_criterion is not None
                    else None
                ),
            )

            # ====================================================
            # AUXILIARY BRANCH LOSSES
            # ====================================================

            aux_valence_loss = 0.0
            aux_arousal_loss = 0.0

            for key in ("eeg", "eda", "ppg"):

                aux_valence_loss = (
                    aux_valence_loss
                    + valence_criterion(
                        output["valence_branch_logits"][key],
                        valence
                    )
                )

                aux_arousal_loss = (
                    aux_arousal_loss
                    + arousal_criterion(
                        output["arousal_branch_logits"][key],
                        arousal
                    )
                )

            aux_loss = (
                aux_valence_loss
                + aux_arousal_loss
            )

            # ====================================================
            # TOTAL LOSS
            # ====================================================
            #
            # arousal_weight = 1.0:
            #     κανονικό Arousal βάρος
            #
            # arousal_weight = 1.25:
            #     25% μεγαλύτερη έμφαση στο Arousal
            #
            # arousal_weight = 1.50:
            #     50% μεγαλύτερη έμφαση στο Arousal
            #
            # ====================================================

            loss = (
                fused_valence_loss
                + arousal_weight * fused_arousal_loss
                + aux_weight * (
                    aux_valence_loss
                    + arousal_weight * aux_arousal_loss
                )
            )

            if is_train:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip
                )

                optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

            all_valence_probs.append(
                output["valence_probs"]
                .detach()
                .cpu()
                .numpy()
            )

            all_arousal_probs.append(
                output["arousal_probs"]
                .detach()
                .cpu()
                .numpy()
            )

            all_valence_labels.append(
                valence
                .detach()
                .cpu()
                .numpy()
            )

            all_arousal_labels.append(
                arousal
                .detach()
                .cpu()
                .numpy()
            )

            all_subjects.extend(
                batch["subject"]
            )

            all_trials.extend(
                [int(t) for t in batch["trial"]]
            )

    valence_probs = np.concatenate(
        all_valence_probs
    )

    arousal_probs = np.concatenate(
        all_arousal_probs
    )

    valence_labels = np.concatenate(
        all_valence_labels
    )

    arousal_labels = np.concatenate(
        all_arousal_labels
    )

    subjects = np.asarray(
        all_subjects
    )

    trials = np.asarray(
        all_trials
    )

    valence_metrics = evaluate_hierarchical_predictions(
        valence_probs,
        valence_labels,
        subjects,
        trials
    )

    arousal_metrics = evaluate_hierarchical_predictions(
        arousal_probs,
        arousal_labels,
        subjects,
        trials
    )

    return {
        "loss": total_loss / max(n_batches, 1),
        "valence": valence_metrics,
        "arousal": arousal_metrics,
    }


def print_final_evaluation(
    task_name: str,
    metrics: dict
):

    print(f"\n--- {task_name.upper()} ---")

    print(
        f"Trial Accuracy   : "
        f"{metrics['trial']['accuracy']:.4f}"
    )

    print(
        f"Trial F1 (macro) : "
        f"{metrics['trial']['f1']:.4f}"
    )

    print(
        f"Subject Trial-F1 : "
        f"{metrics['subject_f1_mean']:.4f} "
        f"± {metrics['subject_f1_std']:.4f}"
    )

    print(
        f"Worst Subject F1 : "
        f"{metrics['worst_subject_f1']:.4f}"
    )

    cm = confusion_matrix(
        metrics["trial_labels"],
        metrics["trial_predictions"],
        labels=[0, 1]
    )

    print(
        f"Trial Confusion Matrix:\n{cm}"
    )

    print(
        "Subject-wise Trial Metrics:"
    )

    for item in metrics["subject_metrics"]:

        print(
            f"  {item['subject']}: "
            f"trials={item['n_trials']}, "
            f"accuracy={item['accuracy']:.4f}, "
            f"macro_f1={item['f1']:.4f}"
        )


def tune_binary_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    trials: np.ndarray,
    threshold_min: float = 0.30,
    threshold_max: float = 0.70,
    threshold_step: float = 0.01,
):
    """
    Tune binary-class threshold αποκλειστικά στο validation set.

    Το threshold εφαρμόζεται στην πιθανότητα της class 1.
    Επιλέγεται αυτό που μεγιστοποιεί το trial-level macro F1.
    Το test set δεν χρησιμοποιείται καθόλου για την επιλογή threshold.
    """
    best_threshold = 0.50
    best_f1 = -1.0
    best_metrics = None

    thresholds = np.arange(
        threshold_min,
        threshold_max + 0.5 * threshold_step,
        threshold_step
    )

    for threshold in thresholds:
        tuned_probs = np.asarray(probs, dtype=np.float64).copy()

        # [P(class0), P(class1)] -> hard decision threshold on class 1
        tuned_probs[:, 1] = (probs[:, 1] >= threshold).astype(np.float64)
        tuned_probs[:, 0] = 1.0 - tuned_probs[:, 1]

        metrics = evaluate_hierarchical_predictions(
            tuned_probs,
            labels,
            subjects,
            trials
        )

        f1 = float(metrics["trial"]["f1"])

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_f1, best_metrics


def apply_binary_threshold(
    probs: np.ndarray,
    threshold: float
) -> np.ndarray:
    """
    Μετατρέπει τις πιθανότητες σε deterministic probabilities
    σύμφωνα με threshold της class 1.
    """
    tuned_probs = np.asarray(probs, dtype=np.float64).copy()
    tuned_probs[:, 1] = (
        probs[:, 1] >= threshold
    ).astype(np.float64)
    tuned_probs[:, 0] = 1.0 - tuned_probs[:, 1]
    return tuned_probs


def main():

    args = parse_args()

    set_seed(args.seed)

    config = Config()

    config.create_directories()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    logger.info(
        f"Device: {device}"
    )

    pipeline = PaperLSTMDataPipeline(
        config,
        window_seconds=1.0,
        overlap=0.0,
        subject_aware_normalization=args.subject_aware_normalization,
    )

    data = pipeline.run(
        max_subjects=args.max_subjects,
        cache_path=args.cache_path
    )

    train_arrays = data["train"]
    val_arrays = data["validation"]
    test_arrays = data["test"]

    eeg_dim = train_arrays["eeg"].shape[-1]
    eda_dim = train_arrays["eda"].shape[-1]
    ppg_dim = train_arrays["ppg"].shape[-1]

    logger.info(
        f"Feature dims -> "
        f"EEG={eeg_dim} "
        f"EDA={eda_dim} "
        f"PPG={ppg_dim}"
    )

    train_loader = make_loader(
        train_arrays,
        args.batch_size,
        shuffle=True
    )

    val_loader = make_loader(
        val_arrays,
        args.batch_size,
        shuffle=False
    )

    test_loader = make_loader(
        test_arrays,
        args.batch_size,
        shuffle=False
    )

    model = PaperLSTMFusionModel(
        eeg_feature_dim=eeg_dim,
        eda_feature_dim=eda_dim,
        ppg_feature_dim=ppg_dim,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        dropout=args.dropout,
        num_classes=config.NUM_CLASSES,
    ).to(device)

    # ============================================================
    # CLASS WEIGHTS
    # ============================================================

    valence_weights = class_weights_from_labels(
        train_arrays["valence"]
    ).to(device)

    arousal_weights = class_weights_from_labels(
        train_arrays["arousal"]
    ).to(device)

    # ============================================================
    # CRITERIA
    # ============================================================

    valence_criterion = nn.CrossEntropyLoss(
        weight=valence_weights
    )

    arousal_criterion = nn.CrossEntropyLoss(
        weight=arousal_weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(
            3,
            args.patience // 3
        ),
        min_lr=1e-5,
    )

    best_score = -1e9

    best_state = None

    epochs_without_improvement = 0

    # ============================================================
    # TRAINING LOOP
    # ============================================================

    for epoch in range(
        1,
        args.epochs + 1
    ):

        start = time.time()

        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            valence_criterion,
            arousal_criterion,
            args.aux_weight,
            args.arousal_weight,
            args.grad_clip,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            None,
            valence_criterion,
            arousal_criterion,
            args.aux_weight,
            args.arousal_weight,
            args.grad_clip,
        )

        valence_score = compute_generalization_score(
            val_metrics["valence"]
        )

        arousal_score = compute_generalization_score(
            val_metrics["arousal"]
        )

        # ============================================================
        # CHECKPOINT / SCHEDULER SELECTION
        # ============================================================
        if args.selection_metric == "arousal":
            selection_score = arousal_score
        elif args.selection_metric == "valence":
            selection_score = valence_score
        elif args.selection_metric == "mean":
            selection_score = 0.5 * (
                valence_score + arousal_score
            )
        else:
            # Original conservative criterion:
            # προστατεύει από collapse του ενός task.
            selection_score = min(
                valence_score,
                arousal_score
            )

        scheduler.step(
            selection_score
        )

        elapsed = time.time() - start

        print(
            f"[Epoch {epoch:3d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_valence_f1="
            f"{val_metrics['valence']['trial']['f1']:.4f} | "
            f"val_arousal_f1="
            f"{val_metrics['arousal']['trial']['f1']:.4f} | "
            f"val_valence_gen="
            f"{valence_score:.4f} | "
            f"val_arousal_gen="
            f"{arousal_score:.4f} | "
            f"selection_score({args.selection_metric})="
            f"{selection_score:.4f} | "
            f"lr="
            f"{optimizer.param_groups[0]['lr']:.6f} | "
            f"elapsed="
            f"{elapsed:.1f}s"
        )

        # ========================================================
        # BEST CHECKPOINT
        # ========================================================

        if selection_score > best_score:

            best_score = selection_score

            best_state = {
                k: v.detach()
                .cpu()
                .clone()
                for k, v in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        # ========================================================
        # EARLY STOPPING
        # ========================================================

        if (
            epochs_without_improvement
            >= args.patience
        ):

            print(
                f"Early stopping "
                f"(no improvement for "
                f"{args.patience} epochs)."
            )

            break

    # ============================================================
    # LOAD BEST CHECKPOINT
    # ============================================================

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    print(
        f"\nBest checkpoint selected with: "
        f"{args.selection_metric}"
    )

    # ============================================================
    # AROUSAL THRESHOLD TUNING ON VALIDATION ONLY
    # ============================================================
    arousal_threshold = 0.50

    if args.tune_arousal_threshold:
        print("\n" + "=" * 80)
        print("AROUSAL THRESHOLD TUNING (VALIDATION SET ONLY)")
        print("=" * 80)

        val_best_metrics = run_epoch(
            model,
            val_loader,
            device,
            None,
            valence_criterion,
            arousal_criterion,
            args.aux_weight,
            args.arousal_weight,
            args.grad_clip,
        )

        val_arousal = val_best_metrics["arousal"]

        # run_epoch returns the metrics but not raw probabilities in the
        # original script, so threshold tuning is performed using the
        # validation labels/predictions already exposed by the metrics.
        #
        # For exact probability-based threshold tuning we reconstruct
        # the validation pass below.
        model.eval()
        val_probs = []
        val_labels = []
        val_subjects = []
        val_trials = []

        with torch.no_grad():
            for batch in val_loader:
                output = model(
                    batch["eeg"].to(device),
                    batch["eda"].to(device),
                    batch["ppg"].to(device),
                )

                val_probs.append(
                    output["arousal_probs"]
                    .detach()
                    .cpu()
                    .numpy()
                )
                val_labels.append(
                    batch["arousal"]
                    .detach()
                    .cpu()
                    .numpy()
                )
                val_subjects.extend(batch["subject"])
                val_trials.extend(
                    [int(t) for t in batch["trial"]]
                )

        val_probs = np.concatenate(val_probs)
        val_labels = np.concatenate(val_labels)
        val_subjects = np.asarray(val_subjects)
        val_trials = np.asarray(val_trials)

        (
            arousal_threshold,
            tuned_val_f1,
            _
        ) = tune_binary_threshold(
            val_probs,
            val_labels,
            val_subjects,
            val_trials,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
        )

        print(
            f"Best validation Arousal threshold: "
            f"{arousal_threshold:.2f}"
        )
        print(
            f"Validation Arousal Trial F1 at tuned threshold: "
            f"{tuned_val_f1:.4f}"
        )
        print("=" * 80)

    # ============================================================
    # SAVE MODEL
    # ============================================================

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "eeg_feature_dim":
                eeg_dim,

            "eda_feature_dim":
                eda_dim,

            "ppg_feature_dim":
                ppg_dim,

            "hidden1":
                args.hidden1,

            "hidden2":
                args.hidden2,

            "arousal_weight":
                args.arousal_weight,

            "aux_weight":
                args.aux_weight,

            "selection_metric":
                args.selection_metric,

            "arousal_threshold":
                arousal_threshold,

            "seed":
                args.seed,
        },
        args.output
    )

    # ============================================================
    # FINAL TEST
    # ============================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "FINAL TEST SET EVALUATION "
        "(best validation checkpoint)"
    )

    print(
        "=" * 80
    )

    test_metrics = run_epoch(
        model,
        test_loader,
        device,
        None,
        valence_criterion,
        arousal_criterion,
        args.aux_weight,
        args.arousal_weight,
        args.grad_clip,
    )

    print(
        f"Test Loss: "
        f"{test_metrics['loss']:.4f}"
    )

    print_final_evaluation(
        "valence",
        test_metrics["valence"]
    )

    print_final_evaluation(
        "arousal",
        test_metrics["arousal"]
    )

    # ============================================================
    # OPTIONAL: TEST EVALUATION WITH VALIDATION-TUNED THRESHOLD
    # ============================================================
    if args.tune_arousal_threshold:
        print("\n" + "=" * 80)
        print(
            "AROUSAL TEST EVALUATION "
            f"(validation-tuned threshold={arousal_threshold:.2f})"
        )
        print("=" * 80)

        model.eval()

        test_probs = []
        test_labels = []
        test_subjects = []
        test_trials = []

        with torch.no_grad():
            for batch in test_loader:
                output = model(
                    batch["eeg"].to(device),
                    batch["eda"].to(device),
                    batch["ppg"].to(device),
                )

                test_probs.append(
                    output["arousal_probs"]
                    .detach()
                    .cpu()
                    .numpy()
                )
                test_labels.append(
                    batch["arousal"]
                    .detach()
                    .cpu()
                    .numpy()
                )
                test_subjects.extend(batch["subject"])
                test_trials.extend(
                    [int(t) for t in batch["trial"]]
                )

        test_probs = np.concatenate(test_probs)
        test_labels = np.concatenate(test_labels)
        test_subjects = np.asarray(test_subjects)
        test_trials = np.asarray(test_trials)

        tuned_test_probs = apply_binary_threshold(
            test_probs,
            arousal_threshold
        )

        tuned_test_arousal = evaluate_hierarchical_predictions(
            tuned_test_probs,
            test_labels,
            test_subjects,
            test_trials
        )

        print_final_evaluation(
            "arousal (tuned threshold)",
            tuned_test_arousal
        )
        print("=" * 80)

    # ============================================================
    # FUSION WEIGHTS
    # ============================================================

    with torch.no_grad():

        sample = next(
            iter(test_loader)
        )

        out = model(
            sample["eeg"].to(device),
            sample["eda"].to(device),
            sample["ppg"].to(device)
        )

        print(
            "\nValence fusion weights (EEG,EDA,PPG): "
        f"{out['valence_fusion_weights'].detach().cpu().numpy()}"
        )

        print(
           "Arousal fusion weights (EEG,EDA,PPG): "
        f"{out['arousal_fusion_weights'].detach().cpu().numpy()}"
        )

    print(
        "=" * 80
    )

    print(
        f"\nModel checkpoint saved to: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()