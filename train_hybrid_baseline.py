"""
train_hybrid_baseline.py

Baseline training script για το Hybrid CNN-MLP.

Σκοπός: να επιβεβαιώσει ότι το μοντέλο μαθαίνει πραγματικά πάνω σε
πραγματικά δεδομένα DEAP (όχι μόνο ότι δεν κάνει crash). Τρέχει το
πλήρες data pipeline (Βήματα 1-13), μετά εκπαιδεύει το Hybrid CNN-MLP
για N epochs, τυπώνοντας ανά epoch:

    - Train Loss / Train Accuracy
    - Validation Loss / Validation Accuracy / Validation F1
    - EEG gate mean (πόσο "εμπιστεύεται" το μοντέλο το EEG branch)

Στο τέλος αξιολογεί στο test set και τυπώνει τελικό Accuracy / F1 /
Confusion Matrix, και σώζει το εκπαιδευμένο μοντέλο.

Χρήση
-----
Πλήρες dataset (32 subjects, αργό λόγω feature extraction):

    python train_hybrid_baseline.py --epochs 15

Γρήγορο δοκιμαστικό run σε λίγα subjects (sanity check, λεπτά):

    python train_hybrid_baseline.py --epochs 10 --num-subjects 8 --trials-per-subject 10

Τι να περιμένεις αν όλα δουλεύουν σωστά
----------------------------------------
    - Train Loss πέφτει σταθερά προς τα κάτω epoch με epoch.
    - Train Accuracy ανεβαίνει πάνω από 50% (baseline τυχαίου μαντέματος
      για binary classification) και συνήθως πλησιάζει/ξεπερνά κάποιο
      σημείο overfitting αν τρέξεις πολλά epochs σε λίγα δεδομένα.
    - Validation Accuracy/F1 δεν είναι απαραίτητο να είναι πολύ υψηλά
      (cross-subject generalization είναι δύσκολο πρόβλημα) αλλά πρέπει
      να είναι > 0.50 και να μην είναι NaN.
    - Το eeg_gate δεν πρέπει να κολλάει μόνιμα σε ακριβώς 0.0 ή 1.0.
"""

from __future__ import annotations

import argparse
import gc
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.config.config import Config
from src.data.deap_loader import DEAPLoader
from src.pipeline.data_pipeline import DataPipeline
from src.models import build_model
from src.utils.evaluation import evaluate_hierarchical_predictions


def parse_args():

    parser = argparse.ArgumentParser(
        description="Baseline training run για το Hybrid CNN-MLP."
    )

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=None,
                         help="Overrides config.BATCH_SIZE if given.")
    parser.add_argument("--num-workers", type=int, default=None,
                         help="Overrides config.NUM_WORKERS if given. "
                              "Keep at 0 on Windows to avoid multiprocessing "
                              "pickle errors with large in-memory datasets.")
    parser.add_argument("--lr", type=float, default=None,
                         help="Overrides config.LEARNING_RATE if given.")
    parser.add_argument("--weight-decay", type=float, default=None,
                         help="Overrides config.WEIGHT_DECAY if given.")
    parser.add_argument("--dropout", type=float, default=None,
                         help="Overrides config.DROPOUT if given.")
    parser.add_argument("--embedding-dim", type=int, default=None,
                         help="Overrides the Hybrid EEG/physio embedding dimension.")
    parser.add_argument("--patience", type=int, default=None,
                         help="Early stopping patience (epochs without "
                              "Val F1 improvement). Overrides "
                              "config.EARLY_STOPPING_PATIENCE if given.")
    parser.add_argument("--label-smoothing", type=float, default=0.05,
                         help="Label smoothing for CrossEntropyLoss "
                              "(reduces over-confident predictions).")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                         help="Max gradient norm for clipping "
                              "(stabilizes training, prevents loss spikes).")
    parser.add_argument("--no-augment", action="store_true",
                         help="Απενεργοποιεί το EEG data augmentation "
                              "(Gaussian noise + channel dropout) στο training.")
    parser.add_argument("--no-adversarial", action="store_true",
                         help="Απενεργοποιεί το domain-adversarial subject "
                              "training (Gradient Reversal Layer).")
    parser.add_argument("--adversarial-weight", type=float, default=0.30,
                         help="Βάρος του adversarial subject loss "
                              "(task_loss + weight * subject_loss).")
    parser.add_argument("--adversarial-lambda-max", type=float, default=None,
                         help="Overrides config.ADVERSARIAL_LAMBDA_MAX if given.")
    parser.add_argument("--no-eeg-handcrafted", action="store_true",
                         help="Απενεργοποιεί το EEG handcrafted-feature "
                              "branch (χρησιμοποιεί μόνο raw-EEG CNN).")
    parser.add_argument("--num-subjects", type=int, default=None,
                         help="Χρησιμοποιεί μόνο τα πρώτα N subjects "
                              "(γρήγορο sanity check). Default: όλα.")
    parser.add_argument("--trials-per-subject", type=int, default=None,
                         help="Κόβει κάθε subject στα πρώτα N trials "
                              "(γρήγορο sanity check). Default: όλα (40).")
    parser.add_argument("--no-save", action="store_true",
                         help="Δεν αποθηκεύει το checkpoint στο τέλος.")

    return parser.parse_args()


def apply_quick_subset(num_subjects, trials_per_subject):
    """
    Monkey-patches τον DEAPLoader ώστε να δουλεύει μόνο πάνω σε ένα
    μικρό subset (λίγα subjects, λίγα trials/subject) για γρήγορο
    sanity-check training χωρίς να χρειάζεται να τρέξει ολόκληρο το
    32-subject dataset (που είναι αργό λόγω feature extraction).
    """

    if num_subjects is None and trials_per_subject is None:
        return

    original_get_subject_files = DEAPLoader.get_subject_files
    original_load_subject = DEAPLoader.load_subject

    def limited_get_subject_files(self):

        files = original_get_subject_files(self)

        if num_subjects is not None:
            files = files[:num_subjects]

        return files

    def limited_load_subject(self, subject_file):

        subject = original_load_subject(self, subject_file)

        if trials_per_subject is not None:

            subject = dict(subject)
            subject["data"] = subject["data"][:trials_per_subject]
            subject["labels"] = subject["labels"][:trials_per_subject]

        return subject

    DEAPLoader.get_subject_files = limited_get_subject_files
    DEAPLoader.load_subject = limited_load_subject


def augment_eeg(eeg: torch.Tensor, noise_std: float = 0.05, channel_dropout_p: float = 0.10) -> torch.Tensor:
    """
    Data augmentation στο raw EEG, ΜΟΝΟ κατά την εκπαίδευση.

    Στόχος: το CNN να μην "απομνημονεύει" το ακριβές υπογραφικό
    πρότυπο (waveform "fingerprint") κάθε train subject, κάτι που
    οδηγεί σε τέλειο overfitting στο train set αλλά μηδενική
    γενίκευση σε άγνωστα subjects (ό,τι παρατηρήθηκε στα
    προηγούμενα runs: Train Acc ~95%, Test Acc ~50% δηλ. τυχαίο).

    - Gaussian noise: μικρή τυχαία διαταραχή σε κάθε δείγμα (το EEG
      είναι ήδη z-scored per-subject, οπότε το noise_std=0.05
      αντιστοιχεί σε ~5% της τυπικής απόκλισης).
    - Channel dropout: τυχαία μηδενίζει ολόκληρα κανάλια EEG για ένα
      δείγμα, αναγκάζοντας το μοντέλο να μη βασίζεται υπερβολικά σε
      συγκεκριμένα κανάλια/ηλεκτρόδια που μπορεί να είναι πιο
      "χαρακτηριστικά" για κάποιο subject παρά για το ίδιο το
      συναίσθημα.
    """

    noise = torch.randn_like(eeg) * noise_std
    eeg = eeg + noise

    if channel_dropout_p > 0:

        batch_size, num_channels, _ = eeg.shape

        mask = (torch.rand(batch_size, num_channels, 1, device=eeg.device) > channel_dropout_p).float()

        eeg = eeg * mask

    return eeg


def build_subject_to_idx(dataset) -> dict:
    """
    Χτίζει ένα mapping από subject id (π.χ. "s01") σε ακέραιο index
    [0, num_subjects), απαραίτητο για τον adversarial subject
    classifier (CrossEntropyLoss χρειάζεται integer class indices).

    Χρησιμοποιείται ΜΟΝΟ πάνω στα subjects του train split -- τα
    subjects του validation/test split είναι εξ ορισμού άγνωστα στο
    μοντέλο, οπότε ο adversarial classifier δεν εφαρμόζεται εκεί.
    """

    unique_subjects = sorted({str(s) for s in dataset.subjects})

    return {subject: idx for idx, subject in enumerate(unique_subjects)}


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    train: bool,
    grad_clip: float = None,
    augment: bool = False,
    subject_to_idx: dict = None,
    grl_lambda: float = 0.0,
    adversarial_weight: float = 0.30,
    heartbeat_every: int = 20,
    heartbeat_label: str = "",
):
    """
    Τρέχει ένα πλήρες πέρασμα (epoch) πάνω στο loader.

    Adversarial subject loss (μόνο αν train=True και δοθεί
    subject_to_idx): προστίθεται ένας δεύτερος όρος
    `adversarial_weight * CrossEntropy(subject_logits, subject_idx)`
    στο loss. Επειδή τα subject_logits περνάνε από Gradient Reversal
    Layer πριν τον subject classifier, ο encoder ουσιαστικά μαθαίνει
    να ΜΠΕΡΔΕΥΕΙ τον subject classifier (adversarial), παράγοντας πιο
    subject-invariant embeddings -- βελτιώνει τη γενίκευση σε άγνωστα
    (val/test) subjects.

    heartbeat_every : τυπώνει progress κάθε Ν batches (flush=True),
        ώστε να είναι ξεκάθαρο ότι το training προχωράει και ΔΕΝ έχει
        κολλήσει -- σε CPU-only training πλήρους dataset ένα epoch
        μπορεί να πάρει πολλή ώρα, οπότε χωρίς αυτό το feedback ο
        χρήστης δεν μπορεί να ξεχωρίσει "αργό" από "κολλημένο".
    """

    model.train() if train else model.eval()

    total_loss = 0.0
    total_task_loss = 0.0
    total_subject_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    all_subjects = []
    all_trials = []
    gate_values = []

    use_adversarial = train and subject_to_idx is not None

    context = torch.enable_grad() if train else torch.no_grad()

    n_batches = len(loader)
    epoch_t0 = time.time()

    with context:

        for batch_idx, batch in enumerate(loader, start=1):

            eeg = batch["eeg"].to(device)
            physio = batch["physio_features"].to(device)
            eeg_handcrafted = batch["features"][:, :model.eeg_handcrafted_dim].to(device) \
                if model.eeg_handcrafted_dim else None
            labels = batch["label"].to(device)

            if train and augment:
                eeg = augment_eeg(eeg)

            if train:
                optimizer.zero_grad()

            out = model(
                eeg, physio,
                eeg_handcrafted_features=eeg_handcrafted,
                grl_lambda=grl_lambda,
            )

            task_loss = criterion(out["logits"], labels)
            loss = task_loss
            subject_loss_value = 0.0

            if use_adversarial and "subject_logits" in out:

                subject_indices = torch.tensor(
                    [subject_to_idx.get(str(s), 0) for s in batch["subject"]],
                    dtype=torch.long, device=device,
                )

                subject_loss = nn.functional.cross_entropy(
                    out["subject_logits"], subject_indices
                )

                loss = task_loss + adversarial_weight * subject_loss
                subject_loss_value = subject_loss.item()

            if train:
                loss.backward()

                if grad_clip is not None:
                    # Σταθεροποιεί την εκπαίδευση και αποτρέπει τις
                    # απότομες εκρήξεις στο loss (όπως παρατηρήθηκε
                    # στο validation loss σε προηγούμενα runs).
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=grad_clip
                    )

                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_task_loss += task_loss.item() * labels.size(0)
            total_subject_loss += subject_loss_value * labels.size(0)

            preds = out["logits"].argmax(dim=-1)

            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            all_probs.append(out["probs"].detach().cpu().numpy())
            all_subjects.extend(str(subject) for subject in batch["subject"])
            all_trials.append(batch["trial"].detach().cpu().numpy())
            gate_values.append(out["eeg_gate"].detach().cpu().numpy())

            if heartbeat_every and (batch_idx % heartbeat_every == 0 or batch_idx == n_batches):
                elapsed = time.time() - epoch_t0
                avg_batch_time = elapsed / batch_idx
                eta = avg_batch_time * (n_batches - batch_idx)
                print(
                    f"    ... {heartbeat_label}batch {batch_idx:>5}/{n_batches} "
                    f"| elapsed {elapsed/60:>6.1f} min | ETA {eta/60:>6.1f} min",
                    flush=True,
                )

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    all_trials = np.concatenate(all_trials)
    gate_values = np.concatenate(gate_values)

    hierarchical_metrics = evaluate_hierarchical_predictions(
        probabilities=all_probs,
        labels=all_labels,
        subjects=np.asarray(all_subjects),
        trials=all_trials,
    )

    n = len(all_labels)
    avg_loss = total_loss / n
    avg_task_loss = total_task_loss / n
    avg_subject_loss = total_subject_loss / n
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "loss": avg_loss,
        "task_loss": avg_task_loss,
        "subject_loss": avg_subject_loss,
        "accuracy": accuracy,
        "f1": f1,
        "trial_accuracy": hierarchical_metrics["trial"]["accuracy"],
        "trial_f1": hierarchical_metrics["trial"]["f1"],
        "subject_f1_mean": hierarchical_metrics["subject_f1_mean"],
        "subject_f1_std": hierarchical_metrics["subject_f1_std"],
        "worst_subject_f1": hierarchical_metrics["worst_subject_f1"],
        "subject_metrics": hierarchical_metrics["subject_metrics"],
        "trial_preds": hierarchical_metrics["trial_predictions"],
        "trial_labels": hierarchical_metrics["trial_labels"],
        "eeg_gate_mean": float(gate_values.mean()),
        "preds": all_preds,
        "labels": all_labels,
    }


def main():

    args = parse_args()

    apply_quick_subset(args.num_subjects, args.trials_per_subject)

    config = Config()

    # Regularized Hybrid defaults. The earlier configuration achieved very
    # high training accuracy but weak Trial F1 on unseen subjects, so this
    # reduces model capacity and strengthens regularization.
    config.EMBEDDING_DIM = 96
    config.LEARNING_RATE = 3e-4
    config.WEIGHT_DECAY = 1e-3
    config.DROPOUT = 0.40
    config.EARLY_STOPPING_PATIENCE = 6

    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size

    if args.num_workers is not None:
        config.NUM_WORKERS = args.num_workers

    if args.lr is not None:
        config.LEARNING_RATE = args.lr

    if args.weight_decay is not None:
        config.WEIGHT_DECAY = args.weight_decay

    if args.dropout is not None:
        config.DROPOUT = args.dropout

    if args.embedding_dim is not None:
        config.EMBEDDING_DIM = args.embedding_dim

    if args.patience is not None:
        config.EARLY_STOPPING_PATIENCE = args.patience

    if args.adversarial_lambda_max is not None:
        config.ADVERSARIAL_LAMBDA_MAX = args.adversarial_lambda_max

    if args.no_eeg_handcrafted:
        config.EEG_HANDCRAFTED_DIM = None

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    device = torch.device(
        "cuda" if (config.DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
    )

    # pin_memory μόνο ωφελεί όταν μεταφέρουμε tensors σε GPU (CUDA).
    # Σε CPU-only training (όπως εδώ), pin_memory=True απλά "κλειδώνει"
    # (non-pageable) σελίδες μνήμης χωρίς κανένα όφελος -- σε μηχάνημα
    # με μόνο 8GB RAM αυτό μπορεί να προκαλέσει έντονο swapping/thrashing
    # που εμφανίζεται σαν το πρόγραμμα να έχει "κολλήσει" επ' αόριστον.
    config.PIN_MEMORY = (device.type == "cuda")

    print("=" * 80, flush=True)
    print("HYBRID CNN-MLP -- REGULARIZED TRAINING RUN", flush=True)
    print("=" * 80, flush=True)
    print(f"Device            : {device}", flush=True)
    print(f"Epochs            : {args.epochs}", flush=True)
    print(f"Batch Size        : {config.BATCH_SIZE}", flush=True)
    print(f"Num Workers       : {config.NUM_WORKERS}", flush=True)
    print(f"Pin Memory        : {config.PIN_MEMORY}", flush=True)
    print(f"Learning Rate     : {config.LEARNING_RATE}", flush=True)
    print(f"Weight Decay      : {config.WEIGHT_DECAY}", flush=True)
    print("Optimizer         : AdamW", flush=True)
    print(f"Dropout           : {config.DROPOUT}", flush=True)
    print(f"Embedding Dim     : {config.EMBEDDING_DIM}", flush=True)
    print(f"Early Stop Patience: {config.EARLY_STOPPING_PATIENCE}", flush=True)
    print(f"Label Smoothing   : {args.label_smoothing}", flush=True)
    print(f"Grad Clip Norm    : {args.grad_clip}", flush=True)
    print(f"EEG Augmentation  : {'OFF' if args.no_augment else 'ON (noise + channel dropout)'}", flush=True)
    print(f"EEG Handcrafted   : {'OFF' if args.no_eeg_handcrafted else f'ON (dim={config.EEG_HANDCRAFTED_DIM})'}", flush=True)
    print(f"Domain-Adversarial: {'OFF' if args.no_adversarial else f'ON (lambda_max={config.ADVERSARIAL_LAMBDA_MAX}, weight={args.adversarial_weight})'}", flush=True)
    print(f"Num Subjects Used : {args.num_subjects or 'ALL (32)'}", flush=True)
    print(f"Trials/Subject    : {args.trials_per_subject or 'ALL (40)'}", flush=True)
    print("=" * 80, flush=True)

    print("\nRunning data pipeline (loading, preprocessing, windowing, "
          "features, datasets, dataloaders)...\n", flush=True)

    t0 = time.time()

    pipeline = DataPipeline(config)

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        train_loader,
        validation_loader,
        test_loader,
        train_labels,
        train_subjects,
    ) = pipeline.run()

    print(f"\nData pipeline finished in {time.time() - t0:.1f}s\n")

    # Το mapping subject->index χτίζεται εδώ επειδή ο αριθμός subjects
    # στο train split είναι δεδομενο-εξαρτώμενος (π.χ. 22 σε πλήρες
    # dataset, λιγότερα σε quick-subset runs) -- ο adversarial subject
    # classifier χρειάζεται να ξέρει εκ των προτέρων πόσες κλάσεις έχει.
    subject_to_idx = None
    num_subjects = None

    if not args.no_adversarial:
        subject_to_idx = build_subject_to_idx(train_dataset)
        num_subjects = len(subject_to_idx)

    model = build_model(config.MODEL_NAME, config, num_subjects=num_subjects).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model             : {config.MODEL_NAME} ({n_params:,} trainable params)\n")

    if num_subjects is not None:
        print(f"Adversarial Subject Classifier : ON ({num_subjects} train subjects)\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Ακολουθεί το ίδιο metric με το checkpoint: Trial Macro-F1.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )

    header = (
        f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | "
        f"{'Val Loss':>8} | {'Val Acc':>7} | {'Val W.F1':>8} | {'Val T.F1':>8} | {'EEG Gate':>8} | "
        f"{'Subj.Loss':>9} | {'LR':>8} | {'Epoch Time':>10}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)

    best_val_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0

    checkpoint_path = config.OUTPUT_DIR / "hybrid_cnn_mlp_regularized.pt"

    for epoch in range(1, args.epochs + 1):

        epoch_t0 = time.time()

        print(f"\n[Epoch {epoch}/{args.epochs}] Training...", flush=True)

        # DANN-style γραμμικό ramp-up του adversarial lambda: 0 στην
        # αρχή (ο encoder μαθαίνει πρώτα τη βασική εργασία) -> lambda_max
        # μέχρι το μέσο της εκπαίδευσης, ώστε το adversarial signal να
        # μην κυριαρχεί πριν ο encoder αποκτήσει χρήσιμα embeddings.
        progress = min(1.0, (epoch - 1) / max(1, args.epochs / 2))
        grl_lambda = config.ADVERSARIAL_LAMBDA_MAX * progress if subject_to_idx else 0.0

        train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, device,
            train=True, grad_clip=args.grad_clip, augment=not args.no_augment,
            subject_to_idx=subject_to_idx, grl_lambda=grl_lambda,
            adversarial_weight=args.adversarial_weight,
            heartbeat_label="train ",
        )

        print(f"[Epoch {epoch}/{args.epochs}] Validating...", flush=True)

        val_metrics = run_epoch(
            model, validation_loader, criterion, optimizer, device, train=False,
            heartbeat_label="val ",
        )

        scheduler.step(val_metrics["trial_f1"])
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_t0

        # Απελευθερώνει ρητά κάθε "νεκρό" (unreferenced) αντικείμενο
        # στο τέλος κάθε epoch -- μικρό κόστος, βοηθάει να μη
        # συσσωρεύεται σταδιακά memory pressure σε πολύωρα runs σε
        # μηχανήματα με περιορισμένη RAM (8GB).
        gc.collect()

        print(
            f"{epoch:>5} | {train_metrics['loss']:>10.4f} | "
            f"{train_metrics['accuracy']:>9.4f} | "
            f"{val_metrics['loss']:>8.4f} | {val_metrics['accuracy']:>7.4f} | "
            f"{val_metrics['f1']:>8.4f} | {val_metrics['trial_f1']:>8.4f} | "
            f"{val_metrics['eeg_gate_mean']:>8.4f} | "
            f"{train_metrics['subject_loss']:>9.4f} | "
            f"{current_lr:>8.6f} | {epoch_time/60:>8.1f} min",
            flush=True,
        )

        if val_metrics["trial_f1"] > best_val_f1:
            best_val_f1 = val_metrics["trial_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0

            # Αποθηκεύει το ΚΑΛΥΤΕΡΟ checkpoint μέχρι στιγμής μετά από
            # ΚΑΘΕ βελτίωση -- έτσι ένα πολύωρο run που διακόπτεται
            # (π.χ. λόγω memory pressure, ή αν ο χρήστης το σταματήσει)
            # δεν χάνει την πρόοδο που έχει ήδη γίνει.
            if not args.no_save:
                config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "config": config,
                        "epoch": epoch,
                        "val_window_f1": val_metrics["f1"],
                        "val_trial_f1": val_metrics["trial_f1"],
                        "val_accuracy": val_metrics["accuracy"],
                    },
                    checkpoint_path,
                )
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
                print(
                    f"\nEarly stopping: no Val F1 improvement for "
                    f"{config.EARLY_STOPPING_PATIENCE} epochs (stopped at epoch {epoch}).",
                    flush=True,
                )
                break

    print("\nTraining finished.\n", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = run_epoch(
        model, test_loader, criterion, optimizer, device, train=False
    )

    cm = confusion_matrix(test_metrics["labels"], test_metrics["preds"])
    trial_cm = confusion_matrix(
        test_metrics["trial_labels"], test_metrics["trial_preds"], labels=[0, 1]
    )

    print("=" * 80)
    print("FINAL TEST SET EVALUATION (best validation checkpoint)")
    print("=" * 80)
    print(f"Test Loss        : {test_metrics['loss']:.4f}")
    print(f"Test Accuracy    : {test_metrics['accuracy']:.4f}")
    print(f"Window F1 (macro): {test_metrics['f1']:.4f}")
    print(f"Trial Accuracy   : {test_metrics['trial_accuracy']:.4f}")
    print(f"Trial F1 (macro) : {test_metrics['trial_f1']:.4f}")
    print(f"Subject Trial-F1 : {test_metrics['subject_f1_mean']:.4f} ± {test_metrics['subject_f1_std']:.4f}")
    print(f"Worst Subject F1 : {test_metrics['worst_subject_f1']:.4f}")
    print(f"EEG Gate Mean    : {test_metrics['eeg_gate_mean']:.4f}")
    print("Window Confusion Matrix:")
    print(cm)
    print("Trial Confusion Matrix:")
    print(trial_cm)
    print("Subject-wise Trial Metrics:")
    for metrics in test_metrics["subject_metrics"]:
        print(
            f"  {metrics['subject']}: trials={metrics['n_trials']}, "
            f"accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['f1']:.4f}"
        )
    print("=" * 80)

    if not args.no_save:

        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint_path = config.OUTPUT_DIR / "hybrid_cnn_mlp_regularized.pt"

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config,
                "test_accuracy": test_metrics["accuracy"],
                "test_window_f1": test_metrics["f1"],
                "test_trial_f1": test_metrics["trial_f1"],
            },
            checkpoint_path,
        )

        print(f"\nModel checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
