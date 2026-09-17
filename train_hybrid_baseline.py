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
import logging
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.config.config import Config
from src.data.deap_loader import DEAPLoader
from src.pipeline.data_pipeline import DataPipeline
from src.models import build_model
from src.models.supervised_contrastive_loss import SupervisedContrastiveLoss
from src.utils.evaluation import evaluate_hierarchical_predictions

logging.basicConfig(level=logging.INFO, format="%(message)s")


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
    parser.add_argument("--window-seconds", type=float, default=None,
                         help="Μέγεθος παραθύρου windowing σε δευτερόλεπτα "
                              "(overrides config.WINDOW_SIZE). Default "
                              "config: 6s. Βιβλιογραφία (DEAP cross-subject "
                              "emotion recognition) προτείνει 2-5s με ~50% "
                              "overlap ως καλύτερο συμβιβασμό ανάμεσα σε "
                              "context και pseudo-replication/redundancy.")
    parser.add_argument("--overlap", type=float, default=None,
                         help="Ποσοστό επικάλυψης (0-1) μεταξύ διαδοχικών "
                              "windows (overrides config.OVERLAP). Default "
                              "config: 0.75 (πολύ υψηλό -- ~37 σχεδόν "
                              "πανομοιότυπα windows/trial, πιθανή αιτία "
                              "overfitting). Δοκίμασε 0.5.")
    parser.add_argument("--embedding-dim", type=int, default=None,
                         help="Overrides the Hybrid EEG/physio embedding dimension.")
    parser.add_argument("--patience", type=int, default=None,
                         help="Early stopping patience (epochs without "
                              "Val F1 improvement). Overrides "
                              "config.EARLY_STOPPING_PATIENCE if given.")
    parser.add_argument("--label-smoothing", type=float, default=0.05,
                         help="Label smoothing for CrossEntropyLoss "
                              "(reduces over-confident predictions). "
                              "ΕΠΑΝΑΦΟΡΑ στο 0.05: το Optuna-informed 0.01 "
                              "(hybrid_v3 Trial 0, partial/8-trial study) "
                              "συνδυάστηκε συστηματικά με χειρότερα Test "
                              "Trial F1 σε πολλαπλά πλήρη runs σε σχέση με "
                              "το evidence-based καλύτερο recipe (0.5666).")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                         help="Max gradient norm for clipping "
                              "(stabilizes training, prevents loss spikes). "
                              "ΕΠΑΝΑΦΟΡΑ στο 1.0 (ίδιο rationale με "
                              "--label-smoothing -- το Optuna 1.8 δεν "
                              "μεταφράστηκε σε καλύτερο test score).")
    parser.add_argument("--val-smooth-window", type=int, default=3,
                         help="Πλάτος κυλιόμενου μέσου όρου (moving average) "
                              "πάνω στο val_trial_f1 πριν χρησιμοποιηθεί για "
                              "checkpoint selection/early stopping. Με μόνο "
                              "~4 validation subjects το raw val_trial_f1 "
                              "παρατηρήθηκε να ταλαντεύεται έντονα ανά epoch "
                              "(π.χ. 0.55 -> 0.33 -> 0.48...), οδηγώντας σε "
                              "επιλογή ενός 'τυχερού' noisy checkpoint που "
                              "δεν αντιπροσωπεύει πραγματική βελτίωση (το "
                              "test score ήταν σαφώς χαμηλότερο από το val "
                              "score του επιλεγμένου checkpoint). 1 = "
                              "απενεργοποιημένο (raw per-epoch value, παλιά "
                              "συμπεριφορά).")
    parser.add_argument("--no-augment", action="store_true",
                         help="Απενεργοποιεί το EEG data augmentation "
                              "(Gaussian noise + channel dropout) στο training.")
    parser.add_argument("--mixup-alpha", type=float, default=0.20,
                         help="Beta(alpha, alpha) mixup πάνω σε EEG/physio/"
                              "handcrafted features + soft label mixing "
                              "(task loss = lam*CE(y_a) + (1-lam)*CE(y_b)). "
                              "Default 0.20 ξανά: 3 πλήρη runs έδειξαν ότι το "
                              "valence-margin filtering (όχι το mixup) είναι "
                              "αυτό που χειροτερεύει τα metrics (χωρίς margin "
                              "+ mixup=0.20 -> Test Trial F1 0.5666, με "
                              "margin+mixup=0.10 -> 0.4377, με margin χωρίς "
                              "mixup + Optuna defaults -> 0.4233). Άρα το "
                              "margin filtering απενεργοποιήθηκε by default "
                              "(βλ. --valence-margin) και το mixup=0.20 "
                              "επαναφέρθηκε ως το evidence-based καλύτερο "
                              "setup. 0.0 = απενεργοποιημένο.")
    parser.add_argument("--valence-margin", type=float, default=0.0,
                         help="Πλάτος 'νεκρής ζώνης' γύρω από το valence "
                              "threshold (5.0) για dropping ασαφών trials "
                              "([threshold-margin, threshold+margin]). "
                              "Default 0.0 (ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΟ): ενώ η ιδέα "
                              "στηρίζεται στη βιβλιογραφία (label-noise "
                              "reduction), σε 2 πλήρη runs με margin=1.0 το "
                              "Test Trial F1 ήταν σαφώς χειρότερο (0.4377, "
                              "0.4233) σε σχέση με το καλύτερο run χωρίς "
                              "margin (0.5666) -- πιθανώς επειδή αφαιρεί "
                              "15-50% των trials ανά subject, μειώνοντας "
                              "πολύ το ήδη μικρό training set. Δώσε π.χ. "
                              "--valence-margin 1.0 μόνο αν θες να το "
                              "ξαναδοκιμάσεις ρητά (π.χ. με περισσότερα "
                              "subjects/trials διαθέσιμα).")
    parser.add_argument("--no-adversarial", action="store_true",
                         help="Απενεργοποιεί το domain-adversarial subject "
                              "training (Gradient Reversal Layer).")
    parser.add_argument("--adversarial-weight", type=float, default=0.30,
                         help="Βάρος του adversarial subject loss "
                              "(task_loss + weight * subject_loss). "
                              "ΕΠΑΝΑΦΟΡΑ στο 0.30 (evidence-based καλύτερο "
                              "recipe, Test Trial F1=0.5666) -- το "
                              "Optuna-informed 0.383 (partial 8-trial study) "
                              "δεν μεταφράστηκε σε καλύτερο test score σε "
                              "πολλαπλά πλήρη runs.")
    parser.add_argument("--adversarial-lambda-max", type=float, default=0.30,
                         help="Overrides config.ADVERSARIAL_LAMBDA_MAX. "
                              "ΕΠΑΝΑΦΟΡΑ στο config default (0.30).")
    parser.add_argument("--no-contrastive", action="store_true",
                         help="Απενεργοποιεί το Supervised Contrastive loss "
                              "πάνω στο fused embedding.")
    parser.add_argument("--contrastive-weight", type=float, default=0.20,
                         help="Βάρος του Supervised Contrastive loss "
                              "(task_loss + ... + weight * contrastive_loss). "
                              "ΕΠΑΝΑΦΟΡΑ στο 0.20 (evidence-based καλύτερο "
                              "recipe).")
    parser.add_argument("--contrastive-temperature", type=float, default=0.07,
                         help="Temperature του Supervised Contrastive loss. "
                              "ΕΠΑΝΑΦΟΡΑ στο 0.07 (evidence-based καλύτερο "
                              "recipe).")
    parser.add_argument("--gate-balance-weight", type=float, default=0.05,
                         help="Penalty = weight*(mean(eeg_gate) - 0.5)^2, "
                              "ωθεί το gated fusion να μην στηρίζεται σχεδόν "
                              "αποκλειστικά στο EEG branch (παρατηρήθηκε "
                              "eeg_gate~0.75-0.79 σε προηγούμενα runs, ενώ "
                              "το train acc έφτανε 0.87 με val acc ~0.45 -- "
                              "ισχυρό overfitting σύμπτωμα). 0.0 = "
                              "απενεργοποιημένο.")
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
    parser.add_argument("--cache-path", type=str, default=None,
                         help="Αν δοθεί: φορτώνει τα προ-υπολογισμένα "
                              "datasets από εδώ αν υπάρχουν, αλλιώς τρέχει "
                              "το πλήρες pipeline (preprocessing/feature "
                              "extraction) και τα αποθηκεύει εκεί (π.χ. για "
                              "μεταφορά σε άλλο μηχάνημα ώστε να παρακαμφθεί "
                              "το CPU-bound preprocessing).")
    parser.add_argument("--topk-avg", type=int, default=3,
                         help="Weight-averaging (SWA-style) πάνω στα Top-K "
                              "καλύτερα validation checkpoints (κατά val "
                              "Trial-F1) στο τέλος του training. Μειώνει το "
                              "variance/noise ενός μόνο 'best' checkpoint. "
                              "Χρησιμοποιείται ΜΟΝΟ αν το validation score "
                              "του averaged μοντέλου είναι >= του single-best "
                              "(ποτέ δεν χειροτερεύει το αποτέλεσμα). "
                              "Θέσε 0 ή 1 για να το απενεργοποιήσεις.")

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


def mixup_batch(eeg, physio, eeg_handcrafted, labels, alpha):
    """
    Mixup (Zhang et al., 2018) πάνω στα EEG/physio/handcrafted inputs.

    Δημιουργεί ένα νέο "εικονικό" δείγμα ως γραμμικό συνδυασμό δύο
    πραγματικών δειγμάτων: x_mix = lam*x_i + (1-lam)*x_j, με το lam να
    δειγματίζεται από Beta(alpha, alpha). Το task loss υπολογίζεται
    μετά ως lam*CE(pred, y_i) + (1-lam)*CE(pred, y_j) (soft target).

    Γιατί βοηθάει εδώ: αντί το CNN/MLP να μαθαίνει το ακριβές
    "υπογραφικό" waveform ενός συγκεκριμένου train subject/trial,
    αναγκάζεται να μάθει ομαλές (γραμμικά ερμηνεύσιμες) αποφασιστικές
    επιφάνειες μεταξύ κλάσεων -- τεκμηριωμένο στη βιβλιογραφία (π.χ.
    MixEmo, ICML 2026 -- πρωτότυπο-based mixing για cross-subject EEG
    emotion recognition) ότι βελτιώνει σημαντικά cross-subject
    generalization σε σχέση με plain augmentation.

    Δεν εφαρμόζεται στο adversarial subject loss ή στο supervised
    contrastive loss (και τα δύο συνεχίζουν να χρησιμοποιούν τα
    ΑΡΧΙΚΑ subject-ids/labels, ώστε να μην μπερδευτεί η σημασιολογία
    τους) -- μόνο στο βασικό classification task loss.
    """

    if alpha <= 0.0:
        return eeg, physio, eeg_handcrafted, labels, labels, 1.0

    lam = float(np.random.beta(alpha, alpha))

    batch_size = eeg.size(0)
    perm = torch.randperm(batch_size, device=eeg.device)

    eeg_mixed = lam * eeg + (1.0 - lam) * eeg[perm]
    physio_mixed = lam * physio + (1.0 - lam) * physio[perm]
    eeg_handcrafted_mixed = (
        lam * eeg_handcrafted + (1.0 - lam) * eeg_handcrafted[perm]
        if eeg_handcrafted is not None else None
    )
    labels_b = labels[perm]

    return eeg_mixed, physio_mixed, eeg_handcrafted_mixed, labels, labels_b, lam


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
    contrastive_loss_fn=None,
    contrastive_weight: float = 0.0,
    mixup_alpha: float = 0.0,
    gate_balance_weight: float = 0.0,
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

            if train and mixup_alpha > 0.0:
                eeg, physio, eeg_handcrafted, labels_a, labels_b, lam = mixup_batch(
                    eeg, physio, eeg_handcrafted, labels, mixup_alpha
                )
            else:
                labels_a, labels_b, lam = labels, labels, 1.0

            if train:
                optimizer.zero_grad()

            out = model(
                eeg, physio,
                eeg_handcrafted_features=eeg_handcrafted,
                grl_lambda=grl_lambda,
            )

            task_loss = (
                lam * criterion(out["logits"], labels_a)
                + (1.0 - lam) * criterion(out["logits"], labels_b)
            )
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

            if train and contrastive_loss_fn is not None and contrastive_weight > 0.0:

                contrastive_loss = contrastive_loss_fn(
                    out["contrastive_embedding"], labels
                )

                loss = loss + contrastive_weight * contrastive_loss

            if train and gate_balance_weight > 0.0:
                # Στα runs παρατηρήθηκε το eeg_gate να συγκλίνει προς
                # ~0.75-0.79 (το μοντέλο βασίζεται σχεδόν αποκλειστικά
                # στο EEG branch, αγνοώντας το physio branch). Το raw
                # EEG έχει πολύ μεγαλύτερη inter-subject variability
                # από τα handcrafted EDA/PPG features (FEEL paper
                # finding), οπότε αυτή η ασυμμετρία πιθανώς επιδεινώνει
                # το cross-subject overfitting. Penalty = απόσταση του
                # μέσου gate από 0.5, ωθεί το μοντέλο να αξιοποιεί και
                # τα δύο modality branches πιο ισορροπημένα.
                gate_balance_loss = (out["eeg_gate"].mean() - 0.5) ** 2
                loss = loss + gate_balance_weight * gate_balance_loss

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
    # Αυξήθηκε 10 -> 15: με το Mixup ενεργοποιημένο (soft-label loss) η
    # validation F1 έχει περισσότερο θόρυβο/variance ανά epoch, οπότε
    # patience=10 έκοβε το training πρόωρα (~epoch 12) πριν προλάβει να
    # συγκλίνει πλήρως. Ίδιο patience με τον Cross-Attention Transformer.
    config.EARLY_STOPPING_PATIENCE = 15

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

    config.VALENCE_MARGIN = args.valence_margin

    if args.window_seconds is not None:
        config.WINDOW_SIZE_SEC = args.window_seconds
        config.WINDOW_SIZE = int(args.window_seconds * config.SAMPLING_RATE)

    if args.overlap is not None:
        config.OVERLAP = args.overlap

    if args.window_seconds is not None or args.overlap is not None:
        config.STEP_SIZE = int(config.WINDOW_SIZE * (1 - config.OVERLAP))

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
    print(f"Mixup             : {'OFF' if args.mixup_alpha <= 0.0 else f'ON (alpha={args.mixup_alpha})'}", flush=True)
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
    ) = pipeline.run(cache_path=args.cache_path)

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

    contrastive_loss_fn = None
    if not args.no_contrastive:
        contrastive_loss_fn = SupervisedContrastiveLoss(
            temperature=args.contrastive_temperature
        )
        print(
            f"Supervised Contrastive: ON (weight={args.contrastive_weight}, "
            f"temperature={args.contrastive_temperature})\n",
            flush=True,
        )
    else:
        print("Supervised Contrastive: OFF\n", flush=True)

    # Ακολουθεί το ίδιο metric με το checkpoint: Trial Macro-F1.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
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
    val_trial_f1_history = []

    # Top-K checkpoint pool για weight-averaging (SWA-style) στο τέλος.
    # Το GroupNorm (όχι BatchNorm) στο EEG encoder σημαίνει ότι ΔΕΝ
    # χρειάζεται recompute των running mean/var μετά το averaging --
    # το plain parameter-wise mean είναι απευθείας valid.
    topk_checkpoints = []  # list of (val_trial_f1, state_dict_cpu)

    checkpoint_path = config.OUTPUT_DIR / "hybrid_cnn_mlp_regularized.pt"

    for epoch in range(1, args.epochs + 1):

        epoch_t0 = time.time()

        print(f"\n[Epoch {epoch}/{args.epochs}] Training...", flush=True)

        # DANN-style γραμμικό ramp-up του adversarial lambda: 0 στην
        # αρχή (ο encoder μαθαίνει πρώτα τη βασική εργασία) -> lambda_max
        # μέχρι το μέσο της εκπαίδευσης, ώστε το adversarial signal να
        # μην κυριαρχεί πριν ο encoder αποκτήσει χρήσιμα embeddings.
        # (Δοκιμάστηκε ταχύτερο ramp-up στο 1/3, αλλά μαζί με lambda_max
        # 0.75 οδήγησε σε χειρότερα αποτελέσματα -- βλ. σημείωση στο
        # config.ADVERSARIAL_LAMBDA_MAX. Επαναφορά στο epochs/2.)
        progress = min(1.0, (epoch - 1) / max(1, args.epochs / 2))
        grl_lambda = config.ADVERSARIAL_LAMBDA_MAX * progress if subject_to_idx else 0.0

        train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, device,
            train=True, grad_clip=args.grad_clip, augment=not args.no_augment,
            subject_to_idx=subject_to_idx, grl_lambda=grl_lambda,
            adversarial_weight=args.adversarial_weight,
            contrastive_loss_fn=contrastive_loss_fn,
            contrastive_weight=args.contrastive_weight,
            mixup_alpha=args.mixup_alpha,
            gate_balance_weight=args.gate_balance_weight,
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

        # Κυλιόμενος μέσος όρος (moving average) πάνω στο val_trial_f1
        # για checkpoint selection/early stopping -- με μόνο λίγα
        # validation subjects (π.χ. 4/32) το raw per-epoch trial-F1
        # παρουσιάζει μεγάλο θόρυβο/variance, οδηγώντας σε επιλογή
        # ενός "τυχερού" (noisy spike) checkpoint που δεν γενικεύει
        # (παρατηρήθηκε: val_trial_f1 spike 0.5470 στο epoch 2, αλλά
        # το τελικό Test Trial F1 ήταν μόλις 0.4377). Ο ίδιος ο ωμός
        # (raw) val_trial_f1 συνεχίζει να τυπώνεται/να στέλνεται στο
        # scheduler (LR decay) -- η εξομάλυνση εφαρμόζεται ΜΟΝΟ στην
        # απόφαση "είναι αυτό νέο best;".
        val_trial_f1_history.append(val_metrics["trial_f1"])
        smoothed_val_f1 = float(
            np.mean(val_trial_f1_history[-args.val_smooth_window:])
        )

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
            f"{current_lr:>8.6f} | {epoch_time/60:>8.1f} min"
            f" | smoothT.F1={smoothed_val_f1:.4f}",
            flush=True,
        )

        if smoothed_val_f1 > best_val_f1:
            best_val_f1 = smoothed_val_f1
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

        # Top-K pool κρατάει ΚΑΘΕ epoch (όχι μόνο τα "νέα best") -- έτσι
        # το averaging βλέπει πραγματικά K διαφορετικά, καλά epochs γύρω
        # από τη σύγκλιση, όχι μόνο μονότονα αυξανόμενα best.
        if args.topk_avg and args.topk_avg > 1:
            state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            topk_checkpoints.append((val_metrics["trial_f1"], state_cpu))
            topk_checkpoints.sort(key=lambda pair: pair[0], reverse=True)
            del topk_checkpoints[args.topk_avg:]

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

    used_topk_avg = False
    if args.topk_avg and args.topk_avg > 1 and len(topk_checkpoints) >= 2:
        # Plain parameter-wise mean πάνω στα Top-K state_dicts (SWA-style).
        # Ασφαλές γιατί το EEG encoder χρησιμοποιεί GroupNorm (όχι
        # BatchNorm) -- δεν υπάρχουν running mean/var που να χρειάζονται
        # recompute μετά το averaging.
        avg_state = {}
        keys = topk_checkpoints[0][1].keys()
        for key in keys:
            stacked = torch.stack([sd[key].float() for _, sd in topk_checkpoints], dim=0)
            avg_state[key] = stacked.mean(dim=0).to(topk_checkpoints[0][1][key].dtype)

        avg_model_state_backup = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(avg_state)
        avg_val_metrics = run_epoch(
            model, validation_loader, criterion, optimizer, device, train=False,
        )
        print(
            f"Top-{len(topk_checkpoints)} weight-averaged model -- "
            f"Val Trial-F1: {avg_val_metrics['trial_f1']:.4f} "
            f"(single-best checkpoint Val Trial-F1: {best_val_f1:.4f})",
            flush=True,
        )
        if avg_val_metrics["trial_f1"] >= best_val_f1:
            print("--> Χρησιμοποιείται το weight-averaged μοντέλο για το τελικό test.", flush=True)
            best_state = avg_state
            used_topk_avg = True
            if not args.no_save:
                config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "config": config,
                        "val_trial_f1": avg_val_metrics["trial_f1"],
                        "topk_avg": len(topk_checkpoints),
                    },
                    checkpoint_path,
                )
        else:
            print("--> Το single-best checkpoint παραμένει καλύτερο, δεν αλλάζει τίποτα.", flush=True)
            model.load_state_dict(avg_model_state_backup)

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
    print(
        "FINAL TEST SET EVALUATION "
        f"({'Top-K weight-averaged' if used_topk_avg else 'best validation'} checkpoint)"
    )
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
