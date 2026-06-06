"""
dataset.py
Βήμα 4:  Binary valence labels
Βήμα 5:  Subject-wise train/val/test split
Βήμα 7:  Windowed segmentation
Βήμα 9:  DEAPDataset class + DataLoaders
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .feature_extractor import FeatureExtractor


SAMPLING_RATE   = 128
WINDOW_SEC      = 6     # ΑΥΞΗΘΗΚΕ ΑΠΟ 4
OVERLAP_RATIO   = 0.75  # ΑΥΞΗΘΗΚΕ ΑΠΟ 0.5
VALENCE_THRESH  = 5.0


# ──────────────────────────────────────────────
# Βήμα 4: Label creation
# ──────────────────────────────────────────────

def make_binary_labels(labels: np.ndarray,
                       threshold: float = VALENCE_THRESH) -> np.ndarray:
    """
    Μετατρέπει valence scores σε binary labels.
    labels[:, 0] = valence (1-9 scale)
    Returns: (n_trials,) με 0/1
    """
    valence = labels[:, 0]
    return (valence >= threshold).astype(np.int64)


# ──────────────────────────────────────────────
# Βήμα 7: Windowed segmentation
# ──────────────────────────────────────────────

def segment_signal(signal: np.ndarray,
                   window_samples: int,
                   step_samples: int) -> np.ndarray:
    """
    Κόβει signal σε overlapping windows.
    signal: (..., n_samples)
    returns: (n_windows, ..., window_samples)
    """
    n_samples = signal.shape[-1]
    windows   = []
    start     = 0

    while start + window_samples <= n_samples:
        w = signal[..., start: start + window_samples]
        windows.append(w)
        start += step_samples

    return np.stack(windows, axis=0)


def segment_trials(signals: dict,
                   window_sec: float = WINDOW_SEC,
                   overlap: float = OVERLAP_RATIO,
                   fs: int = SAMPLING_RATE) -> tuple:
    """
    Segmentation για όλα τα trials ενός subject.
    Returns: X_eeg, X_eda, X_ppg, y
    """
    window_samples = int(window_sec * fs)
    step_samples   = int(window_samples * (1 - overlap))
    binary_labels  = make_binary_labels(signals["labels"])

    all_eeg, all_eda, all_ppg = [], [], []
    all_eeg_feat, all_eda_feat, all_ppg_feat = [], [], []
    all_y = []
    
    feature_extractor = FeatureExtractor()

    for trial_idx in range(signals["eeg"].shape[0]):
        eeg_trial = signals["eeg"][trial_idx]
        eda_trial = signals["eda"][trial_idx]
        ppg_trial = signals["ppg"][trial_idx]
        label     = binary_labels[trial_idx]

        # Raw windows
        eeg_wins = segment_signal(eeg_trial, window_samples, step_samples)
        eda_wins = segment_signal(eda_trial, window_samples, step_samples)
        ppg_wins = segment_signal(ppg_trial, window_samples, step_samples)

        # Handcrafted features
        eeg_feat = feature_extractor.extract_features(eeg_wins, modality="eeg")
        eda_feat = feature_extractor.extract_features(eda_wins, modality="eda")
        ppg_feat = feature_extractor.extract_features(ppg_wins, modality="ppg")

        n_wins = eeg_wins.shape[0]

        all_eeg.append(eeg_wins)
        all_eda.append(eda_wins)
        all_ppg.append(ppg_wins)
        
        all_eeg_feat.append(eeg_feat)
        all_eda_feat.append(eda_feat)
        all_ppg_feat.append(ppg_feat)
        
        all_y.extend([label] * n_wins)

    return (
        np.concatenate(all_eeg, axis=0),
        np.concatenate(all_eda, axis=0),
        np.concatenate(all_ppg, axis=0),
        np.concatenate(all_eeg_feat, axis=0),
        np.concatenate(all_eda_feat, axis=0),
        np.concatenate(all_ppg_feat, axis=0),
        np.array(all_y, dtype=np.int64),
    )


# ──────────────────────────────────────────────
# Βήμα 5: Subject-wise split
# ──────────────────────────────────────────────

def subject_wise_split(all_subjects: dict,
                       train_ratio: float = 0.70,
                       val_ratio:   float = 0.15,
                       seed: int = 42) -> tuple:
    """
    Χωρίζει subjects σε train/val/test.
    ΔΕΝ ανακατεύει windows από διαφορετικά subjects.
    """
    subject_ids = list(all_subjects.keys())
    np.random.seed(seed)
    np.random.shuffle(subject_ids)

    n       = len(subject_ids)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_ids = subject_ids[:n_train]
    val_ids   = subject_ids[n_train: n_train + n_val]
    test_ids  = subject_ids[n_train + n_val:]

    print(f"Train subjects ({len(train_ids)}): {train_ids}")
    print(f"Val   subjects ({len(val_ids)}):   {val_ids}")
    print(f"Test  subjects ({len(test_ids)}):  {test_ids}")

    return train_ids, val_ids, test_ids


# ──────────────────────────────────────────────
# Βήμα 9: Dataset class
# ──────────────────────────────────────────────

class DEAPDataset(Dataset):
    """
    PyTorch Dataset για DEAP multimodal data.
    Κρατά EEG, EDA, PPG και binary valence label ανά window.
    """

    def __init__(self,
                 subject_ids: list,
                 all_subjects: dict,
                 window_sec: float = WINDOW_SEC,
                 overlap: float = OVERLAP_RATIO,
                 fs: int = SAMPLING_RATE):

        self.samples = []

        for sid in subject_ids:
            signals = all_subjects[sid]
            eeg_wins, eda_wins, ppg_wins, eeg_feat, eda_feat, ppg_feat, y = segment_trials(
                signals, window_sec=window_sec, overlap=overlap, fs=fs
            )

            for i in range(len(y)):
                self.samples.append({
                    "eeg_raw": torch.tensor(eeg_wins[i], dtype=torch.float32),
                    "eda_raw": torch.tensor(eda_wins[i], dtype=torch.float32),
                    "ppg_raw": torch.tensor(ppg_wins[i], dtype=torch.float32),
                    "eeg_feat": torch.tensor(eeg_feat[i], dtype=torch.float32),
                    "eda_feat": torch.tensor(eda_feat[i], dtype=torch.float32),
                    "ppg_feat": torch.tensor(ppg_feat[i], dtype=torch.float32),
                    "label": torch.tensor(y[i], dtype=torch.long),
                })

        print(f"  Dataset size: {len(self.samples)} windows")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ──────────────────────────────────────────────
# DataLoader factory — ΔΙΟΡΘΩΜΕΝΟ για Windows
# ──────────────────────────────────────────────

def build_dataloaders(all_subjects: dict,
                      train_ids: list,
                      val_ids:   list,
                      test_ids:  list,
                      batch_size:  int   = 64,
                      num_workers: int   = 0,        # 0 για Windows
                      window_sec:  float = WINDOW_SEC,
                      overlap:     float = OVERLAP_RATIO) -> tuple:
    """
    Δημιουργεί train_loader, val_loader, test_loader.
    num_workers=0 για Windows compatibility.
    """
    print("Building train dataset...")
    train_ds = DEAPDataset(train_ids, all_subjects, window_sec, overlap)

    print("Building val dataset...")
    val_ds   = DEAPDataset(val_ids,   all_subjects, window_sec, overlap)

    print("Building test dataset...")
    test_ds  = DEAPDataset(test_ids,  all_subjects, window_sec, overlap)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,       # Windows-safe
        pin_memory=False,    # False γιατί δεν έχουμε GPU ακόμα
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader
