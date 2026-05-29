"""
deap_loader.py
Βήμα 1: Φόρτωση DEAP dataset
Βήμα 2: Επιλογή σημάτων (EEG, EDA, PPG)
Βήμα 3: Οργάνωση δεδομένων ανά subject
"""

import os
import pickle
import numpy as np
from pathlib import Path


# DEAP channel indices (0-based)
EEG_CHANNELS   = list(range(0, 32))   # 32 EEG channels
EDA_CHANNEL    = 36                    # Galvanic Skin Response
PPG_CHANNEL    = 38                    # Blood Volume Pulse (proxy for PPG)

SAMPLING_RATE  = 128   # Hz
BASELINE_SEC   = 3     # αρχικά 3 δευτερόλεπτα baseline
TRIAL_SEC      = 60    # ολικό trial duration (με baseline)


def load_subject(data_dir: str, subject_id: int) -> dict:
    """
    Φορτώνει ένα subject από το DEAP dataset.

    Args:
        data_dir:   path στον φάκελο με τα .dat αρχεία
        subject_id: αριθμός subject (1-32)

    Returns:
        dict με keys:
            'data'   -> np.array shape (40, n_channels, n_samples)
            'labels' -> np.array shape (40, 4)  [valence, arousal, dominance, liking]
    """
    filename = f"s{subject_id:02d}.dat"
    filepath = Path(data_dir) / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Δεν βρέθηκε: {filepath}")

    with open(filepath, "rb") as f:
        subject_data = pickle.load(f, encoding="latin1")

    data   = subject_data["data"]    # (40, 40, 8064)
    labels = subject_data["labels"]  # (40, 4)

    return {"data": data, "labels": labels}


def extract_signals(data: np.ndarray, remove_baseline: bool = True) -> dict:
    """
    Επιλέγει EEG, EDA, PPG από τα raw data ενός subject.
    Αφαιρεί το baseline (πρώτα 3 δευτερόλεπτα) αν remove_baseline=True.

    Args:
        data: shape (40, 40, 8064)

    Returns:
        dict με:
            'eeg' -> (40, 32, samples)
            'eda' -> (40, 1,  samples)
            'ppg' -> (40, 1,  samples)
    """
    baseline_samples = BASELINE_SEC * SAMPLING_RATE  # 384 samples

    if remove_baseline:
        data = data[:, :, baseline_samples:]   # (40, 40, 7680)

    eeg = data[:, EEG_CHANNELS, :]             # (40, 32, 7680)
    eda = data[:, EDA_CHANNEL:EDA_CHANNEL+1, :] # (40, 1,  7680)
    ppg = data[:, PPG_CHANNEL:PPG_CHANNEL+1, :] # (40, 1,  7680)

    return {"eeg": eeg, "eda": eda, "ppg": ppg}


def load_all_subjects(data_dir: str,
                      subject_ids: list = None,
                      remove_baseline: bool = True) -> dict:
    """
    Φορτώνει όλα τα subjects και τα οργανώνει σε ενιαίο dict.

    Returns:
        {
          subject_id (int): {
              'eeg':    (40, 32, samples),
              'eda':    (40, 1,  samples),
              'ppg':    (40, 1,  samples),
              'labels': (40, 4)
          },
          ...
        }
    """
    if subject_ids is None:
        subject_ids = list(range(1, 33))  # 1..32

    all_subjects = {}

    for sid in subject_ids:
        try:
            raw      = load_subject(data_dir, sid)
            signals  = extract_signals(raw["data"], remove_baseline=remove_baseline)
            signals["labels"] = raw["labels"]
            all_subjects[sid] = signals
            print(f"  ✓ Subject {sid:02d} loaded — "
                  f"EEG: {signals['eeg'].shape}, "
                  f"EDA: {signals['eda'].shape}, "
                  f"PPG: {signals['ppg'].shape}")
        except FileNotFoundError as e:
            print(f"  ✗ {e}")

    return all_subjects


# --- Quick test ---
if __name__ == "__main__":
    DATA_DIR = "data/deap"
    subjects = load_all_subjects(DATA_DIR, subject_ids=[1, 2])
    s1 = subjects[1]
    print(f"\nSubject 1 labels shape: {s1['labels'].shape}")
    print(f"Valence range: {s1['labels'][:, 0].min():.1f} – {s1['labels'][:, 0].max():.1f}")
