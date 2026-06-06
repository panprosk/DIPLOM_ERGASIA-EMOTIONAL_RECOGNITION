"""
feature_extractor.py
Βήμα 8: Εξαγωγή Handcrafted Features (στατιστικά, συχνότητας).
"""

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch

SAMPLING_RATE = 128 # Hz

def extract_eeg_features(eeg_window: np.ndarray, fs: int = SAMPLING_RATE) -> np.ndarray:
    """
    Εξάγει χαρακτηριστικά από ένα EEG window.
    Args:
        eeg_window: Ένα EEG παράθυρο, σχήμα (n_channels, n_samples).
    Returns:
        Ένα 1D numpy array με τα εξαγόμενα χαρακτηριστικά.
    """
    n_channels, n_samples = eeg_window.shape
    features = []

    # 1. Temporal Features (ανά κανάλι)
    for ch in range(n_channels):
        sig = eeg_window[ch, :]
        features.append(np.mean(sig))
        features.append(np.std(sig))
        features.append(np.min(sig))
        features.append(np.max(sig))
        features.append(skew(sig))
        features.append(kurtosis(sig))

    # 2. Frequency Features (Power Spectral Density - PSD)
    # Θα υπολογίσουμε PSD για όλο το σήμα ή ανά κανάλι, και θα εξάγουμε μέσες τιμές σε ζώνες.
    # Για αρχή, απλοποιημένο PSD για κάθε κανάλι.
    
    # Ζώνες συχνοτήτων EEG
    # Delta: 0.5-4 Hz
    # Theta: 4-8 Hz
    # Alpha: 8-12 Hz
    # Beta:  12-30 Hz
    # Gamma: 30-45 Hz (μέχρι 45 λόγω preprocessing filter)

    freq_bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 12),
        "beta":  (12, 30),
        "gamma": (30, 45),
    }

    for ch in range(n_channels):
        sig = eeg_window[ch, :]
        f, psd = welch(sig, fs=fs, nperseg=fs*2) # 2-sec segments for PSD

        for band_name, (low, high) in freq_bands.items():
            idx_band = np.where((f >= low) & (f <= high))[0]
            if len(idx_band) > 0:
                band_power = np.sum(psd[idx_band])
                features.append(band_power)
            else:
                features.append(0.0) # Αν δεν βρεθεί ζώνη, προσθέτουμε 0

    return np.array(features, dtype=np.float32)

def extract_eda_features(eda_window: np.ndarray, fs: int = SAMPLING_RATE) -> np.ndarray:
    """
    Εξάγει χαρακτηριστικά από ένα EDA window.
    Args:
        eda_window: Ένα EDA παράθυρο, σχήμα (1, n_samples).
    Returns:
        Ένα 1D numpy array με τα εξαγόμενα χαρακτηριστικά.
    """
    # Το EDA είναι ένα αργό σήμα, οπότε πιο απλά στατιστικά.
    sig = eda_window[0, :] # Μόνο 1 κανάλι για EDA

    features = [
        np.mean(sig),
        np.std(sig),
        np.min(sig),
        np.max(sig),
        skew(sig),
        kurtosis(sig),
        np.diff(sig).mean(), # Μέση τιμή πρώτης παραγώγου (ρυθμός αλλαγής)
        np.diff(sig).std(),  # Τυπική απόκλιση πρώτης παραγώγου
    ]

    return np.array(features, dtype=np.float32)


def extract_ppg_features(ppg_window: np.ndarray, fs: int = SAMPLING_RATE) -> np.ndarray:
    """
    Εξάγει χαρακτηριστικά από ένα PPG window.
    Args:
        ppg_window: Ένα PPG παράθυρο, σχήμα (1, n_samples).
    Returns:
        Ένα 1D numpy array με τα εξαγόμενα χαρακτηριστικά.
    """
    sig = ppg_window[0, :] # Μόνο 1 κανάλι για PPG

    features = [
        np.mean(sig),
        np.std(sig),
        np.min(sig),
        np.max(sig),
        skew(sig),
        kurtosis(sig),
        np.diff(sig).mean(),
        np.diff(sig).std(),
    ]

    return np.array(features, dtype=np.float32)


class FeatureExtractor:
    def __init__(self, fs: int = SAMPLING_RATE):
        self.fs = fs

    def extract_features(self, windows: np.ndarray, modality: str = "eeg") -> np.ndarray:
        """
        Εξάγει χαρακτηριστικά για μια σειρά από παράθυρα.
        windows: (n_windows, n_channels, n_samples)
        """
        all_features = []
        for i in range(windows.shape[0]):
            win = windows[i]
            if modality == "eeg":
                feat = extract_eeg_features(win, fs=self.fs)
            elif modality == "eda":
                feat = extract_eda_features(win, fs=self.fs)
            elif modality == "ppg":
                feat = extract_ppg_features(win, fs=self.fs)
            else:
                raise ValueError(f"Unknown modality: {modality}")
            all_features.append(feat)
        
        return np.stack(all_features, axis=0)


def extract_all_features(eeg_window: np.ndarray = None,
                         eda_window: np.ndarray = None,
                         ppg_window: np.ndarray = None,
                         fs: int = SAMPLING_RATE) -> np.ndarray:
    """
    Εξάγει χαρακτηριστικά από όλα τα διαθέσιμα παράθυρα σημάτων και τα συνενώνει.
    """
    all_features = []
    if eeg_window is not None:
        all_features.append(extract_eeg_features(eeg_window, fs))
    if eda_window is not None:
        all_features.append(extract_eda_features(eda_window, fs))
    if ppg_window is not None:
        all_features.append(extract_ppg_features(ppg_window, fs))
    
    if not all_features:
        return np.array([])

    return np.concatenate(all_features)
