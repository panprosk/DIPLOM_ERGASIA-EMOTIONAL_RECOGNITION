"""
preprocessing.py
Βήμα 6: Signal cleaning & normalization ανά subject και ανά signal type.

EEG  -> bandpass filter, notch filter, z-score normalization
EDA  -> lowpass filter, min-max normalization
PPG  -> bandpass filter, z-score normalization
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import neurokit2 as nk


SAMPLING_RATE = 128  # Hz


# ──────────────────────────────────────────────
# Utility filters
# ──────────────────────────────────────────────

def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_lowpass(cutoff: float, fs: float, order: int = 4):
    nyq = fs / 2.0
    cut = cutoff / nyq
    b, a = butter(order, cut, btype="low")
    return b, a


def apply_notch(signal: np.ndarray, freq: float = 50.0, fs: float = 128.0, Q: float = 30.0):
    """Notch filter για power line noise (50Hz EU / 60Hz US)."""
    b, a = iirnotch(freq / (fs / 2), Q)
    return filtfilt(b, a, signal)


# ──────────────────────────────────────────────
# EEG Preprocessing
# ──────────────────────────────────────────────

def preprocess_eeg(eeg: np.ndarray, fs: float = SAMPLING_RATE) -> np.ndarray:
    """
    Args:
        eeg: shape (n_trials, 32, n_samples)

    Returns:
        cleaned eeg: same shape, z-score normalized per channel per trial
    """
    n_trials, n_channels, n_samples = eeg.shape
    cleaned = np.zeros_like(eeg)

    b_band, a_band = butter_bandpass(4.0, 45.0, fs, order=4)  # theta έως γ

    for trial in range(n_trials):
        for ch in range(n_channels):
            sig = eeg[trial, ch, :]

            # 1. Bandpass 4-45 Hz
            sig = filtfilt(b_band, a_band, sig)

            # 2. Notch 50 Hz
            sig = apply_notch(sig, freq=50.0, fs=fs)

            # 3. Z-score normalization
            mean, std = sig.mean(), sig.std()
            sig = (sig - mean) / (std + 1e-8)

            cleaned[trial, ch, :] = sig

    return cleaned


# ──────────────────────────────────────────────
# EDA Preprocessing
# ──────────────────────────────────────────────

def preprocess_eda(eda: np.ndarray, fs: float = SAMPLING_RATE) -> np.ndarray:
    """
    Args:
        eda: shape (n_trials, 1, n_samples)

    Returns:
        cleaned eda: same shape, min-max normalized per trial
    """
    n_trials = eda.shape[0]
    cleaned  = np.zeros_like(eda)

    b_low, a_low = butter_lowpass(3.0, fs, order=4)  # EDA αργό σήμα

    for trial in range(n_trials):
        sig = eda[trial, 0, :]

        # 1. Lowpass filter
        sig = filtfilt(b_low, a_low, sig)

        # 2. Min-max normalization [0, 1]
        sig_min, sig_max = sig.min(), sig.max()
        sig = (sig - sig_min) / (sig_max - sig_min + 1e-8)

        cleaned[trial, 0, :] = sig

    return cleaned


# ──────────────────────────────────────────────
# PPG Preprocessing
# ──────────────────────────────────────────────

def preprocess_ppg(ppg: np.ndarray, fs: float = SAMPLING_RATE) -> np.ndarray:
    """
    Args:
        ppg: shape (n_trials, 1, n_samples)

    Returns:
        cleaned ppg: same shape, z-score normalized per trial
    """
    n_trials = ppg.shape[0]
    cleaned  = np.zeros_like(ppg)

    b_band, a_band = butter_bandpass(0.5, 8.0, fs, order=4)  # καρδιακός ρυθμός

    for trial in range(n_trials):
        sig = ppg[trial, 0, :]

        # 1. Bandpass 0.5-8 Hz
        sig = filtfilt(b_band, a_band, sig)

        # 2. Z-score normalization
        mean, std = sig.mean(), sig.std()
        sig = (sig - mean) / (std + 1e-8)

        cleaned[trial, 0, :] = sig

    return cleaned


# ──────────────────────────────────────────────
# Master preprocessing function
# ──────────────────────────────────────────────

def preprocess_subject(signals: dict, fs: float = SAMPLING_RATE) -> dict:
    """
    Εφαρμόζει preprocessing σε όλα τα modalities ενός subject.

    Args:
        signals: {'eeg': ..., 'eda': ..., 'ppg': ..., 'labels': ...}

    Returns:
        ίδιο dict με cleaned signals
    """
    return {
        "eeg":    preprocess_eeg(signals["eeg"],  fs=fs),
        "eda":    preprocess_eda(signals["eda"],  fs=fs),
        "ppg":    preprocess_ppg(signals["ppg"],  fs=fs),
        "labels": signals["labels"],
    }
