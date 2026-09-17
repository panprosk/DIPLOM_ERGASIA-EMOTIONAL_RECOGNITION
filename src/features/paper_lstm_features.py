"""
paper_lstm_features.py

Lightweight, "paper-faithful" feature extraction για την αναπαραγωγή
του Saffaryazdi et al. (2022), "Using Facial Micro-Expressions in
Combination With EEG and Physiological Signals for Emotion
Recognition", Frontiers in Psychology, 13:864047.

Σε αντίθεση με το src/features/eeg_features.py (13 features/κανάλι) και
τα src/features/eda_features.py, src/features/ppg_features.py (βαριά
NeuroKit2-based decomposition), εδώ αναπαράγονται ΑΚΡΙΒΩΣ τα minimal
features που περιγράφει το paper, υπολογισμένα πάνω σε 1-δευτερόλεπτο
(128 δείγματα) μη-επικαλυπτόμενα windows:

    EEG : FFT band power σε 5 ζώνες (Delta, Theta, Alpha, Beta, Gamma)
          ανά κανάλι -> 32 * 5 = 160 features / window.
    EDA : mean, std, |1ης τάξης διαφορά|, |2ης τάξης διαφορά| (μετά από
          median filter για αφαίρεση rapid transient artifacts) ->
          4 features / window.
    PPG : mean, std -> 2 features / window.

Η NeuroKit2-based εξαγωγή (EDAFeatureExtractor/PPGFeatureExtractor) δεν
χρησιμοποιείται εδώ επίτηδες: πάνω σε windows του 1 δευτερολέπτου η
phasic/tonic decomposition και το peak detection της NeuroKit2 δεν
είναι αξιόπιστα (πολύ μικρό μήκος σήματος) και θα εισήγαγαν θόρυβο,
ενώ το paper χρησιμοποιεί ρητά μόνο απλά στατιστικά.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import medfilt, welch


class PaperEEGFeatureExtractor:
    """FFT band-power features (Delta/Theta/Alpha/Beta/Gamma) ανά κανάλι."""

    def __init__(self, sampling_rate: int = 128):
        self.fs = sampling_rate
        self.bands = (
            (1, 4),    # Delta
            (4, 8),    # Theta
            (8, 12),   # Alpha
            (12, 30),  # Beta
            (30, 45),  # Gamma
        )

    def _channel_bandpowers(self, channel_signal: np.ndarray) -> list:

        nperseg = min(128, len(channel_signal))

        freqs, psd = welch(channel_signal, fs=self.fs, nperseg=nperseg)

        powers = []

        for low, high in self.bands:

            idx = np.logical_and(freqs >= low, freqs <= high)

            if not np.any(idx):
                powers.append(0.0)
                continue

            if hasattr(np, "trapezoid"):
                powers.append(float(np.trapezoid(psd[idx], freqs[idx])))
            else:
                powers.append(float(np.trapz(psd[idx], freqs[idx])))

        return powers

    def extract(self, eeg_window: np.ndarray) -> np.ndarray:
        """
        eeg_window : (n_channels, window_samples)

        Returns
        -------
        np.ndarray, shape (n_channels * 5,)
        """

        features = []

        for channel in eeg_window:
            features.extend(self._channel_bandpowers(channel))

        return np.asarray(features, dtype=np.float32)


class PaperEDAFeatureExtractor:
    """mean, std, mean(|Δ1|), mean(|Δ2|), μετά από median filter."""

    def __init__(self, sampling_rate: int = 128):
        self.fs = sampling_rate

    def extract(self, eda_window: np.ndarray) -> np.ndarray:

        signal = np.asarray(eda_window, dtype=np.float64).squeeze()

        if signal.ndim == 0:
            signal = signal[np.newaxis]

        # Median filter (kernel=5) -- αφαιρεί rapid transient artifacts,
        # όπως περιγράφεται στο paper. Χρειάζεται περιττό kernel <= μήκος.
        kernel = 5 if signal.size >= 5 else (1 if signal.size % 2 == 1 else 1)

        if kernel > 1:
            cleaned = medfilt(signal, kernel_size=kernel)
        else:
            cleaned = signal

        diff1 = np.diff(cleaned) if cleaned.size > 1 else np.zeros(1)
        diff2 = np.diff(diff1) if diff1.size > 1 else np.zeros(1)

        features = [
            float(np.mean(cleaned)),
            float(np.std(cleaned)),
            float(np.mean(np.abs(diff1))),
            float(np.mean(np.abs(diff2))),
        ]

        return np.asarray(features, dtype=np.float32)


class PaperPPGFeatureExtractor:
    """mean, std."""

    def extract(self, ppg_window: np.ndarray) -> np.ndarray:

        signal = np.asarray(ppg_window, dtype=np.float64).squeeze()

        features = [
            float(np.mean(signal)),
            float(np.std(signal)),
        ]

        return np.asarray(features, dtype=np.float32)
