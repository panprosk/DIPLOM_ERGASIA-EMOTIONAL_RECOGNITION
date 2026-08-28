import numpy as np

from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.signal import welch


class EEGFeatureExtractor:
    """
    EEG Handcrafted Feature Extraction.

    Features per channel

    Time Domain
    -----------
    Mean
    Std
    RMS
    Min
    Max
    Skewness
    Kurtosis
    Energy

    Frequency Domain
    ----------------
    Delta
    Theta
    Alpha
    Beta
    Gamma
    """

    def __init__(self, sampling_rate=128):

        self.fs = sampling_rate

        # (low, high) όρια για κάθε συχνοτική ζώνη ενδιαφέροντος
        self.bands = (
            (1, 4),    # Delta
            (4, 8),    # Theta
            (8, 13),   # Alpha
            (13, 30),  # Beta
            (30, 45),  # Gamma
        )

    def bandpowers(self, signal):

        """
        Υπολογίζει το Welch PSD μία μόνο φορά ανά κανάλι
        και εξάγει την ισχύ όλων των ζωνών ενδιαφέροντος
        πάνω στο ίδιο PSD (αντί να καλείται ξανά το welch()
        για κάθε ζώνη ξεχωριστά).
        """

        freqs, psd = welch(
            signal,
            fs=self.fs,
            nperseg=min(256, len(signal))
        )

        powers = []

        for low, high in self.bands:

            idx = np.logical_and(freqs >= low, freqs <= high)

            if np.sum(idx) == 0:
                powers.append(0.0)
                continue

            if hasattr(np, 'trapezoid'):
                powers.append(np.trapezoid(psd[idx], freqs[idx]))
            else:
                powers.append(np.trapz(psd[idx], freqs[idx]))

        return powers

    def extract_channel_features(self, signal):

        features = []

        features.append(np.mean(signal))
        features.append(np.std(signal))
        features.append(np.sqrt(np.mean(signal ** 2)))
        features.append(np.min(signal))
        features.append(np.max(signal))
        features.append(skew(signal))
        features.append(kurtosis(signal))
        features.append(np.sum(signal ** 2))

        features.extend(self.bandpowers(signal))

        return np.asarray(features, dtype=np.float32)

    def extract(self, eeg_window):

        """
        eeg_window

        shape

        (32, window_samples)
        """

        features = []

        for channel in eeg_window:

            features.extend(
                self.extract_channel_features(channel)
            )

        return np.asarray(features, dtype=np.float32)