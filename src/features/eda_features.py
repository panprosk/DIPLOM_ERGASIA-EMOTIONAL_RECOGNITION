import warnings
warnings.filterwarnings("ignore")

import numpy as np
import neurokit2 as nk

from scipy.stats import skew
from scipy.stats import kurtosis


class EDAFeatureExtractor:

    """
    EDA handcrafted features.

    Uses NeuroKit2 decomposition with robust error handling for empty/flat windows.
    """

    def __init__(self, sampling_rate=128):

        self.fs = sampling_rate

    def extract(self, eda_window):

        signal = np.asarray(eda_window, dtype=np.float64).reshape(-1)

        try:
            cleaned = np.asarray(
                nk.eda_clean(signal, sampling_rate=self.fs),
                dtype=np.float64,
            ).reshape(-1)
            phasic = nk.eda_phasic(cleaned, sampling_rate=self.fs)
            tonic = np.asarray(phasic["EDA_Tonic"].values, dtype=np.float64)
            phasic_signal = np.asarray(
                phasic["EDA_Phasic"].values, dtype=np.float64
            )
        except (TypeError, ValueError, IndexError):
            # NeuroKit2/SciPy versions differ in their short-window filter
            # handling. Keep the feature dimensionality stable if filtering
            # fails on a degenerate window.
            cleaned = signal
            tonic = signal
            phasic_signal = np.diff(signal, prepend=signal[0])

        # --- Safe Peak Detection ---
        try:
            peaks, info = nk.eda_peaks(
                phasic_signal,
                sampling_rate=self.fs
            )
            
            # Ασφαλής υπολογισμός αριθμού peaks
            scr_peaks = info.get("SCR_Peaks", [])
            num_peaks = len(scr_peaks) if scr_peaks is not None else 0

            # Ασφαλής υπολογισμός μέσης αμπλιτούδας (χωρίς NaNs)
            scr_amp = info.get("SCR_Amplitude", [])
            if scr_amp is not None and len(scr_amp) > 0 and not np.all(np.isnan(scr_amp)):
                mean_amplitude = np.nanmean(scr_amp)
            else:
                mean_amplitude = 0.0

        except Exception:
            # Fallback αν η neurokit2 αποτύχει (π.χ. zero-size array / επίπεδο σήμα)
            num_peaks = 0
            mean_amplitude = 0.0

        features = [

            np.mean(cleaned),
            np.std(cleaned),

            np.min(cleaned),
            np.max(cleaned),

            skew(cleaned),
            kurtosis(cleaned),

            np.sum(cleaned ** 2),

            np.mean(tonic),

            np.mean(phasic_signal),

            num_peaks,

            mean_amplitude,

            np.sum(phasic_signal)

        ]

        return np.nan_to_num(
            np.asarray(features, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )