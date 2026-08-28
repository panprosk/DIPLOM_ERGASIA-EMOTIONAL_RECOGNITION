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

        signal = eda_window.squeeze()

        cleaned = nk.eda_clean(
            signal,
            sampling_rate=self.fs
        )

        phasic = nk.eda_phasic(
            cleaned,
            sampling_rate=self.fs
        )

        tonic = phasic["EDA_Tonic"].values

        phasic_signal = phasic["EDA_Phasic"].values

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

        return np.asarray(
            features,
            dtype=np.float32
        )