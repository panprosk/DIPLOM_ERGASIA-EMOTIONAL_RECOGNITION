import numpy as np
import neurokit2 as nk

from scipy.stats import skew
from scipy.stats import kurtosis


class PPGFeatureExtractor:

    """
    PPG handcrafted features.

    NeuroKit2 based.

    """

    def __init__(self, sampling_rate=128):

        self.fs = sampling_rate

    def extract(self, ppg_window):

        signal = ppg_window.squeeze()

        cleaned = nk.ppg_clean(
            signal,
            sampling_rate=self.fs
        )

        # --- Safe Peak Detection ---
        # Σε ελάχιστα windows (π.χ. flat/κορεσμένο ή πολύ θορυβώδες
        # σήμα σε κάποιο subject) η NeuroKit2 μπορεί να πετάξει
        # exception (π.χ. IndexError όταν δεν βρεθούν peaks) αντί να
        # επιστρέψει NaN. Το πιάνουμε εδώ ώστε να μη σταματάει όλο
        # το feature extraction pipeline.
        try:
            peaks, info = nk.ppg_peaks(
                cleaned,
                sampling_rate=self.fs
            )

            ppg_peaks = info.get("PPG_Peaks", [])

            if ppg_peaks is not None and len(ppg_peaks) > 1:

                rate = nk.signal_rate(
                    ppg_peaks,
                    sampling_rate=self.fs
                )

                rate_mean = np.nanmean(rate)
                rate_std = np.nanstd(rate)
                num_peaks = len(ppg_peaks)

            else:
                rate_mean = 0.0
                rate_std = 0.0
                num_peaks = 0

        except Exception:
            # Fallback αν η neurokit2 αποτύχει (π.χ. flat/degenerate σήμα)
            rate_mean = 0.0
            rate_std = 0.0
            num_peaks = 0

        features = [

            np.mean(cleaned),
            np.std(cleaned),

            np.min(cleaned),
            np.max(cleaned),

            skew(cleaned),
            kurtosis(cleaned),

            np.sum(cleaned ** 2),

            rate_mean,

            rate_std,

            num_peaks

        ]

        return np.asarray(
            features,
            dtype=np.float32
        )