"""
Preprocessing v2 για το Hybrid CNN-MLP πάνω στην ΕΠΙΣΗΜΑ προεπεξεργασμένη
έκδοση του DEAP (preprocessed_python, 128 Hz).

Αρχές
-----
* Το επίσημο DEAP EEG έχει ήδη: 128 Hz, αφαίρεση οφθαλμικών artifacts,
  band-pass 4-45 Hz. Γι' αυτό ΔΕΝ εφαρμόζεται δεύτερο detrend / band-pass
  / notch στο EEG.
* Baseline correction: αφαιρείται ο μέσος όρος των πρώτων 3 s ανά trial
  και κανάλι και το baseline τμήμα πετιέται (7680 samples μένουν).
* Raw-CNN είσοδος: per-trial, per-channel z-score (eps=1e-6).
* Handcrafted EEG features (256) υπολογίζονται από το baseline-corrected
  (ΟΧΙ z-scored) EEG, ώστε η απόλυτη ισχύς να έχει νόημα. Μόνο οι ζώνες
  theta/alpha/beta/gamma (4-45 Hz): το delta (1-4 Hz) δεν υπάρχει στο
  φιλτραρισμένο σήμα.
* EDA/PPG φιλτράρονται στο πλήρες trial (zero-phase) ΠΡΙΝ το windowing και
  τα 22 physio features υπολογίζονται από τα φιλτραρισμένα σήματα χωρίς
  z-score, ώστε να διατηρείται η πληροφορία πλάτους/δραστηριότητας.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List

import neurokit2 as nk
import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt, welch
from scipy.stats import kurtosis, skew

from src.preprocessing.baseline_correction import remove_trial_baseline

PROTOCOL_NAME = "deap_official_v2"

EEG_BANDS = ((4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))

EEG_FEATURE_DIM = 256
EDA_FEATURE_DIM = 12
PPG_FEATURE_DIM = 10
PHYSIO_FEATURE_DIM = EDA_FEATURE_DIM + PPG_FEATURE_DIM


class OfficialDeapV2Preprocessor:

    def __init__(
        self,
        sampling_rate: int = 128,
        window_size: int = 768,
        overlap: float = 0.75,
        baseline_samples: int = 384,
        eeg_channels: slice = slice(0, 32),
        eda_channel: int = 36,
        ppg_channel: int = 38,
        welch_nperseg: int = 256,
        welch_noverlap: int = 128,
        z_eps: float = 1e-6,
        eeg_norm: str = "trial",
    ):
        if eeg_norm not in ("trial", "subject"):
            raise ValueError("eeg_norm must be 'trial' or 'subject'.")
        self.eeg_norm = eeg_norm
        self.fs = sampling_rate
        self.window_size = int(window_size)
        self.step = max(1, int(round(self.window_size * (1.0 - overlap))))
        self.baseline_samples = int(baseline_samples)
        self.eeg_channels = eeg_channels
        self.eda_channel = eda_channel
        self.ppg_channel = ppg_channel
        self.welch_nperseg = welch_nperseg
        self.welch_noverlap = welch_noverlap
        self.z_eps = z_eps

        # EDA: Butterworth 4ης τάξης 1 Hz (low-pass) και 2ης τάξης 0.05 Hz
        # (tonic). SOS μορφή για αριθμητική σταθερότητα σε πολύ χαμηλές fc.
        self._eda_sos = butter(4, 1.0, btype="low", fs=self.fs, output="sos")
        self._tonic_sos = butter(2, 0.05, btype="low", fs=self.fs, output="sos")

        self.overlap = overlap

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": PROTOCOL_NAME,
            "baseline_samples": self.baseline_samples,
            "eeg_filtering": "none (official DEAP 4-45 Hz)",
            "eeg_cnn_normalization": (
                "per-subject per-channel z-score (all trials)"
                if self.eeg_norm == "subject"
                else "per-trial per-channel z-score"
            ),
            "eeg_features": "log abs + relative Welch band power (4 bands)",
            "eda_filter": "butter4 low-pass 1 Hz, tonic butter2 0.05 Hz",
            "ppg_clean": "neurokit2 elgendi",
            "window_size": self.window_size,
            "overlap": self.overlap,
            "step": self.step,
        }

    ##################################################################
    # Trial-level signal conditioning
    ##################################################################

    def _zscore(self, signal: np.ndarray) -> np.ndarray:
        mean = signal.mean(axis=-1, keepdims=True)
        std = signal.std(axis=-1, keepdims=True)
        return ((signal - mean) / (std + self.z_eps)).astype(np.float32)

    def _clean_ppg(self, ppg: np.ndarray) -> np.ndarray:
        ppg = np.asarray(ppg, dtype=np.float64).reshape(-1)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cleaned = np.asarray(
                    nk.ppg_clean(ppg, sampling_rate=self.fs, method="elgendi"),
                    dtype=np.float64,
                )
        except Exception:
            cleaned = ppg - ppg.mean()
        return np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)

    ##################################################################
    # Features
    ##################################################################

    def _eeg_spectral_features(self, eeg_bc: np.ndarray, starts: List[int]) -> np.ndarray:
        """eeg_bc: (channels, samples) baseline-corrected -> (n_windows, 256)."""

        windows = np.stack(
            [eeg_bc[:, s:s + self.window_size] for s in starts], axis=1
        ).astype(np.float64)  # (channels, n_windows, window)

        freqs, psd = welch(
            windows,
            fs=self.fs,
            nperseg=self.welch_nperseg,
            noverlap=self.welch_noverlap,
            axis=-1,
        )  # (channels, n_windows, n_freqs)

        df = freqs[1] - freqs[0]
        band_power = []
        last = len(EEG_BANDS) - 1
        for i, (low, high) in enumerate(EEG_BANDS):
            mask = (freqs >= low) & ((freqs <= high) if i == last else (freqs < high))
            band_power.append(psd[..., mask].sum(axis=-1) * df)
        power = np.stack(band_power, axis=-1)  # (channels, n_windows, 4)

        log_abs = np.log10(power + 1e-12)
        relative = power / (power.sum(axis=-1, keepdims=True) + 1e-12)

        n_windows = power.shape[1]
        log_abs = log_abs.transpose(1, 0, 2).reshape(n_windows, -1)
        relative = relative.transpose(1, 0, 2).reshape(n_windows, -1)

        return np.concatenate([log_abs, relative], axis=1).astype(np.float32)

    @staticmethod
    def _basic_stats(x: np.ndarray) -> List[float]:
        return [
            float(np.mean(x)),
            float(np.std(x)),
            float(np.min(x)),
            float(np.max(x)),
            float(skew(x)),
            float(kurtosis(x)),
            float(np.sum(x ** 2)),
        ]

    def _eda_features(
        self,
        filt: np.ndarray,
        tonic: np.ndarray,
        phasic: np.ndarray,
        scr_prominence: float,
    ) -> List[float]:

        peaks, props = find_peaks(
            phasic, prominence=scr_prominence, distance=self.fs
        )
        num_peaks = float(len(peaks))
        mean_amp = float(np.mean(props["prominences"])) if len(peaks) else 0.0

        return self._basic_stats(filt) + [
            float(np.mean(tonic)),
            float(np.mean(phasic)),
            num_peaks,
            mean_amp,
            float(np.sum(phasic)),
        ]

    def _ppg_features(self, cleaned: np.ndarray, peak_prominence: float) -> List[float]:

        peaks, _ = find_peaks(
            cleaned,
            distance=max(1, int(0.33 * self.fs)),
            prominence=peak_prominence,
        )

        hr_mean = hr_std = 0.0
        if len(peaks) >= 2:
            ibi = np.diff(peaks) / self.fs
            ibi = ibi[(ibi >= 60.0 / 180.0) & (ibi <= 60.0 / 40.0)]
            if len(ibi) >= 1:
                hr = 60.0 / ibi
                hr_mean = float(np.mean(hr))
                hr_std = float(np.std(hr))

        return self._basic_stats(cleaned) + [hr_mean, hr_std, float(len(peaks))]

    ##################################################################
    # Subject-level API
    ##################################################################

    def process_subject(
        self,
        subject_id: str,
        data: np.ndarray,
        labels: np.ndarray,
        binary_valence: np.ndarray | None = None,
    ) -> List[Dict[str, Any]]:
        """
        data   : (n_trials, 40, 8064) επίσημο preprocessed DEAP
        labels : (n_trials, 4)
        Επιστρέφει flat λίστα από window-dicts με 'eeg' (z-scored),
        'eda', 'ppg', 'features' (256 EEG + 22 physio) και 'physio_features'.
        """

        eeg = remove_trial_baseline(
            data[:, self.eeg_channels, :], self.baseline_samples
        )
        eda = remove_trial_baseline(
            data[:, self.eda_channel:self.eda_channel + 1, :], self.baseline_samples
        )
        ppg = remove_trial_baseline(
            data[:, self.ppg_channel:self.ppg_channel + 1, :], self.baseline_samples
        )

        windows: List[Dict[str, Any]] = []

        if self.eeg_norm == "subject":
            # Στατιστικά ανά κανάλι από ΟΛΑ τα trials του subject: κρατά τις
            # σχετικές διαφορές πλάτους μεταξύ trials, αφαιρεί τις διαφορές
            # μεταξύ ανθρώπων. Χωρίς labels.
            subj_mean = eeg.mean(axis=(0, 2), keepdims=True).astype(np.float64)
            subj_std = eeg.std(axis=(0, 2), keepdims=True).astype(np.float64)

        for trial_idx in range(eeg.shape[0]):

            eeg_bc = eeg[trial_idx].astype(np.float64)
            n_samples = eeg_bc.shape[-1]
            starts = list(range(0, n_samples - self.window_size + 1, self.step))
            if not starts:
                continue

            if self.eeg_norm == "subject":
                eeg_z = ((eeg_bc - subj_mean[0]) / (subj_std[0] + self.z_eps)).astype(np.float32)
            else:
                eeg_z = self._zscore(eeg_bc)

            eda_filt = sosfiltfilt(self._eda_sos, eda[trial_idx, 0].astype(np.float64))
            eda_tonic = sosfiltfilt(self._tonic_sos, eda_filt)
            eda_phasic = eda_filt - eda_tonic
            ppg_clean = self._clean_ppg(ppg[trial_idx, 0])

            scr_prominence = max(0.5 * float(np.std(eda_phasic)), 1e-12)
            ppg_prominence = max(0.3 * float(np.std(ppg_clean)), 1e-12)

            eeg_feats = self._eeg_spectral_features(eeg_bc, starts)

            eda_z = self._zscore(eda_filt[None, :])
            ppg_z = self._zscore(ppg_clean[None, :])

            label_row = labels[trial_idx]
            if binary_valence is not None:
                binary_label = int(binary_valence[trial_idx])
            else:
                binary_label = int(label_row[0] > 5.0)

            for window_idx, start in enumerate(starts):
                end = start + self.window_size

                physio = np.asarray(
                    self._eda_features(
                        eda_filt[start:end],
                        eda_tonic[start:end],
                        eda_phasic[start:end],
                        scr_prominence,
                    )
                    + self._ppg_features(ppg_clean[start:end], ppg_prominence),
                    dtype=np.float32,
                )
                physio = np.nan_to_num(physio, nan=0.0, posinf=0.0, neginf=0.0)

                features = np.concatenate([eeg_feats[window_idx], physio]).astype(
                    np.float32
                )

                windows.append({
                    "subject_id": subject_id,
                    "trial_id": trial_idx,
                    "window_id": window_idx,
                    "eeg": eeg_z[:, start:end].astype(np.float16),
                    "eda": eda_z[:, start:end].astype(np.float16),
                    "ppg": ppg_z[:, start:end].astype(np.float16),
                    "label": binary_label,
                    "raw_labels": label_row,
                    "features": np.nan_to_num(
                        features, nan=0.0, posinf=0.0, neginf=0.0
                    ),
                    "physio_features": physio,
                })

        return windows
