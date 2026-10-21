import logging
from typing import Dict, Any
import numpy as np
from src.preprocessing.eeg_preprocessing import EEGPreprocessor
from src.preprocessing.eda_preprocessing import EDAPreprocessor
from src.preprocessing.ppg_preprocessing import PPGPreprocessor
from src.preprocessing.subject_statistics import SubjectStatistics

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Applies preprocessing separately to EEG, EDA and PPG,
    supporting subject-wise normalization statistics.
    """

    def __init__(self, sampling_rate: int = 128):
        self.eeg = EEGPreprocessor(sampling_rate=sampling_rate)
        self.eda = EDAPreprocessor(sampling_rate=sampling_rate)
        self.ppg = PPGPreprocessor(sampling_rate=sampling_rate)
        self.subject_stats = {}
        self.population_stats = {}

    @staticmethod
    def _fit_channel_statistics(
        signals: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit channel-wise train-only mean/std without retaining all samples."""
        if not signals:
            raise ValueError("Cannot fit normalization statistics on empty data.")

        n_channels = signals[0].shape[0]
        sums = np.zeros(n_channels, dtype=np.float64)
        squared_sums = np.zeros(n_channels, dtype=np.float64)
        count = 0

        for signal in signals:
            if signal.ndim != 2 or signal.shape[0] != n_channels:
                raise ValueError("All signals must have a consistent (channels, samples) shape.")
            values = np.asarray(signal, dtype=np.float64)
            sums += values.sum(axis=1)
            squared_sums += np.square(values).sum(axis=1)
            count += values.shape[1]

        mean = sums / max(count, 1)
        variance = np.maximum(squared_sums / max(count, 1) - np.square(mean), 1e-6)
        return mean[:, None].astype(np.float32), np.sqrt(variance)[:, None].astype(np.float32)

    @staticmethod
    def _normalize(
        signal: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> np.ndarray:
        return ((signal - mean) / (std + 1e-8)).astype(np.float32)

    def fit_population_statistics(self, train_subjects: Dict[str, Dict[str, Any]]):
        """Fit leakage-safe channel statistics using training subjects only."""
        filtered = {"eeg": [], "eda": [], "ppg": []}
        for subject_data in train_subjects.values():
            filtered["eeg"].extend(
                self.eeg.preprocess(trial, normalize=False)
                for trial in subject_data["eeg"]
            )
            filtered["eda"].extend(
                self.eda.preprocess(trial, normalize=False)
                for trial in subject_data["eda"]
            )
            filtered["ppg"].extend(
                self.ppg.preprocess(trial, normalize=False)
                for trial in subject_data["ppg"]
            )

        self.population_stats = {
            modality: self._fit_channel_statistics(signals)
            for modality, signals in filtered.items()
        }
        logger.info(
            "Fitted train-only population normalization statistics "
            "(channel-wise mean/std)."
        )

    def compute_subject_statistics(self, train_subjects: Dict[str, Dict[str, Any]]):
        """
        Υπολογίζει τα Z-score statistics (mean, std) για κάθε υποκείμενο 
        βασισμένος στα train δεδομένα του.
        """
        for subject_id, data in train_subjects.items():
            # data["eeg"] είναι λίστα από 40 trials
            eeg_mean, eeg_std = SubjectStatistics.compute(data["eeg"])
            eda_mean, eda_std = SubjectStatistics.compute(data["eda"])
            ppg_mean, ppg_std = SubjectStatistics.compute(data["ppg"])

            self.subject_stats[subject_id] = {
                "eeg": (eeg_mean, eeg_std),
                "eda": (eda_mean, eda_std),
                "ppg": (ppg_mean, ppg_std),
            }

    def process_subject(self, subject_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Επεξεργάζεται ένα υποκείμενο (όλα τα trials του).
        """
        subject_id = subject_data.get("subject_id", None)
        stats = self.subject_stats.get(subject_id, None)

        if self.population_stats:
            eeg_mean, eeg_std = self.population_stats["eeg"]
            eda_mean, eda_std = self.population_stats["eda"]
            ppg_mean, ppg_std = self.population_stats["ppg"]
        else:
            eeg_mean, eeg_std = stats["eeg"] if stats else (None, None)
            eda_mean, eda_std = stats["eda"] if stats else (None, None)
            ppg_mean, ppg_std = stats["ppg"] if stats else (None, None)

        if self.population_stats:
            processed_eeg = [
                self._normalize(
                    self.eeg.preprocess(trial, normalize=False),
                    eeg_mean,
                    eeg_std,
                )
                for trial in subject_data["eeg"]
            ]
            processed_eda = [
                self._normalize(
                    self.eda.preprocess(trial, normalize=False),
                    eda_mean,
                    eda_std,
                )
                for trial in subject_data["eda"]
            ]
            processed_ppg = [
                self._normalize(
                    self.ppg.preprocess(trial, normalize=False),
                    ppg_mean,
                    ppg_std,
                )
                for trial in subject_data["ppg"]
            ]
        else:
            processed_eeg = [
                self.eeg.preprocess(trial, subject_mean=eeg_mean, subject_std=eeg_std)
                for trial in subject_data["eeg"]
            ]
            processed_eda = [
                self.eda.preprocess(trial, subject_mean=eda_mean, subject_std=eda_std)
                for trial in subject_data["eda"]
            ]
            processed_ppg = [
                self.ppg.preprocess(trial, subject_mean=ppg_mean, subject_std=ppg_std)
                for trial in subject_data["ppg"]
            ]

        return {
            "subject_id": subject_id,
            "eeg": processed_eeg,
            "eda": processed_eda,
            "ppg": processed_ppg,
            "labels": subject_data["labels"],
            "binary_valence": subject_data.get("binary_valence"),
        }

    def process_dataset(self, subjects: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        processed = {}
        for subject_name, subject in subjects.items():
            processed[subject_name] = self.process_subject(subject)
        return processed