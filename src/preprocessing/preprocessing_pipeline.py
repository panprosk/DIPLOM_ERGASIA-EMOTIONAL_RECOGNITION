import logging
from typing import Dict, Any
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

        eeg_mean, eeg_std = stats["eeg"] if stats else (None, None)
        eda_mean, eda_std = stats["eda"] if stats else (None, None)
        ppg_mean, ppg_std = stats["ppg"] if stats else (None, None)

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