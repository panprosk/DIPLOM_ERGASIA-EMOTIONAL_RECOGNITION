from typing import Dict, Any, List, Union
import numpy as np
from .sliding_window import SlidingWindow


class WindowSegmenter:
    """
    Segments EEG / EDA / PPG signals simultaneously across all trials of a subject or dataset.
    """

    def __init__(
        self,
        sampling_rate: int = 128,
        window_seconds: float = 4.0,
        overlap: float = 0.5
    ):
        self.segmenter = SlidingWindow(
            sampling_rate=sampling_rate,
            window_seconds=window_seconds,
            overlap=overlap
        )

    def segment_subject(self, subject: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Διατρέχει όλα τα trials (40) ενός υποκειμένου, τεμαχίζει τα σήματα 
        και επιστρέφει μια επίπεδη λίστα από samples (dictionaries).
        """
        subject_id = subject.get("subject_id", "unknown")
        eeg_trials = subject["eeg"]      # List[np.ndarray] (40 trials)
        eda_trials = subject["eda"]      # List[np.ndarray]
        ppg_trials = subject["ppg"]      # List[np.ndarray]
        labels_trials = subject["labels"]  # np.ndarray (40, 4)
        binary_valence = subject.get("binary_valence")  # np.ndarray (40,)

        subject_windows = []

        # Διατρέχουμε κάθε trial (βίντεο) ξεχωριστά
        for i in range(len(eeg_trials)):
            eeg_w = self.segmenter.segment(eeg_trials[i])
            eda_w = self.segmenter.segment(eda_trials[i])
            ppg_w = self.segmenter.segment(ppg_trials[i])
            trial_label = labels_trials[i]

            # Binary Valence label (0/1) που θα χρησιμοποιηθεί από τα μοντέλα
            if binary_valence is not None:
                trial_binary_label = int(binary_valence[i])
            else:
                trial_binary_label = int(trial_label[0] > 5.0)

            num_windows = len(eeg_w)

            # Δημιουργούμε ξεχωριστό dict για κάθε παράθυρο
            for w_idx in range(num_windows):
                sample = {
                    "subject_id": subject_id,
                    "trial_id": i,
                    "window_id": w_idx,
                    "eeg": eeg_w[w_idx],  # Shape: (32, window_samples)
                    "eda": eda_w[w_idx],  # Shape: (1, window_samples)
                    "ppg": ppg_w[w_idx],  # Shape: (1, window_samples)
                    "label": trial_binary_label,  # Binary Valence (0/1)
                    "raw_labels": trial_label,  # Shape: (4,) - original DEAP labels
                }
                subject_windows.append(sample)

        return subject_windows

    def process_subjects(self, subjects: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Επεξεργάζεται μια συλλογή από subjects (είτε λίστα είτε dictionary) 
        και επιστρέφει όλα τα παράθυρα σε μια ενιαία επίπεδη λίστα.
        """
        all_windows = []
        
        # Αν η είσοδος είναι dictionary (π.χ. {"s01": subject_dict, ...})
        if isinstance(subjects, dict):
            subject_iterable = subjects.values()
        else:
            subject_iterable = subjects

        for subject in subject_iterable:
            windows = self.segment_subject(subject)
            all_windows.extend(windows)

        return all_windows

    def segment_subjects(self, subjects: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Alias της process_subjects για συμβατότητα."""
        return self.process_subjects(subjects)