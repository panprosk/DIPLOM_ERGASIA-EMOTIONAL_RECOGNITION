"""
subject_manager.py

Organizes the DEAP dataset by subject.

Responsibilities
----------------
1. Store each subject independently.
2. Easy access to any subject.
3. Return subject ids.
4. Compute subject statistics.
5. Used by:
    - DataPipeline
    - Subject Split
    - Trainer
    - SHG-Agent
"""

from __future__ import annotations

from typing import Dict
from typing import List

import numpy as np


class SubjectManager:
    """
    Organizes DEAP data per subject.

    Internal format
    ---------------

    subjects =

    {

        1:
        {
            "eeg": ...
            "eda": ...
            "ppg": ...
            "labels": ...
        },

        2:
        {
            ...
        }

    }
    """

    def __init__(self):

        self.subjects: Dict[int, dict] = {}

    # --------------------------------------------------
    # Add subject
    # --------------------------------------------------

    def add_subject(

        self,

        subject_id: int,

        eeg: np.ndarray,

        eda: np.ndarray,

        ppg: np.ndarray,

        labels: np.ndarray,

        binary_valence: np.ndarray = None,

    ) -> None:

        self.subjects[subject_id] = {

            "subject_id": subject_id,

            "eeg": eeg,

            "eda": eda,

            "ppg": ppg,

            "labels": labels,

            "binary_valence": binary_valence,

        }

    # --------------------------------------------------
    # Get subject
    # --------------------------------------------------

    def get_subject(

        self,

        subject_id: int,

    ) -> dict:

        return self.subjects[subject_id]

    # --------------------------------------------------
    # Get all subject ids
    # --------------------------------------------------

    def get_subject_ids(self) -> List[int]:

        return sorted(self.subjects.keys())

    # --------------------------------------------------
    # Number of subjects
    # --------------------------------------------------

    def num_subjects(self) -> int:

        return len(self.subjects)

    # --------------------------------------------------
    # Iterate
    # --------------------------------------------------

    def items(self):

        return self.subjects.items()

    # --------------------------------------------------
    # Subject statistics
    # --------------------------------------------------

    def compute_statistics(self):

        """
        Computes statistics for every subject.

        Returns
        -------
        dict
        """

        statistics = {}

        for subject_id, data in self.subjects.items():

            eeg = data["eeg"]
            eda = data["eda"]
            ppg = data["ppg"]

            statistics[subject_id] = {

                "eeg_mean": np.mean(
                    eeg,
                    axis=-1,
                    keepdims=True,
                ),

                "eeg_std": np.std(
                    eeg,
                    axis=-1,
                    keepdims=True,
                ),

                "eda_mean": np.mean(
                    eda,
                    axis=-1,
                    keepdims=True,
                ),

                "eda_std": np.std(
                    eda,
                    axis=-1,
                    keepdims=True,
                ),

                "ppg_mean": np.mean(
                    ppg,
                    axis=-1,
                    keepdims=True,
                ),

                "ppg_std": np.std(
                    ppg,
                    axis=-1,
                    keepdims=True,
                ),

            }

        return statistics

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    def summary(self):

        print("=" * 60)
        print("Subject Manager Summary")
        print("=" * 60)

        print(f"Subjects : {self.num_subjects()}")

        for sid in self.get_subject_ids():

            subject = self.subjects[sid]

            print(
                f"Subject {sid:02d}"
                f" | EEG {subject['eeg'].shape}"
                f" | EDA {subject['eda'].shape}"
                f" | PPG {subject['ppg'].shape}"
                f" | Labels {subject['labels'].shape}"
            )

        print("=" * 60)