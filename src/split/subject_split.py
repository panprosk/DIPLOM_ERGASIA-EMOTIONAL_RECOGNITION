"""
subject_split.py

Subject-wise splitting utilities for the DEAP dataset.

Supports

1. Train / Validation / Test split
2. Leave-One-Subject-Out (LOSO)

This module is used by:

- DataPipeline
- Generic Trainer
- SHG-Agent
"""

from __future__ import annotations

import random
from typing import Dict
from typing import List


class SubjectSplitter:
    """
    Performs subject-wise dataset splitting.

    No trial from the same subject can appear
    simultaneously in train and test.

    This guarantees proper Cross-Subject Evaluation.
    """

    def __init__(

        self,

        random_seed: int = 42,

    ):

        self.random_seed = random_seed

        random.seed(random_seed)

    # ---------------------------------------------------------
    # Train / Validation / Test Split
    # ---------------------------------------------------------

    def train_val_test_split(

        self,

        subject_ids: List[int],

        train_ratio: float = 0.70,

        val_ratio: float = 0.15,

        test_ratio: float = 0.15,

        shuffle: bool = True,

    ) -> Dict[str, List[int]]:

        """
        Splits subjects into

        Train

        Validation

        Test

        Parameters
        ----------

        subject_ids

            List of subject ids.

        train_ratio

        val_ratio

        test_ratio

        Returns
        -------

        dict
        """

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:

            raise ValueError(
                "Train + Validation + Test ratios must sum to 1."
            )

        subjects = list(subject_ids)

        if shuffle:

            random.shuffle(subjects)

        n_subjects = len(subjects)

        n_train = int(train_ratio * n_subjects)

        n_val = int(val_ratio * n_subjects)

        train_subjects = subjects[:n_train]

        validation_subjects = subjects[
            n_train:
            n_train + n_val
        ]

        test_subjects = subjects[
            n_train + n_val:
        ]

        return {

            "train": sorted(train_subjects),

            "validation": sorted(validation_subjects),

            "test": sorted(test_subjects),

        }

    # ---------------------------------------------------------
    # Leave-One-Subject-Out
    # ---------------------------------------------------------

    def loso(

        self,

        subject_ids: List[int],

    ):

        """
        Generator for Leave-One-Subject-Out.

        Yields

        train_subjects

        test_subject

        Example

        for train, test in splitter.loso(ids):
            ...
        """

        subject_ids = sorted(subject_ids)

        for test_subject in subject_ids:

            train_subjects = [

                sid

                for sid in subject_ids

                if sid != test_subject

            ]

            yield {

                "train": train_subjects,

                "test": [test_subject],

            }

    # ---------------------------------------------------------
    # Pretty Summary
    # ---------------------------------------------------------

    def summary(

        self,

        split: Dict[str, List[int]],

    ):

        print("=" * 60)

        print("Subject Split Summary")

        print("=" * 60)

        print()

        print(

            f"Train ({len(split['train'])}): "

            f"{split['train']}"

        )

        print()

        print(

            f"Validation ({len(split['validation'])}): "

            f"{split['validation']}"

        )

        print()

        print(

            f"Test ({len(split['test'])}): "

            f"{split['test']}"

        )

        print()

        print("=" * 60)

    # ---------------------------------------------------------
    # Number of folds
    # ---------------------------------------------------------

    def loso_folds(

        self,

        subject_ids: List[int],

    ) -> int:

        """
        Number of LOSO folds.

        For DEAP this is normally 32.
        """

        return len(subject_ids)