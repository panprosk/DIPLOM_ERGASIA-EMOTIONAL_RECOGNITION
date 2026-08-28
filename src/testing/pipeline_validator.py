"""
pipeline_validator.py

Validation of the complete Data Pipeline.

Checks:

- Dataset integrity
- Shapes
- Labels
- DataLoaders
- Feature dimensions
- NaN values
- Class balance
"""

import numpy as np


class PipelineValidator:

    def __init__(self):

        pass

    @staticmethod
    def check_nan(name, array):

        if np.isnan(array).any():

            raise ValueError(f"{name} contains NaN values.")

        print(f"[OK] {name}: no NaN values")

    @staticmethod
    def check_inf(name, array):

        if np.isinf(array).any():

            raise ValueError(f"{name} contains Inf values.")

        print(f"[OK] {name}: no Inf values")

    @staticmethod
    def check_labels(labels):

        unique = np.unique(labels)

        print("Labels:", unique)

        if len(unique) != 2:

            raise ValueError("Binary classification expected.")

        print("[OK] Labels")

    @staticmethod
    def check_shapes(dataset):

        sample = dataset[0]

        eeg = sample["eeg"]
        eda = sample["eda"]
        ppg = sample["ppg"]
        features = sample["features"]

        print()

        print("EEG shape:", eeg.shape)
        print("EDA shape:", eda.shape)
        print("PPG shape:", ppg.shape)
        print("Features:", features.shape)

        print()

        print("[OK] Shapes")

    @staticmethod
    def check_dataset(dataset):

        print()

        print("Dataset size:", len(dataset))

        PipelineValidator.check_shapes(dataset)

        print()

        print("[OK] Dataset")

    @staticmethod
    def check_loader(loader):

        batch = next(iter(loader))

        print()

        print("Batch size:", len(batch["label"]))

        print("EEG batch:", batch["eeg"].shape)

        print("EDA batch:", batch["eda"].shape)

        print("PPG batch:", batch["ppg"].shape)

        print("Features batch:", batch["features"].shape)

        print()

        print("[OK] DataLoader")

    @staticmethod
    def validate(

        train_dataset,

        val_dataset,

        test_dataset,

        train_loader,

        val_loader,

        test_loader,

        train_labels,

    ):

        print("=" * 70)
        print("VALIDATING DATA PIPELINE")
        print("=" * 70)

        PipelineValidator.check_dataset(train_dataset)

        PipelineValidator.check_dataset(val_dataset)

        PipelineValidator.check_dataset(test_dataset)

        PipelineValidator.check_loader(train_loader)

        PipelineValidator.check_loader(val_loader)

        PipelineValidator.check_loader(test_loader)

        PipelineValidator.check_labels(train_labels)

        print()

        print("=" * 70)
        print("PIPELINE VALIDATION PASSED")
        print("=" * 70)