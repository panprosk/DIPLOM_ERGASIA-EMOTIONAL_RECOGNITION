import numpy as np
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Base Dataset for all models.

    Responsibilities
    ----------------
    • Store labels
    • Store metadata
    • Convert NumPy arrays to PyTorch tensors
    • Provide common utilities for all datasets
    """

    def __init__(
        self,
        labels,
        subjects=None,
        trials=None,
        windows=None,
    ):

        self.labels = np.asarray(labels)

        self.subjects = subjects
        self.trials = trials
        self.windows = windows

    def __len__(self):

        return len(self.labels)

    @staticmethod
    def to_tensor(array, dtype=torch.float32):

        if isinstance(array, torch.Tensor):
            return array.to(dtype)

        return torch.tensor(
            array,
            dtype=dtype,
        )

    def get_metadata(self, idx):

        metadata = {}

        if self.subjects is not None:
            metadata["subject"] = self._safe_int(self.subjects[idx])

        if self.trials is not None:
            metadata["trial"] = int(self.trials[idx])

        if self.windows is not None:
            metadata["window"] = int(self.windows[idx])

        return metadata

    @staticmethod
    def _safe_int(value):
        """
        Subject ids in DEAP are strings such as "s01". This helper
        keeps the value as a string when it cannot be safely cast
        to int, instead of raising an exception.
        """

        try:
            return int(value)
        except (TypeError, ValueError):
            return str(value)