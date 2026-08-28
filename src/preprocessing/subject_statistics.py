import numpy as np


class SubjectStatistics:
    """
    Computes subject-wise statistics used for normalization.
    """

    @staticmethod
    def compute(signals: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """
        signals: list of ndarrays (channels, samples)
        """
        # Συνένωση όλων των trials στον άξονα του χρόνου (samples)
        merged = np.concatenate(signals, axis=-1)

        mean = merged.mean(axis=-1, keepdims=True)
        std = merged.std(axis=-1, keepdims=True)

        return mean, std