import logging
import neurokit2 as nk
import numpy as np

from src.preprocessing.base_preprocessor import BasePreprocessor

logger = logging.getLogger(__name__)


class PPGPreprocessor(BasePreprocessor):
    """
    Subject-aware preprocessing for PPG.
    """

    def __init__(self, sampling_rate: int = 128):
        super().__init__(sampling_rate=sampling_rate)

    def preprocess(
        self,
        signal: np.ndarray,
        subject_mean: float | np.ndarray | None = None,
        subject_std: float | np.ndarray | None = None,
    ) -> np.ndarray:

        if not isinstance(signal, np.ndarray):
            raise TypeError("signal must be a numpy.ndarray")

        # Ensure input is 1D for NeuroKit2
        if signal.ndim == 2:
            if signal.shape[0] != 1:
                raise ValueError("PPG must contain exactly one channel.")
            signal_1d = signal.squeeze(0)
        elif signal.ndim == 1:
            signal_1d = signal
        else:
            raise ValueError("PPG signal must be 1D or shape (1, samples).")

        # NeuroKit2 Cleaning
        cleaned = nk.ppg_clean(signal_1d, sampling_rate=self.fs)

        # Reshape back to (1, samples) for consistent multi-channel logic
        cleaned = cleaned[np.newaxis, :]

        # Normalization
        if subject_mean is not None and subject_std is not None:
            cleaned = (cleaned - subject_mean) / (subject_std + 1e-8)
        else:
            mean = np.mean(cleaned, axis=-1, keepdims=True)
            std = np.std(cleaned, axis=-1, keepdims=True)
            cleaned = (cleaned - mean) / (std + 1e-8)

        return cleaned.astype(np.float32)