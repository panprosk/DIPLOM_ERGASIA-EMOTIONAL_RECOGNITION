import logging
import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.preprocessing.base_preprocessor import BasePreprocessor
from src.utils.signal_filters import lowpass_filter

logger = logging.getLogger(__name__)


class EDAPreprocessor(BasePreprocessor):
    """
    Subject-aware preprocessing for EDA.
    """

    def __init__(
        self,
        sampling_rate: int = 128,
        highcut: float = 1.0,
        sigma: float = 1.0,
    ):
        super().__init__(sampling_rate=sampling_rate)
        self.highcut = highcut
        self.sigma = sigma

    def preprocess(
        self,
        signal: np.ndarray,
        subject_mean: float | np.ndarray | None = None,
        subject_std: float | np.ndarray | None = None,
    ) -> np.ndarray:

        if signal.ndim == 1:
            signal = signal[np.newaxis, :]

        # Lowpass filtering - Ευθυγραμμισμένο με το signal_filters.py
        signal = lowpass_filter(
            signal=signal,
            cutoff=self.highcut,
            fs=self.fs,
        )

        # Gaussian smoothing
        signal = gaussian_filter1d(
            signal,
            sigma=self.sigma,
            axis=-1,
        )

        # Normalization
        if subject_mean is not None and subject_std is not None:
            signal = (signal - subject_mean) / (subject_std + 1e-8)
        else:
            mean = signal.mean(axis=-1, keepdims=True)
            std = signal.std(axis=-1, keepdims=True)
            signal = (signal - mean) / (std + 1e-8)

        return signal.astype(np.float32)