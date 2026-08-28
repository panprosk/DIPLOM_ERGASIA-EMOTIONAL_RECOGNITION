import numpy as np
from src.preprocessing.base_preprocessor import BasePreprocessor
from src.utils.signal_filters import (
    bandpass_filter,
    notch_filter,
    remove_linear_trend,
)


class EEGPreprocessor(BasePreprocessor):
    """
    Subject-aware EEG preprocessing.
    """

    def __init__(
        self,
        sampling_rate: int = 128,
        lowcut: float = 4.0,
        highcut: float = 45.0,
        notch: bool = False,
    ):
        super().__init__(sampling_rate=sampling_rate)
        self.lowcut = lowcut
        self.highcut = highcut
        self.use_notch = notch

    def preprocess(
        self,
        signal: np.ndarray,
        subject_mean: np.ndarray | None = None,
        subject_std: np.ndarray | None = None,
    ) -> np.ndarray:

        signal = remove_linear_trend(signal)

        signal = bandpass_filter(
            signal,
            lowcut=self.lowcut,
            highcut=self.highcut,
            fs=self.fs,
        )

        if self.use_notch:
            signal = notch_filter(signal, fs=self.fs)

        if subject_mean is not None and subject_std is not None:
            signal = (signal - subject_mean) / (subject_std + 1e-8)
        else:
            mean = signal.mean(axis=-1, keepdims=True)
            std = signal.std(axis=-1, keepdims=True)
            signal = (signal - mean) / (std + 1e-8)

        return signal.astype(np.float32)