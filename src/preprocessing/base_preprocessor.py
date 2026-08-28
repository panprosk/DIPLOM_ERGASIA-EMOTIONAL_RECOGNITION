from abc import ABC, abstractmethod
import numpy as np


class BasePreprocessor(ABC):
    """
    Abstract Base Class για όλους τους preprocessors.
    """

    def __init__(self, sampling_rate: int = 128):
        self.fs = sampling_rate

    @abstractmethod
    def preprocess(self, signal: np.ndarray, **kwargs) -> np.ndarray:
        pass

    def __call__(self, signal: np.ndarray, **kwargs) -> np.ndarray:
        return self.preprocess(signal, **kwargs)