import numpy as np


class SlidingWindow:
    """
    Generic sliding window implementation.

    Input shape: (channels, samples)
    Output shape: (num_windows, channels, window_samples)
    """

    def __init__(
        self,
        sampling_rate: int = 128,
        window_seconds: float = 4.0,
        overlap: float = 0.5
    ):
        self.fs = sampling_rate

        # Έλεγχος αν το window_seconds έχει δοθεί σε δείγματα αντί για δευτερόλεπτα
        if window_seconds > 60:  # Κανένα παράθυρο συνασθήματος δεν είναι >60 δευτερόλεπτα
            self.window_size = int(window_seconds)
            self.window_seconds = window_seconds / sampling_rate
        else:
            self.window_seconds = window_seconds
            self.window_size = int(window_seconds * sampling_rate)

        self.step = int(self.window_size * (1 - overlap))

        if self.step <= 0:
            raise ValueError("Invalid overlap value. Step size must be > 0.")

    def segment(self, signal: np.ndarray) -> np.ndarray:
        """
        Κόβει ένα 2D σήμα (channels, samples) σε παράθυρα.
        """
        if signal.ndim == 1:
            signal = signal[np.newaxis, :]

        channels, length = signal.shape

        if length < self.window_size:
            raise ValueError(
                f"Signal length ({length}) is smaller than window size ({self.window_size})."
            )

        windows = []
        start = 0

        while start + self.window_size <= length:
            end = start + self.window_size
            windows.append(signal[:, start:end])
            start += self.step

        return np.stack(windows)  # Shape: (num_windows, channels, window_samples)