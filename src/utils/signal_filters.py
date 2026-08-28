from typing import Tuple

import numpy as np
from scipy.signal import butter
from scipy.signal import detrend
from scipy.signal import filtfilt
from scipy.signal import iirnotch


# -------------------------------------------------
# Butterworth Filters
# -------------------------------------------------

def butter_bandpass(
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4
) -> Tuple[np.ndarray, np.ndarray]:

    nyquist = 0.5 * fs

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(
        order,
        [low, high],
        btype="band"
    )

    return b, a


def butter_lowpass(
    cutoff: float,
    fs: float,
    order: int = 4
):

    nyquist = 0.5 * fs

    cutoff = cutoff / nyquist

    b, a = butter(
        order,
        cutoff,
        btype="low"
    )

    return b, a


def butter_highpass(
    cutoff: float,
    fs: float,
    order: int = 4
):

    nyquist = 0.5 * fs

    cutoff = cutoff / nyquist

    b, a = butter(
        order,
        cutoff,
        btype="high"
    )

    return b, a


# -------------------------------------------------
# Filters
# -------------------------------------------------

def bandpass_filter(
    signal: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4
):

    b, a = butter_bandpass(
        lowcut,
        highcut,
        fs,
        order
    )

    return filtfilt(
        b,
        a,
        signal,
        axis=-1
    )


def lowpass_filter(
    signal,
    cutoff,
    fs,
    order=4
):

    b, a = butter_lowpass(
        cutoff,
        fs,
        order
    )

    return filtfilt(
        b,
        a,
        signal,
        axis=-1
    )


def highpass_filter(
    signal,
    cutoff,
    fs,
    order=4
):

    b, a = butter_highpass(
        cutoff,
        fs,
        order
    )

    return filtfilt(
        b,
        a,
        signal,
        axis=-1
    )


def notch_filter(
    signal,
    fs,
    freq=50,
    quality=30
):

    b, a = iirnotch(
        freq,
        quality,
        fs
    )

    return filtfilt(
        b,
        a,
        signal,
        axis=-1
    )


# -------------------------------------------------
# Utilities
# -------------------------------------------------

def remove_linear_trend(signal):

    return detrend(
        signal,
        axis=-1
    )


def zscore(signal):

    mean = np.mean(
        signal,
        axis=-1,
        keepdims=True
    )

    std = np.std(
        signal,
        axis=-1,
        keepdims=True
    )

    std[std < 1e-8] = 1e-8

    return (signal - mean) / std


def minmax(signal):

    minimum = np.min(
        signal,
        axis=-1,
        keepdims=True
    )

    maximum = np.max(
        signal,
        axis=-1,
        keepdims=True
    )

    return (signal - minimum) / (
        maximum - minimum + 1e-8
    )