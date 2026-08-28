import numpy as np

from config.signals import (
    SELECTED_CHANNELS,
    EEG_CHANNELS,
    EDA_CHANNEL,
    PPG_CHANNEL
)


class SignalSelector:

    """
    Επιλογή των σημάτων που
    χρησιμοποιεί η διπλωματική.
    """

    def __init__(self):

        self.selected_channels = SELECTED_CHANNELS

    def select(self, data: np.ndarray):

        """
        data

        (40,40,8064)

        επιστρέφει

        (40,34,8064)
        """

        return data[:, self.selected_channels, :]

    def split_modalities(self, data):

        """
        Επιστρέφει χωριστά:

        EEG

        EDA

        PPG
        """

        eeg = data[:, :32, :]

        eda = data[:, 32:33, :]

        ppg = data[:, 33:34, :]

        return {

            "EEG": eeg,

            "EDA": eda,

            "PPG": ppg

        }