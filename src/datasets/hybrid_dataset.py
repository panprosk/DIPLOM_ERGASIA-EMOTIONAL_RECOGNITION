import torch

from src.datasets.base_dataset import BaseDataset


class HybridDataset(BaseDataset):
    """
    Dataset for the Dual-Branch Hybrid CNN-MLP.

    Branch 1
    --------
    EEG window -> 1D CNN Encoder

    Branch 2
    --------
    Handcrafted physiological features (EDA + PPG)
    -> MLP Encoder

    Raw EDA and PPG windows are also kept so that the
    dataset remains fully compatible with the pipeline's
    validation and visualization utilities.

    Returns (per sample)
    ---------------------
    eeg              : (32, window_samples)
    eda              : (1, window_samples)
    ppg              : (1, window_samples)
    features         : (n_features,)
    label            : scalar (0/1 binary valence)
    subject / trial  : optional metadata
    """

    def __init__(
        self,
        eeg,
        eda,
        ppg,
        features,
        labels,
        physio_features=None,
        subjects=None,
        trials=None,
        windows=None,
    ):

        super().__init__(
            labels=labels,
            subjects=subjects,
            trials=trials,
            windows=windows,
        )

        self.eeg = eeg

        self.eda = eda

        self.ppg = ppg

        self.features = features

        # Physiological-only (EDA + PPG) handcrafted features, used as
        # the input of the MLP Encoder branch. Falls back to the full
        # feature vector if not explicitly provided.
        self.physio_features = (
            physio_features if physio_features is not None else features
        )

    def __getitem__(self, idx):

        sample = {

            "eeg": self.to_tensor(
                self.eeg[idx]
            ),

            "eda": self.to_tensor(
                self.eda[idx]
            ),

            "ppg": self.to_tensor(
                self.ppg[idx]
            ),

            "features": self.to_tensor(
                self.features[idx]
            ),

            "physio_features": self.to_tensor(
                self.physio_features[idx]
            ),

            "label": torch.tensor(
                self.labels[idx],
                dtype=torch.long,
            ),
        }

        sample.update(
            self.get_metadata(idx)
        )

        return sample