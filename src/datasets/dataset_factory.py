from src.datasets.hybrid_dataset import HybridDataset
from src.datasets.multimodal_dataset import MultimodalDataset


def build_dataset(
    model_name,
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
    """
    Builds the appropriate PyTorch Dataset for the given model.

    Parameters
    ----------
    model_name : str
        "hybrid" -> HybridDataset (Hybrid CNN-MLP)
        anything else -> MultimodalDataset (Transformer / Mamba / GNN / AGAT-AGS)
    """

    if model_name == "hybrid":

        return HybridDataset(
            eeg=eeg,
            eda=eda,
            ppg=ppg,
            features=features,
            physio_features=physio_features if physio_features is not None else features,
            labels=labels,
            subjects=subjects,
            trials=trials,
            windows=windows,
        )

    return MultimodalDataset(
        eeg_windows=eeg,
        eda_windows=eda,
        ppg_windows=ppg,
        labels=labels,
        subjects=subjects,
        trials=trials,
        windows=windows,
    )