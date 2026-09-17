import torch
import torch.nn as nn


class PhysioMLPEncoder(nn.Module):
    """
    MLP Encoder για το physiological branch (EDA + PPG).

    Αντί να μαθαίνει το μοντέλο ξανά γνωστά φυσιολογικά χαρακτηριστικά
    από τα ακατέργαστα σήματα μέσω CNN, χρησιμοποιούνται handcrafted
    features (NeuroKit2 / SciPy) τα οποία ήδη περιγράφουν tonic/phasic
    EDA activity, SCR peaks, heart rate, HRV κ.λπ.

    Feature Vector (EDA + PPG)
        -> Linear -> BatchNorm -> ReLU -> Dropout
        -> Linear -> ReLU
        -> Physiological Embedding (batch, embedding_dim)
    """

    def __init__(
        self,
        input_dim: int = 22,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.30,
    ):
        super().__init__()

        self.mlp = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, physio_features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        physio_features : torch.Tensor
            Shape (batch, input_dim)

        Returns
        -------
        torch.Tensor
            Physiological embedding, shape (batch, embedding_dim)
        """

        return self.mlp(physio_features)
