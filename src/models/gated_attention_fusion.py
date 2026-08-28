import torch
import torch.nn as nn


class GatedAttentionFusion(nn.Module):
    """
    Gated Attention Fusion.

    Η ποιότητα των βιομετρικών σημάτων διαφέρει σημαντικά μεταξύ
    subjects (π.χ. καθαρό EEG αλλά θορυβώδη περιφερειακά σήματα, ή
    το αντίστροφο). Ο μηχανισμός αυτός επιτρέπει στο δίκτυο να μαθαίνει
    δυναμικά, ξεχωριστά για κάθε δείγμα, πόσο θα "εμπιστεύεται" κάθε
    modality αντί για μια απλή, στατική συνένωση (concatenation).

        gate  = sigmoid(Linear([eeg_embedding ; physio_embedding]))
        fused = gate * eeg_embedding + (1 - gate) * physio_embedding
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()

        self.gate_layer = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(
        self,
        eeg_embedding: torch.Tensor,
        physio_embedding: torch.Tensor,
    ):
        """
        Parameters
        ----------
        eeg_embedding    : (batch, embedding_dim)
        physio_embedding : (batch, embedding_dim)

        Returns
        -------
        fused    : (batch, embedding_dim)
        eeg_gate : (batch, embedding_dim)
            Το βάρος (0..1) που δόθηκε στο EEG embedding ανά dimension.
            Χρησιμοποιείται από τον SHG-Agent για interpretability
            (πόσο συνεισφέρει κάθε modality ανά δείγμα).
        """

        concatenated = torch.cat(
            [eeg_embedding, physio_embedding], dim=-1
        )

        eeg_gate = torch.sigmoid(
            self.gate_layer(concatenated)
        )

        fused = (
            eeg_gate * eeg_embedding
            + (1.0 - eeg_gate) * physio_embedding
        )

        return fused, eeg_gate
