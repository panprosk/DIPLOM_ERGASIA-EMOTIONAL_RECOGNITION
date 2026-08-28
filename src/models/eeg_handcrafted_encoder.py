import torch
import torch.nn as nn


class EEGHandcraftedEncoder(nn.Module):
    """
    MLP Encoder για τα handcrafted EEG χαρακτηριστικά (band powers,
    στατιστικά ανά κανάλι κ.λπ. — 13 features x 32 κανάλια = 416-dim).

    Γιατί προστέθηκε
    -----------------
    Το raw-EEG CNN branch έδειξε πολύ έντονο cross-subject generalization
    gap (τέλειο fit στο train set, τυχαία απόδοση σε άγνωστα subjects):
    το CNN μαθαίνει εύκολα το "υπογραφικό" πρότυπο (waveform pattern)
    κάθε subject αντί για γενικεύσιμα χαρακτηριστικά συναισθήματος.

    Τα handcrafted band-power χαρακτηριστικά (delta/theta/alpha/beta/
    gamma ισχύς, στατιστικά ροπών κ.λπ.) είναι πιο σταθερά μεταξύ
    subjects (βιβλιογραφικά καλύτερα cross-subject transferable από
    ό,τι το raw waveform) και ήδη υπολογίζονται από το FeatureExtractor
    αλλά προηγουμένως δεν τροφοδοτούσαν καθόλου το μοντέλο.

    EEG Handcrafted Feature Vector (416-dim)
        -> Linear -> LayerNorm -> ReLU -> Dropout
        -> Linear -> ReLU
        -> EEG Handcrafted Embedding (batch, embedding_dim)

    Χρησιμοποιείται LayerNorm αντί για BatchNorm (όπως και το EEG CNN
    Encoder) ώστε να μην εξαρτάται από population statistics του
    train set, κάτι που βοηθάει τη γενίκευση σε άγνωστα subjects.
    """

    def __init__(
        self,
        input_dim: int = 416,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.30,
    ):
        super().__init__()

        self.mlp = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, eeg_handcrafted_features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        eeg_handcrafted_features : torch.Tensor
            Shape (batch, input_dim)

        Returns
        -------
        torch.Tensor
            EEG handcrafted embedding, shape (batch, embedding_dim)
        """

        return self.mlp(eeg_handcrafted_features)
