import torch
import torch.nn as nn


class EEGCNNEncoder(nn.Module):
    """
    1D CNN Encoder για το EEG branch του Hybrid CNN-MLP.

    Το EEG είναι το πιο σύνθετο σήμα (32 κανάλια, υψηλή χρονική
    ανάλυση, έντονα μη γραμμικά πρότυπα). Οι πυρήνες Conv1D μαθαίνουν
    αυτόματα τοπικά χρονικά μοτίβα και σχέσεις μεταξύ καναλιών.

    RAW EEG (batch, in_channels, window_samples)
        -> 3x [Conv1D + GroupNorm + ReLU (+ MaxPool)]
        -> Global Average Pooling
        -> Linear Projection
        -> EEG Embedding (batch, embedding_dim)

    NOTE (cross-subject generalization): χρησιμοποιείται GroupNorm
    αντί για BatchNorm1d. Το BatchNorm μαθαίνει running mean/var από
    τα batches εκπαίδευσης (δηλαδή από συγκεκριμένα train subjects),
    και αυτά τα στατιστικά χρησιμοποιούνται και σε άγνωστα subjects
    κατά το evaluation· αν η κατανομή του σήματος διαφέρει σημαντικά
    μεταξύ subjects (πολύ συνηθισμένο σε EEG), αυτό το mismatch
    επιδεινώνει τη γενίκευση. Το GroupNorm κανονικοποιεί ανά δείγμα
    (δεν εξαρτάται από population statistics), άρα είναι subject-
    invariant και ταιριάζει καλύτερα σε cross-subject σενάρια.
    """

    def __init__(
        self,
        in_channels: int = 32,
        embedding_dim: int = 128,
        dropout: float = 0.30,
    ):
        super().__init__()

        self.conv_block = nn.Sequential(

            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=256),
            nn.ReLU(inplace=True),

            nn.Dropout(dropout),
        )

        # Global Average Pooling πάνω στη χρονική διάσταση
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.projection = nn.Linear(256, embedding_dim)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        eeg : torch.Tensor
            Shape (batch, in_channels, window_samples)

        Returns
        -------
        torch.Tensor
            EEG embedding, shape (batch, embedding_dim)
        """

        x = self.conv_block(eeg)          # (batch, 256, L')
        x = self.global_pool(x)           # (batch, 256, 1)
        x = x.squeeze(-1)                 # (batch, 256)
        x = self.projection(x)            # (batch, embedding_dim)

        return x
