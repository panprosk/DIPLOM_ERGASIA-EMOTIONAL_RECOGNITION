import torch
import torch.nn as nn

from src.models.channel_attention import ChannelAttention


class EEGCNNEncoder(nn.Module):
    """
    1D CNN Encoder για το EEG branch του Hybrid CNN-MLP.

    Το EEG είναι το πιο σύνθετο σήμα (32 κανάλια, υψηλή χρονική
    ανάλυση, έντονα μη γραμμικά πρότυπα). Οι πυρήνες Conv1D μαθαίνουν
    αυτόματα τοπικά χρονικά μοτίβα και σχέσεις μεταξύ καναλιών.

    RAW EEG (batch, in_channels, window_samples)
        -> Channel Attention (ποια EEG ηλεκτρόδια είναι πιο χρήσιμα)
        -> 3x [Conv1D + GroupNorm + ReLU (+ MaxPool) (+ Channel Attention)]
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

    NOTE (Channel Attention): προστέθηκε ένα Squeeze-and-Excitation
    style gate (βλ. src/models/channel_attention.py) τόσο στα raw
    EEG κανάλια (32 ηλεκτρόδια) όσο και στα learned feature maps του
    πρώτου conv block. Ιδέα από πρόσφατη cross-subject EEG emotion
    recognition βιβλιογραφία (π.χ. CA-DASCLNet, Su et al. 2026): το
    δίκτυο μαθαίνει να δίνει μεγαλύτερο βάρος σε κανάλια/feature-maps
    που είναι πιο emotion-discriminative και λιγότερο σε αυτά που
    κωδικοποιούν κυρίως subject-specific θόρυβο.
    """

    def __init__(
        self,
        in_channels: int = 32,
        embedding_dim: int = 128,
        dropout: float = 0.30,
    ):
        super().__init__()

        # Channel Attention πάνω στα raw EEG ηλεκτρόδια (πριν οποιοδήποτε
        # conv), ώστε το δίκτυο να μάθει ποια ηλεκτρόδια είναι πιο
        # χρήσιμα για το emotion recognition task.
        self.input_channel_attention = ChannelAttention(
            num_channels=in_channels, reduction=4
        )

        self.conv_block_1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        # Channel Attention πάνω στα learned feature maps (64 κανάλια)
        # του πρώτου conv block (SE-style).
        self.block1_channel_attention = ChannelAttention(
            num_channels=64, reduction=8
        )

        self.conv_block_2 = nn.Sequential(
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

        x = self.input_channel_attention(eeg)   # (batch, in_channels, L)
        x = self.conv_block_1(x)                # (batch, 64, L/2)
        x = self.block1_channel_attention(x)     # (batch, 64, L/2)
        x = self.conv_block_2(x)                # (batch, 256, L')
        x = self.global_pool(x)                 # (batch, 256, 1)
        x = x.squeeze(-1)                       # (batch, 256)
        x = self.projection(x)                  # (batch, embedding_dim)

        return x

