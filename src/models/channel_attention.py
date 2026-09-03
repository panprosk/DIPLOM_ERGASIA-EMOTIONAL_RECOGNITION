import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style Channel Attention (Hu et al., 2018),
    προσαρμοσμένο σε 1D feature maps.

    Κίνητρο (cross-subject generalization): σε EEG-based emotion
    recognition όχι όλα τα κανάλια/feature-maps είναι εξίσου χρήσιμα
    για το task, και η σημασία τους μπορεί να διαφέρει ανά subject.
    Ένα learned per-channel gate επιτρέπει στο δίκτυο να "ενισχύσει"
    δυναμικά τα πιο emotion-discriminative κανάλια και να "καταστείλει"
    channels που κωδικοποιούν κυρίως subject-specific θόρυβο -- ιδέα
    που χρησιμοποιείται και σε πρόσφατη βιβλιογραφία cross-subject EEG
    emotion recognition (π.χ. CA-DASCLNet, Su et al. 2026) σε
    συνδυασμό με domain-adversarial training.

    RAW input (batch, channels, length)
        -> Global Average Pool (squeeze)   -> (batch, channels)
        -> FC reduce -> ReLU -> FC expand -> Sigmoid (excitation)
        -> (batch, channels, 1)
        -> scale το αρχικό input (channel-wise)

    Το τελευταίο υπολογισμένο attention gate αποθηκεύεται στο
    `self.last_attention_weights` (χωρίς gradient) για μελλοντική
    ερμηνευσιμότητα (π.χ. SHG-Agent ανάλυση ποια EEG κανάλια/feature
    maps "εμπιστεύεται" περισσότερο το μοντέλο).
    """

    def __init__(self, num_channels: int, reduction: int = 8):
        super().__init__()

        reduced = max(num_channels // reduction, 4)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.excitation = nn.Sequential(
            nn.Linear(num_channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, num_channels),
            nn.Sigmoid(),
        )

        self.last_attention_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, channels, length)

        Returns
        -------
        torch.Tensor, shape (batch, channels, length) -- re-weighted input
        """

        batch, channels, _ = x.shape

        squeezed = self.avg_pool(x).view(batch, channels)   # (batch, channels)

        gate = self.excitation(squeezed)                    # (batch, channels)

        self.last_attention_weights = gate.detach()

        gate = gate.view(batch, channels, 1)                 # (batch, channels, 1)

        return x * gate
