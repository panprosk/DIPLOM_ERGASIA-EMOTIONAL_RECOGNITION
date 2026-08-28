import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Μετατρέπει ένα ακατέργαστο πολυκαναλικό χρονικό σήμα σε μια
    ακολουθία από "patch tokens", έτοιμη να τροφοδοτήσει έναν
    Transformer Encoder -- η ίδια λογική με το "patchify" του Vision
    Transformer (ViT), αλλά σε 1D χρονικά σήματα (παρόμοιο με
    PatchTST/EEG-Conformer προσεγγίσεις στη βιβλιογραφία).

    RAW SIGNAL (batch, in_channels, window_samples)
        -> Conv1d(kernel_size=patch_size, stride=patch_size)
        -> (batch, d_model, num_patches)
        -> transpose
        -> (batch, num_patches, d_model)   [sequence of patch tokens]

    Γιατί Conv1d patchify αντί για ένα token ανά time-step
    --------------------------------------------------------
    Ένα πλήρες 6-δευτερόλεπτο παράθυρο έχει 768 δείγματα. Αν κάθε
    time-step γινόταν ξεχωριστό token, η self-attention (O(n^2)) θα
    ήταν υπολογιστικά πολύ ακριβή για CPU-only training σε πλήρες
    32-subject dataset. Ομαδοποιώντας σε patches (π.χ. 32 δείγματα/
    patch -> 24 tokens) μειώνεται δραστικά το μήκος ακολουθίας
    διατηρώντας παράλληλα τοπικά μοτίβα μέσα σε κάθε patch (μέσω του
    conv kernel), κάνοντας το training εφικτό σε CPU.
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int = 64,
        patch_size: int = 32,
    ):
        super().__init__()

        self.patch_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, in_channels, window_samples)

        Returns
        -------
        torch.Tensor, shape (batch, num_patches, d_model)
        """

        tokens = self.patch_conv(x)          # (batch, d_model, num_patches)
        tokens = tokens.transpose(1, 2)       # (batch, num_patches, d_model)

        return tokens
