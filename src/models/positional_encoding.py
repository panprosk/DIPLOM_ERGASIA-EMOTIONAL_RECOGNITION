import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Κλασική (μη-εκπαιδεύσιμη) sinusoidal positional encoding, όπως στο
    "Attention is All You Need" (Vaswani et al., 2017).

    Χρησιμοποιείται (αντί για learned positional embeddings) επειδή:
    - Δεν προσθέτει επιπλέον εκπαιδεύσιμες παραμέτρους (σημαντικό σε
      μικρό dataset/CPU-only training, όπου η υπερβολική χωρητικότητα
      οδηγεί εύκολα σε overfitting -- ακριβώς το πρόβλημα που
      παρατηρήθηκε στο πρώτο μοντέλο, Hybrid CNN-MLP).
    - Γενικεύεται σε οποιοδήποτε μήκος ακολουθίας χωρίς να χρειάζεται
      re-training, χρήσιμο αν αλλάξει το patch size σε μελλοντικά
      πειράματα.
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register ως buffer (όχι parameter) -- δεν εκπαιδεύεται, αλλά
        # μεταφέρεται αυτόματα με .to(device) και σώζεται στο state_dict.
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, d_model)

        Returns
        -------
        torch.Tensor, ίδιο shape, με προστιθέμενο positional encoding
        """

        seq_len = x.size(1)

        return x + self.pe[:, :seq_len, :]
