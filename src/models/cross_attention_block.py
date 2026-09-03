import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """
    Ένα (διπλής κατεύθυνσης) Cross-Attention block μεταξύ δύο
    modalities (εδώ: EEG tokens <-> Physio tokens).

    EEG attends to Physio:
        eeg_out = LayerNorm(eeg + MultiHeadAttention(query=eeg, key=physio, value=physio))
        eeg_out = LayerNorm(eeg_out + FeedForward(eeg_out))

    Physio attends to EEG:
        physio_out = LayerNorm(physio + MultiHeadAttention(query=physio, key=eeg, value=eeg))
        physio_out = LayerNorm(physio_out + FeedForward(physio_out))

    Αυτό επιτρέπει σε κάθε modality να "ρωτήσει" το άλλο modality
    ποια στοιγμιότυπα/patches είναι πιο σχετικά (π.χ. ποιο physio
    patch συσχετίζεται περισσότερο με ποιο EEG patch), υλοποιώντας
    ακριβώς τη "δυναμική ανταλλαγή πληροφορίας" που περιγράφεται στο
    spec της διπλωματικής (Cross-Attention Transformer Multimodal
    Model).

    Επιστρέφει επίσης τα attention weights (μέσο όρο πάνω σε heads
    και query-tokens) ως ερμηνεύσιμο δείκτη -- ανάλογο του EEG/Physio
    Gate του Hybrid CNN-MLP -- ώστε ο Scientific SHG-Agent να μπορεί
    να αναλύσει πόσο "κοιτάει" κάθε modality το άλλο.
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        dropout: float = 0.30,
        attn_dropout: float | None = None,
    ):
        super().__init__()

        # ΣΗΜΕΙΩΣΗ (ενημερωμένη): δοκιμάστηκε χαμηλότερο attn_dropout=0.10
        # στο MultiheadAttention με την υπόθεση ότι το dropout=0.30 πάνω
        # στα attention weights προκαλούσε "κατάρρευση" σε ομοιόμορφη
        # κατανομή. Αποδείχτηκε όμως ότι ο δείκτης diagnostics που το
        # έδειχνε (mean πάνω σε softmax row) ήταν μαθηματικά ΠΑΝΤΑ
        # 1/num_keys ανεξαρτήτως πραγματικής κατανομής (διορθώθηκε στο
        # cross_attention_transformer.py). Το χαμηλότερο dropout απλά
        # αύξησε το overfitting και χειροτέρεψε το αποτέλεσμα, άρα το
        # attn_dropout επαναφέρεται στο ίδιο με το γενικό dropout
        # (default attn_dropout=None -> χρησιμοποιεί την τιμή του dropout).
        if attn_dropout is None:
            attn_dropout = dropout

        self.eeg_attends_physio = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=attn_dropout, batch_first=True
        )
        self.physio_attends_eeg = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=attn_dropout, batch_first=True
        )

        self.eeg_norm1 = nn.LayerNorm(d_model)
        self.physio_norm1 = nn.LayerNorm(d_model)

        self.eeg_ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.physio_ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

        self.eeg_norm2 = nn.LayerNorm(d_model)
        self.physio_norm2 = nn.LayerNorm(d_model)

    def forward(self, eeg_tokens: torch.Tensor, physio_tokens: torch.Tensor):
        """
        Parameters
        ----------
        eeg_tokens    : (batch, n_eeg_patches, d_model)
        physio_tokens : (batch, n_physio_patches, d_model)

        Returns
        -------
        eeg_out, physio_out : ίδια shapes με τα inputs
        eeg_to_physio_attn  : (batch, n_eeg_patches, n_physio_patches) attention weights
        """

        eeg_attended, eeg_to_physio_attn = self.eeg_attends_physio(
            query=eeg_tokens, key=physio_tokens, value=physio_tokens
        )
        eeg_out = self.eeg_norm1(eeg_tokens + eeg_attended)
        eeg_out = self.eeg_norm2(eeg_out + self.eeg_ff(eeg_out))

        physio_attended, _ = self.physio_attends_eeg(
            query=physio_tokens, key=eeg_tokens, value=eeg_tokens
        )
        physio_out = self.physio_norm1(physio_tokens + physio_attended)
        physio_out = self.physio_norm2(physio_out + self.physio_ff(physio_out))

        return eeg_out, physio_out, eeg_to_physio_attn
