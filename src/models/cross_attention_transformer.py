import torch
import torch.nn as nn

from src.models.patch_embedding import PatchEmbedding
from src.models.positional_encoding import SinusoidalPositionalEncoding
from src.models.cross_attention_block import CrossAttentionBlock
from src.models.gradient_reversal import GradientReversalLayer


class CrossAttentionTransformer(nn.Module):
    """
    Hierarchical Multimodal Cross-Attention Transformer.

    Υλοποιεί το 2ο προτεινόμενο μοντέλο της διπλωματικής (βλ.
    ΔΙΠΛΩΜΑΤΙΚΗ_ΕΡΓΑΣΙΑ.txt, "Cross-Attention Transformer Multimodal
    Model" / ΣΕΙΡΑ_ΥΛΟΠΟΙΗΣΗΣ.txt Βήμα 12).

    NOTE σχετικά με τα modalities: το spec αναφέρει EEG/ECG/EDA, αλλά
    το πραγματικό DEAP-based pipeline αυτής της διπλωματικής παράγει
    EEG/EDA/PPG (το PPG χρησιμοποιείται στη θέση του ECG, όπως και
    στο πρώτο μοντέλο, Hybrid CNN-MLP).

    Architecture ("Hierarchical": πρώτα per-modality self-attention,
    μετά cross-modal attention -- δύο επίπεδα ιεραρχίας)
    --------------------------------------------------------------
    Επίπεδο 1 -- Per-Modality Temporal Self-Attention:
        RAW EEG  (batch, 32, W) -> Patch Embedding -> + PosEnc
                 -> Transformer Encoder (self-attention, N layers)
                 -> EEG tokens

        RAW EDA+PPG (batch, 2, W) -> Patch Embedding -> + PosEnc
                 -> Transformer Encoder (self-attention, N layers)
                 -> Physio tokens

    Επίπεδο 2 -- Cross-Modal Attention:
        EEG tokens, Physio tokens
                 -> M x CrossAttentionBlock (αμφίδρομο cross-attention)
                 -> EEG tokens (physio-aware), Physio tokens (eeg-aware)

    Pooling & Classification:
        Mean-pool tokens ανά modality -> EEG Embedding, Physio Embedding
                 -> concat -> Fully Connected -> Softmax

    [Προαιρετικό] Domain-Adversarial Subject-Invariance:
        Ίδια τεχνική με το Hybrid CNN-MLP (Gradient Reversal Layer +
        subject classifier) -- προστέθηκε επειδή στο πρώτο μοντέλο
        αποδείχθηκε ουσιαστική για τη μείωση cross-subject overfitting.
        Ενεργοποιείται μόνο αν δοθεί `num_subjects`.

    Interpretability
    ----------------
    Επιστρέφει επίσης `cross_attention_weight` (πόσο "κοιτάει" το EEG
    το physio branch, μέσος όρος πάνω σε heads/tokens) ως ανάλογο του
    EEG/Physio Gate του πρώτου μοντέλου, για συγκρίσιμη ανάλυση από
    τον Scientific SHG-Agent.
    """

    def __init__(
        self,
        eeg_channels: int = 32,
        physio_channels: int = 2,
        d_model: int = 64,
        num_heads: int = 4,
        num_self_attn_layers: int = 2,
        num_cross_attn_layers: int = 1,
        dim_feedforward: int = 128,
        patch_size: int = 32,
        num_classes: int = 2,
        dropout: float = 0.30,
        num_subjects: int | None = None,
    ):
        super().__init__()

        self.eeg_patch_embed = PatchEmbedding(
            in_channels=eeg_channels, d_model=d_model, patch_size=patch_size,
        )
        self.physio_patch_embed = PatchEmbedding(
            in_channels=physio_channels, d_model=d_model, patch_size=patch_size,
        )

        self.eeg_pos_encoding = SinusoidalPositionalEncoding(d_model=d_model)
        self.physio_pos_encoding = SinusoidalPositionalEncoding(d_model=d_model)

        # --- Επίπεδο 1: Per-modality self-attention (Transformer Encoder) ---
        eeg_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.eeg_self_attention = nn.TransformerEncoder(
            eeg_encoder_layer, num_layers=num_self_attn_layers,
        )

        physio_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.physio_self_attention = nn.TransformerEncoder(
            physio_encoder_layer, num_layers=num_self_attn_layers,
        )

        # --- Επίπεδο 2: Cross-modal attention ---
        self.cross_attention_blocks = nn.ModuleList([
            CrossAttentionBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(num_cross_attn_layers)
        ])

        embedding_dim = d_model

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes),
        )

        # --- Προαιρετικός Domain-Adversarial Subject Classifier ---
        if num_subjects is not None and num_subjects > 0:

            self.grl = GradientReversalLayer(lambda_=1.0)

            self.subject_classifier = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embedding_dim, num_subjects),
            )

        else:
            self.grl = None
            self.subject_classifier = None

    def forward(
        self,
        eeg: torch.Tensor,
        eda: torch.Tensor,
        ppg: torch.Tensor,
        grl_lambda: float = 1.0,
    ) -> dict:
        """
        Parameters
        ----------
        eeg        : torch.Tensor, shape (batch, eeg_channels, window_samples)
        eda        : torch.Tensor, shape (batch, 1, window_samples)
        ppg        : torch.Tensor, shape (batch, 1, window_samples)
        grl_lambda : float, ένταση του gradient reversal (adversarial training)

        Returns
        -------
        dict with:
            logits                 : (batch, num_classes)
            probs                  : (batch, num_classes)
            eeg_embedding          : (batch, d_model)
            physio_embedding       : (batch, d_model)
            cross_attention_weight : (batch,) -- μέσο "βάρος προσοχής" EEG->Physio
            subject_logits         : (batch, num_subjects) -- μόνο αν υπάρχει adversarial head
        """

        physio = torch.cat([eda, ppg], dim=1)   # (batch, 2, window_samples)

        eeg_tokens = self.eeg_patch_embed(eeg)
        eeg_tokens = self.eeg_pos_encoding(eeg_tokens)
        eeg_tokens = self.eeg_self_attention(eeg_tokens)

        physio_tokens = self.physio_patch_embed(physio)
        physio_tokens = self.physio_pos_encoding(physio_tokens)
        physio_tokens = self.physio_self_attention(physio_tokens)

        cross_attn_weight = None

        for block in self.cross_attention_blocks:
            eeg_tokens, physio_tokens, cross_attn_weight = block(eeg_tokens, physio_tokens)

        eeg_embedding = eeg_tokens.mean(dim=1)
        physio_embedding = physio_tokens.mean(dim=1)

        fused = torch.cat([eeg_embedding, physio_embedding], dim=-1)

        logits = self.classifier(fused)
        probs = torch.softmax(logits, dim=-1)

        # Μέσο attention weight (πάνω σε query-tokens και key-tokens) ως
        # ερμηνεύσιμος δείκτης "πόσο εμπιστεύεται" το EEG το physio branch.
        cross_attention_weight = (
            cross_attn_weight.mean(dim=(1, 2))
            if cross_attn_weight is not None
            else torch.zeros(eeg.size(0), device=eeg.device)
        )

        output = {
            "logits": logits,
            "probs": probs,
            "eeg_embedding": eeg_embedding,
            "physio_embedding": physio_embedding,
            "cross_attention_weight": cross_attention_weight,
        }

        if self.subject_classifier is not None:

            self.grl.lambda_ = grl_lambda

            reversed_embedding = self.grl(fused)

            output["subject_logits"] = self.subject_classifier(reversed_embedding)

        return output
