import torch
import torch.nn as nn

from src.models.eeg_cnn_encoder import EEGCNNEncoder
from src.models.eeg_handcrafted_encoder import EEGHandcraftedEncoder
from src.models.physio_mlp_encoder import PhysioMLPEncoder
from src.models.gated_attention_fusion import GatedAttentionFusion
from src.models.gradient_reversal import GradientReversalLayer


class HybridCNNMLP(nn.Module):
    """
    Hybrid CNN-MLP — το πρώτο και βασικό μοντέλο της διπλωματικής.

    Πάνω σε αυτό θα συγκριθούν όλα τα επόμενα νευρωνικά δίκτυα
    (Cross-Attention Transformer, Mamba, GNN, AGAT-AGS).

    Architecture
    ------------
    Branch A (EEG):
        RAW EEG -> 1D CNN Encoder -> Global Average Pooling
                -> Linear Projection(128) -> EEG-CNN Embedding

        [Προαιρετικό εμπλουτισμό] Handcrafted EEG χαρακτηριστικά
        (band powers κ.λπ., 416-dim) -> MLP Encoder -> EEG-Handcrafted
        Embedding. Τα δύο EEG embeddings συνδυάζονται (concat + linear)
        σε ένα ενιαίο EEG Embedding, εμπλουτισμένο με πιο σταθερά
        (cross-subject robust) χαρακτηριστικά — προστέθηκε επειδή το
        raw-only CNN branch παρουσίαζε πολύ έντονο overfitting σε
        subject-specific waveform patterns.

    Branch B (EDA + PPG handcrafted features):
        Feature Vector -> MLP Encoder
                -> Linear Projection(128) -> Physiological Embedding

    Fusion:
        EEG Embedding, Physiological Embedding
            -> Gated Attention Fusion
            -> Residual Connection
            -> Layer Normalization
            -> Dropout
            -> Fully Connected
            -> Softmax

    [Προαιρετικό] Domain-Adversarial Subject-Invariance:
        Αν δοθεί `num_subjects`, προστίθεται ένας μικρός "subject
        classifier" (μέσω Gradient Reversal Layer) πάνω στο fused
        embedding. Κατά το training, ο encoder εκπαιδεύεται να παράγει
        embeddings από τα οποία ΔΕΝ μπορεί να προβλεφθεί η ταυτότητα
        του subject (adversarial), αναγκάζοντάς τον να μάθει πιο
        subject-invariant (και άρα καλύτερα γενικεύσιμα) χαρακτηριστικά.
        Αυτό αντιμετωπίζει άμεσα το πρόβλημα του τίτλου της
        διπλωματικής: Cross-Subject Generalization.

    Interpretability (για τον SHG-Agent)
    -------------------------------------
    Εκτός από τα logits/probabilities, το μοντέλο επιστρέφει και τα
    EEG Gate / Physio Gate βάρη ανά δείγμα, ώστε ο Scientific SHG-Agent
    να μπορεί να αναλύσει τη συνεισφορά κάθε modality (π.χ. αν η
    συμβολή του EEG αυξάνεται σε συγκεκριμένα subjects).
    """

    def __init__(
        self,
        eeg_channels: int = 32,
        physio_feature_dim: int = 22,
        embedding_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.30,
        eeg_handcrafted_dim: int | None = None,
        num_subjects: int | None = None,
    ):
        super().__init__()

        self.eeg_encoder = EEGCNNEncoder(
            in_channels=eeg_channels,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

        # --- Προαιρετικό EEG handcrafted-feature branch ---
        self.eeg_handcrafted_dim = eeg_handcrafted_dim

        if eeg_handcrafted_dim is not None and eeg_handcrafted_dim > 0:

            self.eeg_handcrafted_encoder = EEGHandcraftedEncoder(
                input_dim=eeg_handcrafted_dim,
                embedding_dim=embedding_dim,
                dropout=dropout,
            )

            self.eeg_fusion_layer = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )

        else:
            self.eeg_handcrafted_encoder = None
            self.eeg_fusion_layer = None

        self.physio_encoder = PhysioMLPEncoder(
            input_dim=physio_feature_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

        self.fusion = GatedAttentionFusion(
            embedding_dim=embedding_dim
        )

        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_classes),
        )

        # --- Προαιρετικός Domain-Adversarial Subject Classifier ---
        if num_subjects is not None and num_subjects > 0:

            self.grl = GradientReversalLayer(lambda_=1.0)

            self.subject_classifier = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(embedding_dim // 2, num_subjects),
            )

        else:
            self.grl = None
            self.subject_classifier = None

    def forward(
        self,
        eeg: torch.Tensor,
        physio_features: torch.Tensor,
        eeg_handcrafted_features: torch.Tensor | None = None,
        grl_lambda: float = 1.0,
    ) -> dict:
        """
        Parameters
        ----------
        eeg                      : torch.Tensor, shape (batch, eeg_channels, window_samples)
        physio_features          : torch.Tensor, shape (batch, physio_feature_dim)
        eeg_handcrafted_features : torch.Tensor, shape (batch, eeg_handcrafted_dim), προαιρετικό
        grl_lambda               : float, ένταση του gradient reversal (0 = χωρίς adversarial effect)

        Returns
        -------
        dict with:
            logits           : (batch, num_classes)   raw scores (χρησιμοποιούνται με CrossEntropyLoss)
            probs            : (batch, num_classes)   softmax probabilities
            eeg_embedding    : (batch, embedding_dim)
            physio_embedding : (batch, embedding_dim)
            eeg_gate         : (batch,)  μέσο βάρος εμπιστοσύνης στο EEG ανά δείγμα
            physio_gate      : (batch,)  μέσο βάρος εμπιστοσύνης στα physio features ανά δείγμα
            subject_logits   : (batch, num_subjects) -- μόνο αν το μοντέλο έχει adversarial head
        """

        eeg_embedding = self.eeg_encoder(eeg)

        if self.eeg_handcrafted_encoder is not None and eeg_handcrafted_features is not None:

            eeg_handcrafted_embedding = self.eeg_handcrafted_encoder(eeg_handcrafted_features)

            eeg_embedding = self.eeg_fusion_layer(
                torch.cat([eeg_embedding, eeg_handcrafted_embedding], dim=-1)
            )

        physio_embedding = self.physio_encoder(physio_features)

        fused, eeg_gate = self.fusion(eeg_embedding, physio_embedding)

        # Residual Connection: διατηρεί την πληροφορία των αρχικών
        # embeddings και βελτιώνει τη ροή των gradients.
        residual = fused + eeg_embedding + physio_embedding

        normalized = self.layer_norm(residual)

        dropped = self.dropout(normalized)

        logits = self.classifier(dropped)

        probs = torch.softmax(logits, dim=-1)

        output = {
            "logits": logits,
            "probs": probs,
            "eeg_embedding": eeg_embedding,
            "physio_embedding": physio_embedding,
            "eeg_gate": eeg_gate.mean(dim=-1),
            "physio_gate": 1.0 - eeg_gate.mean(dim=-1),
        }

        if self.subject_classifier is not None:

            self.grl.lambda_ = grl_lambda

            reversed_embedding = self.grl(normalized)

            output["subject_logits"] = self.subject_classifier(reversed_embedding)

        return output
