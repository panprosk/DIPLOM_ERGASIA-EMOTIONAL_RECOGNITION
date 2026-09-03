from src.models.hybrid_cnn_mlp import HybridCNNMLP
from src.models.cross_attention_transformer import CrossAttentionTransformer


def build_model(model_name: str, config, num_subjects: int | None = None):
    """
    Δημιουργεί το κατάλληλο μοντέλο με βάση το config.

    Υποστηρίζονται προς το παρόν: "hybrid" (Hybrid CNN-MLP) και
    "cross_attention_transformer" (Multimodal Cross-Attention
    Transformer). Οι υπόλοιπες αρχιτεκτονικές (Mamba, GNN, AGAT-AGS)
    θα προστεθούν στη συνέχεια της Φάσης Β.

    Parameters
    ----------
    num_subjects : int, προαιρετικό
        Αριθμός distinct train subjects. Αν δοθεί, ενεργοποιείται ο
        domain-adversarial subject classifier (Gradient Reversal
        Layer) για βελτίωση της cross-subject γενίκευσης. Είναι
        data-dependent, γι' αυτό περνιέται ως ξεχωριστό argument
        (όχι μέσω config) — υπολογίζεται από τα πραγματικά subjects
        του train split.
    """

    if model_name == "hybrid":

        return HybridCNNMLP(
            eeg_channels=config.EEG_CHANNELS,
            physio_feature_dim=getattr(config, "PHYSIO_FEATURE_DIM", 22),
            embedding_dim=getattr(config, "EMBEDDING_DIM", 128),
            num_classes=config.NUM_CLASSES,
            dropout=config.DROPOUT,
            eeg_handcrafted_dim=getattr(config, "EEG_HANDCRAFTED_DIM", None),
            num_subjects=num_subjects,
        )

    if model_name == "cross_attention_transformer":

        return CrossAttentionTransformer(
            eeg_channels=config.EEG_CHANNELS,
            physio_channels=2,  # EDA + PPG
            d_model=getattr(config, "TRANSFORMER_D_MODEL", 64),
            num_heads=getattr(config, "TRANSFORMER_NUM_HEADS", 4),
            num_self_attn_layers=getattr(config, "TRANSFORMER_SELF_ATTN_LAYERS", 2),
            num_cross_attn_layers=getattr(config, "TRANSFORMER_CROSS_ATTN_LAYERS", 1),
            dim_feedforward=getattr(config, "TRANSFORMER_FF_DIM", 128),
            patch_size=getattr(config, "TRANSFORMER_PATCH_SIZE", 32),
            num_classes=config.NUM_CLASSES,
            dropout=getattr(config, "TRANSFORMER_DROPOUT", 0.20),
            num_subjects=num_subjects,
        )

    raise ValueError(f"Unknown model_name: {model_name}")
