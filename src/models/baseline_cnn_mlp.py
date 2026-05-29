"""
baseline_cnn_mlp.py

Hybrid CNN-MLP για multimodal emotion recognition.
- CNN branch: εξάγει temporal features από raw signals (EEG, EDA, PPG)
- MLP branch: επεξεργάζεται handcrafted features (αν υπάρχουν)
- Fusion: concatenation → classifier

Input shapes:
    eeg: (batch, 32, 512)   # 32 channels, 4sec @ 128Hz
    eda: (batch, 1,  512)
    ppg: (batch, 1,  512)

Output: (batch, 2)  # binary valence logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Building block: 1D CNN Encoder
# ──────────────────────────────────────────────

class CNNEncoder(nn.Module):
    """
    1D CNN encoder για ένα modality.
    Εφαρμόζεται ξεχωριστά σε EEG, EDA, PPG.
    """

    def __init__(self,
                 in_channels: int,
                 out_dim: int = 128,
                 dropout: float = 0.3):
        super().__init__()

        self.conv_block = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),   # 512 → 256
            nn.Dropout(dropout),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),   # 256 → 128
            nn.Dropout(dropout),

            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),   # 128 → 64
            nn.Dropout(dropout),
        )

        # Global Average Pooling → fixed size ανεξαρτήτως input length
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.proj = nn.Sequential(
            nn.Linear(128, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, channels, time)
        returns: (batch, out_dim)
        """
        x = self.conv_block(x)   # (batch, 128, T')
        x = self.gap(x)          # (batch, 128, 1)
        x = x.squeeze(-1)        # (batch, 128)
        x = self.proj(x)         # (batch, out_dim)
        return x


# ──────────────────────────────────────────────
# Main model: Hybrid CNN-MLP
# ──────────────────────────────────────────────

class HybridCNNMLP(nn.Module):
    """
    Multimodal Hybrid CNN-MLP.

    Κάθε modality έχει το δικό του CNN encoder.
    Τα features από όλα τα modalities concatenate
    και περνούν από MLP classifier.

    Args:
        signals:     list από ['EEG', 'EDA', 'PPG']
        cnn_out_dim: output dim κάθε CNN encoder
        hidden_dim:  hidden size του MLP
        n_classes:   2 για binary valence
        dropout:     dropout rate
    """

    # Channel counts ανά modality
    SIGNAL_CHANNELS = {
        "EEG": 32,
        "EDA": 1,
        "PPG": 1,
    }

    def __init__(self,
                 signals: list = None,
                 cnn_out_dim: int = 128,
                 hidden_dim: int = 256,
                 n_classes: int = 2,
                 dropout: float = 0.3):
        super().__init__()

        if signals is None:
            signals = ["EEG", "EDA", "PPG"]

        self.signals     = signals
        self.cnn_out_dim = cnn_out_dim

        # Ένας CNN encoder ανά modality
        self.encoders = nn.ModuleDict({
            sig: CNNEncoder(
                in_channels=self.SIGNAL_CHANNELS[sig],
                out_dim=cnn_out_dim,
                dropout=dropout,
            )
            for sig in signals
        })

        # MLP Classifier
        fusion_dim = cnn_out_dim * len(signals)

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """He initialization για Conv, Xavier για Linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, batch: dict) -> dict:
        """
        batch: {
            'eeg': (B, 32, T),
            'eda': (B, 1,  T),
            'ppg': (B, 1,  T),
        }

        Returns:
            {
                'logits':     (B, 2),
                'probs':      (B, 2),
                'confidence': (B,),
                'features':   (B, fusion_dim)
            }
        """
        # Map key names (lowercase) → signal names (uppercase)
        key_map = {"EEG": "eeg", "EDA": "eda", "PPG": "ppg"}

        features = []
        for sig in self.signals:
            x   = batch[key_map[sig]]          # (B, C, T)
            enc = self.encoders[sig](x)        # (B, cnn_out_dim)
            features.append(enc)

        fused  = torch.cat(features, dim=-1)   # (B, fusion_dim)
        logits = self.classifier(fused)        # (B, 2)
        probs  = F.softmax(logits, dim=-1)     # (B, 2)
        conf   = probs.max(dim=-1).values      # (B,)

        return {
            "logits":     logits,
            "probs":      probs,
            "confidence": conf,
            "features":   fused,
        }

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────
# Factory function (χρησιμοποιείται από τον agent)
# ──────────────────────────────────────────────

def build_model(config: dict) -> HybridCNNMLP:
    """
    Δημιουργεί μοντέλο από config dict.

    Παράδειγμα config:
        {
            "model":       "cnn_mlp",
            "signals":     ["EEG", "EDA"],
            "cnn_out_dim": 128,
            "hidden_dim":  256,
            "dropout":     0.3,
            "n_classes":   2
        }
    """
    return HybridCNNMLP(
        signals     = config.get("signals",     ["EEG", "EDA", "PPG"]),
        cnn_out_dim = config.get("cnn_out_dim", 128),
        hidden_dim  = config.get("hidden_dim",  256),
        n_classes   = config.get("n_classes",   2),
        dropout     = config.get("dropout",     0.3),
    )
