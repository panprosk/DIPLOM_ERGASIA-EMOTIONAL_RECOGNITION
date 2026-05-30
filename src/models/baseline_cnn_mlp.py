"""
baseline_cnn_mlp.py

Regularized Hybrid CNN-MLP για cross-subject emotion recognition.
Optimized for DEAP generalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# CNN Encoder
# ──────────────────────────────────────────────

class CNNEncoder(nn.Module):
    """
    Regularized 1D CNN encoder.
    """

    def __init__(
        self,
        in_channels: int,
        out_dim: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.conv_block = nn.Sequential(

            # Block 1
            nn.Conv1d(in_channels, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(dropout),

            # Block 2
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(dropout),

            # Block 3
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(dropout),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.proj = nn.Sequential(
            nn.Linear(64, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv_block(x)
        x = self.gap(x)
        x = x.squeeze(-1)
        x = self.proj(x)

        return x


# ──────────────────────────────────────────────
# Hybrid CNN-MLP
# ──────────────────────────────────────────────

class HybridCNNMLP(nn.Module):

    SIGNAL_CHANNELS = {
        "EEG": 32,
        "EDA": 1,
        "PPG": 1,
    }

    def __init__(
        self,
        signals: list = None,
        cnn_out_dim: int = 64,
        hidden_dim: int = 128,
        n_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()

        if signals is None:
            signals = ["EEG"]

        self.signals = signals
        self.cnn_out_dim = cnn_out_dim

        self.encoders = nn.ModuleDict({
            sig: CNNEncoder(
                in_channels=self.SIGNAL_CHANNELS[sig],
                out_dim=cnn_out_dim,
                dropout=dropout,
            )
            for sig in signals
        })

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

        for m in self.modules():

            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, batch: dict) -> dict:

        key_map = {
            "EEG": "eeg",
            "EDA": "eda",
            "PPG": "ppg",
        }

        features = []

        for sig in self.signals:

            x = batch[key_map[sig]]
            enc = self.encoders[sig](x)

            features.append(enc)

        fused = torch.cat(features, dim=-1)

        logits = self.classifier(fused)

        probs = F.softmax(logits, dim=-1)

        confidence = probs.max(dim=-1).values

        return {
            "logits": logits,
            "probs": probs,
            "confidence": confidence,
            "features": fused,
        }

    def get_num_params(self) -> int:

        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────

def build_model(config: dict) -> HybridCNNMLP:

    return HybridCNNMLP(

        signals=config.get("signals", ["EEG"]),

        cnn_out_dim=config.get("cnn_out_dim", 64),

        hidden_dim=config.get("hidden_dim", 128),

        n_classes=config.get("n_classes", 2),

        dropout=config.get("dropout", 0.5),
    )