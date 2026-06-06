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
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3), # ΑΥΞΗΘΗΚΕ
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.2),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2), # ΑΥΞΗΘΗΚΕ
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.2),

            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1), # ΑΥΞΗΘΗΚΕ
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.2),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.proj = nn.Sequential(
            nn.Linear(128, out_dim), # ΑΛΛΑΞΕ ΑΠΟ 64
            nn.ReLU(),
            nn.Dropout(0.2),
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
        feature_dims: dict = None, # π.χ. {"eeg": 352, "eda": 8, "ppg": 8}
        cnn_out_dim: int = 64,
        hidden_dim: int = 128,
        n_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()

        if signals is None:
            signals = ["EEG"]
        
        # Μετατροπή σε lower case για συνέπεια με το dataset
        self.signals = [s.upper() for s in signals]
        self.cnn_out_dim = cnn_out_dim

        # 1. CNN Branch (Raw signals)
        self.cnn_encoders = nn.ModuleDict({
            sig: CNNEncoder(
                in_channels=self.SIGNAL_CHANNELS[sig],
                out_dim=cnn_out_dim,
                dropout=dropout,
            )
            for sig in self.signals
        })

        # 2. Feature Branch (Handcrafted features)
        self.feature_dims = feature_dims if feature_dims else {}
        total_feat_dim = sum([self.feature_dims.get(sig.lower(), 0) for sig in self.signals])
        
        if total_feat_dim > 0:
            self.feature_mlp = nn.Sequential(
                nn.Linear(total_feat_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.feat_out_dim = hidden_dim
        else:
            self.feature_mlp = None
            self.feat_out_dim = 0

        # 3. Fusion & Classifier
        fusion_dim = (cnn_out_dim * len(self.signals)) + self.feat_out_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout / 2),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),

            nn.Linear(hidden_dim // 2, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        # ...existing code...
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, batch: dict) -> dict:
        
        # CNN Branch
        cnn_features = []
        for sig in self.signals:
            x_raw = batch[f"{sig.lower()}_raw"]
            enc = self.cnn_encoders[sig](x_raw)
            cnn_features.append(enc)
        
        fused_cnn = torch.cat(cnn_features, dim=-1)

        # Feature Branch
        if self.feature_mlp is not None:
            feat_list = []
            for sig in self.signals:
                feat_list.append(batch[f"{sig.lower()}_feat"])
            
            x_feat = torch.cat(feat_list, dim=-1)
            fused_feat = self.feature_mlp(x_feat)
            
            # Final Fusion
            fused_all = torch.cat([fused_cnn, fused_feat], dim=-1)
        else:
            fused_all = fused_cnn

        logits = self.classifier(fused_all)
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values

        return {
            "logits": logits,
            "probs": probs,
            "confidence": confidence,
            "features": fused_all,
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