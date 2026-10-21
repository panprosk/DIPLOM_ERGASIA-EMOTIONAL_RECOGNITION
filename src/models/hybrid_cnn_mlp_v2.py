import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.channel_attention import ChannelAttention
from src.models.gradient_reversal import GradientReversalLayer


def _mlp_block(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    )


class EEGTemporalSpatialCNN(nn.Module):
    """EEGNet-like temporal/spatial stem + multi-scale Conv1D + SE + residual DS-conv."""

    def __init__(self, eeg_channels: int = 32, embedding_dim: int = 96, dropout: float = 0.30):
        super().__init__()

        self.temporal = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 63), padding=(0, 31), bias=False),
            nn.GroupNorm(4, 16),
            nn.ELU(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(eeg_channels, 1), groups=16, bias=False),
            nn.GroupNorm(8, 32),
            nn.ELU(),
        )
        self.pool1 = nn.AvgPool2d((1, 4))

        self.multi_scale = nn.ModuleList(
            [nn.Conv1d(32, 32, kernel_size=k, padding=k // 2) for k in (7, 15, 31)]
        )

        self.projection = nn.Sequential(
            nn.Conv1d(96, 128, kernel_size=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )
        self.se = ChannelAttention(128)
        self.pool2 = nn.AvgPool1d(2)

        self.res_depthwise = nn.Conv1d(128, 128, kernel_size=7, padding=3, groups=128)
        self.res_pointwise = nn.Conv1d(128, 128, kernel_size=1)
        self.res_norm = nn.GroupNorm(8, 128)
        self.res_act = nn.GELU()

        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        x = eeg.unsqueeze(1)                     # (B, 1, C, T)
        x = self.spatial(self.temporal(x))       # (B, 32, 1, T)
        x = self.pool1(x).squeeze(2)             # (B, 32, T/4)

        x = torch.cat([conv(x) for conv in self.multi_scale], dim=1)  # (B, 96, T/4)
        x = self.se(self.projection(x))
        x = self.pool2(x)

        residual = x
        x = self.res_pointwise(self.res_depthwise(x))
        x = self.res_act(self.res_norm(x))
        x = self.drop(x) + residual

        pooled = torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=-1)  # (B, 256)
        return self.head(pooled)


class HybridCNNMLPv2(nn.Module):
    """
    Hybrid CNN-MLP v2.

    Raw EEG (32x768)  -> temporal/spatial multi-scale CNN -> 96
    EEG spectral (256) -> MLP 256-128-96                  -> 96
        concat -> Linear 192->96 -> LayerNorm             = EEG representation
    EDA+PPG features (22) -> MLP 22-64-96                 = physio representation
    Gate g = sigmoid(W[eeg, physio]); fused = LayerNorm(g*eeg + (1-g)*physio)
    (χωρίς επιπλέον residual που εξουδετέρωνε το gate)
    Classifier: [fused, eeg, physio, eeg*physio] 384 -> 128 -> 48 -> 2

    Το output dict είναι συμβατό με το HybridCNNMLP (v1).
    """

    def __init__(
        self,
        eeg_channels: int = 32,
        physio_feature_dim: int = 22,
        embedding_dim: int = 96,
        num_classes: int = 2,
        dropout: float = 0.30,
        eeg_handcrafted_dim: int | None = 256,
        num_subjects: int | None = None,
    ):
        super().__init__()

        self.eeg_handcrafted_dim = eeg_handcrafted_dim

        self.eeg_cnn = EEGTemporalSpatialCNN(eeg_channels, embedding_dim, dropout)

        if eeg_handcrafted_dim:
            self.spectral_mlp = nn.Sequential(
                _mlp_block(eeg_handcrafted_dim, 128, dropout),
                nn.Linear(128, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.eeg_fusion = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.LayerNorm(embedding_dim),
            )
        else:
            self.spectral_mlp = None
            self.eeg_fusion = None

        self.physio_mlp = nn.Sequential(
            _mlp_block(physio_feature_dim, 64, dropout),
            nn.Linear(64, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gate = nn.Linear(embedding_dim * 2, embedding_dim)
        self.fusion_norm = nn.LayerNorm(embedding_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 4, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(48, num_classes),
        )

        self.contrastive_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 64),
        )

        if num_subjects is not None and num_subjects > 0:
            self.grl = GradientReversalLayer(lambda_=1.0)
            self.subject_classifier = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.GELU(),
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

        eeg_embedding = self.eeg_cnn(eeg)

        if self.spectral_mlp is not None and eeg_handcrafted_features is not None:
            spectral = self.spectral_mlp(eeg_handcrafted_features)
            eeg_embedding = self.eeg_fusion(torch.cat([eeg_embedding, spectral], dim=-1))

        physio_embedding = self.physio_mlp(physio_features)

        gate = torch.sigmoid(self.gate(torch.cat([eeg_embedding, physio_embedding], dim=-1)))
        fused = self.fusion_norm(gate * eeg_embedding + (1.0 - gate) * physio_embedding)

        logits = self.classifier(
            torch.cat(
                [fused, eeg_embedding, physio_embedding, eeg_embedding * physio_embedding],
                dim=-1,
            )
        )

        mean_gate = gate.mean(dim=-1)

        output = {
            "logits": logits,
            "probs": torch.softmax(logits, dim=-1),
            "eeg_embedding": eeg_embedding,
            "physio_embedding": physio_embedding,
            "contrastive_embedding": F.normalize(
                self.contrastive_projection(fused), p=2, dim=-1
            ),
            "eeg_gate": mean_gate,
            "physio_gate": 1.0 - mean_gate,
        }

        if self.subject_classifier is not None:
            self.grl.lambda_ = grl_lambda
            output["subject_logits"] = self.subject_classifier(self.grl(fused))

        return output
