"""
CT-ELCAN (Li et al., Bioengineering 2025, 12, 528): Cross-modal Transformer
+ Enhanced Learning-Classifying Adversarial Network (c-SAGAN + LCAN).

Προσαρμογή για DEAP με 3 modalities (EEG 32ch, GSR/EDA, PPG) αντί των
EEG/EMG/EOG/GSR του paper. Ό,τι ΔΕΝ ορίζει το paper (π.χ. μέγεθος ConvBlock,
διαστάσεις GAN, βάρη α/β/φ) σημειώνεται ρητά στα docstrings.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from src.models.gradient_reversal import GradientReversalLayer


def sinusoidal_encoding(length: int, dim: int) -> torch.Tensor:
    """Εξίσωση (2) του paper: PE[pos,2k]=sin, PE[pos,2k+1]=cos."""
    pos = torch.arange(1, length + 1, dtype=torch.float32).unsqueeze(1)
    k = torch.arange(0, dim, 2, dtype=torch.float32)
    div = torch.pow(10000.0, k / dim)
    pe = torch.zeros(length, dim)
    pe[:, 0::2] = torch.sin(pos / div)
    pe[:, 1::2] = torch.cos(pos / div)
    return pe


class CrossModalEncoderLayer(nn.Module):
    """
    Fig. 1: Q από το unimodal sequence, K/V από το global multimodal context
    X_F (concat των modalities στον χρονικό άξονα). Pre-norm (οι δύο "Norm"
    του σχήματος), multi-head cross-modal attention, Add, FFN, Add.
    """

    def __init__(self, dim: int, heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        kv = self.norm_kv(context)
        out, _ = self.attn(self.norm_q(x), kv, kv, need_weights=False)
        x = x + self.drop(out)
        return x + self.drop(self.ff(self.norm_ff(x)))


class CrossModalTransformer(nn.Module):
    """
    Modality-specific 1D pointwise conv -> + sinusoidal PE -> concat στον
    χρόνο (X_F) -> U στοιβαγμένα cross-modal encoder layers (κοινά βάρη για
    όλες τις modalities, ο global context X_F είναι σταθερός σε κάθε layer).
    Επιστρέφει λίστα Y_mi, καθένα (B, L, D).
    """

    def __init__(
        self,
        modality_channels: Sequence[int],
        seq_len: int,
        dim: int = 128,
        layers: int = 4,
        heads: int = 4,
        ff_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Conv1d(c, dim, kernel_size=1) for c in modality_channels
        )
        self.register_buffer("pe", sinusoidal_encoding(seq_len, dim), persistent=False)
        self.layers = nn.ModuleList(
            CrossModalEncoderLayer(dim, heads, ff_dim or 4 * dim, dropout)
            for _ in range(layers)
        )
        self.in_drop = nn.Dropout(dropout)

    def forward(self, signals: List[torch.Tensor]) -> List[torch.Tensor]:
        # signals[i]: (B, C_i, L)
        xs = [
            self.in_drop(proj(s).transpose(1, 2) + self.pe)
            for proj, s in zip(self.projections, signals)
        ]
        context = torch.cat(xs, dim=1)  # (B, m*L, D)
        for layer in self.layers:
            xs = [layer(x, context) for x in xs]
        return xs


class SelfAttention1d(nn.Module):
    """SAGAN self-attention (εξ. 7-9): f,g,h,v 1x1 convs, γ αρχικά 0."""

    def __init__(self, channels: int, use_sn: bool = False):
        super().__init__()
        wrap = spectral_norm if use_sn else (lambda m: m)
        c_bar = max(channels // 8, 1)
        c_half = max(channels // 2, 1)
        self.f = wrap(nn.Conv1d(channels, c_bar, 1))
        self.g = wrap(nn.Conv1d(channels, c_bar, 1))
        self.h = wrap(nn.Conv1d(channels, c_half, 1))
        self.v = wrap(nn.Conv1d(c_half, channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N)
        q = self.f(x).transpose(1, 2)            # (B, N, C')
        k = self.g(x)                            # (B, C', N)
        beta = torch.softmax(torch.bmm(q, k), dim=-1)  # (B, N, N)
        o = torch.bmm(self.h(x), beta.transpose(1, 2))  # (B, C/2, N)
        return self.gamma * self.v(o) + x


class ConditionalGenerator(nn.Module):
    """
    c-SAGAN generator: feature map x + noise z + label embedding Y_r ->
    1x1 conv -> self-attention -> 1x1 conv -> enhanced feature map.
    (Διαστάσεις z/label embedding δεν ορίζονται στο paper.)
    """

    def __init__(self, dim: int, n_classes: int, z_dim: int = 16, label_dim: int = 8):
        super().__init__()
        self.z_dim = z_dim
        self.label_emb = nn.Embedding(n_classes, label_dim)
        self.inp = nn.Conv1d(dim + z_dim + label_dim, dim, 1)
        self.attn = SelfAttention1d(dim)
        self.out = nn.Sequential(nn.ReLU(), nn.Conv1d(dim, dim, 1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (B, D, L)
        b, _, length = x.shape
        z = torch.randn(b, self.z_dim, 1, device=x.device, dtype=x.dtype).expand(-1, -1, length)
        e = self.label_emb(y).unsqueeze(-1).expand(-1, -1, length)
        h = F.relu(self.inp(torch.cat([x, z, e], dim=1)))
        return self.out(self.attn(h))


class ConditionalDiscriminator(nn.Module):
    """Projection discriminator D(x, y) με spectral norm + self-attention."""

    def __init__(self, dim: int, n_classes: int):
        super().__init__()
        self.c1 = spectral_norm(nn.Conv1d(dim, dim, 3, padding=1))
        self.attn = SelfAttention1d(dim, use_sn=True)
        self.c2 = spectral_norm(nn.Conv1d(dim, dim, 3, stride=2, padding=1))
        self.fc = spectral_norm(nn.Linear(dim, 1))
        self.emb = spectral_norm(nn.Embedding(n_classes, dim))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.c1(x), 0.1)
        h = self.attn(h)
        h = F.leaky_relu(self.c2(h), 0.1).sum(dim=-1)  # (B, D)
        return self.fc(h).squeeze(-1) + (self.emb(y) * h).sum(dim=1)


class CTELCAN(nn.Module):
    """
    Cross-modal Transformer + LCAN (feature learning module, modality
    classifier, subject classifier, 2-layer emotion classifier, GRL).
    Το c-SAGAN (generator/discriminator) ζει δίπλα στο μοντέλο και δεν
    χρησιμοποιείται στο test phase (Fig. 2).
    """

    def __init__(
        self,
        modality_channels: Sequence[int],
        seq_len: int,
        n_classes: int,
        n_train_subjects: int,
        dim: int = 128,
        layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        conv_channels: int = 64,
        classifier_hidden: int = 128,
    ):
        super().__init__()
        self.n_modalities = len(modality_channels)
        self.dim = dim
        self.transformer = CrossModalTransformer(
            modality_channels, seq_len, dim, layers, heads, dropout=dropout
        )

        # LCAN feature learning module (εξ. 12): Conv-BN-ReLU-Pool x2 + Flatten
        # (κοινό για όλες τις modalities).
        self.feature_module = nn.Sequential(
            nn.Conv1d(dim, conv_channels, 3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(conv_channels, conv_channels, 3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
        )
        z_dim = conv_channels * (seq_len // 4)
        self.z_dim = z_dim

        self.grl = GradientReversalLayer(1.0)
        self.modality_classifier = nn.Sequential(
            nn.Linear(z_dim, classifier_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(classifier_hidden, self.n_modalities),
        )
        self.subject_classifier = nn.Sequential(
            nn.Linear(z_dim * self.n_modalities, classifier_hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, n_train_subjects),
        )
        self.emotion_classifier = nn.Sequential(
            nn.Linear(z_dim * self.n_modalities, classifier_hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, n_classes),
        )

    def set_grl_lambda(self, value: float) -> None:
        self.grl.lambda_ = float(value)

    def encode(self, signals: List[torch.Tensor]) -> List[torch.Tensor]:
        """Y_mi ως (B, D, L) feature maps (είσοδος c-SAGAN / LCAN)."""
        return [y.transpose(1, 2) for y in self.transformer(signals)]

    def lcan(self, feature_maps: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """feature_maps[i]: (B, D, L). Επιστρέφει logits για τους 3 classifiers."""
        z = [self.feature_module(f) for f in feature_maps]   # Z_Mi
        z_f = torch.cat(z, dim=1)                              # Z_F
        modality_logits = torch.cat(
            [self.modality_classifier(self.grl(zi)) for zi in z], dim=0
        )
        return {
            "emotion": self.emotion_classifier(z_f),
            "subject": self.subject_classifier(self.grl(z_f)),
            "modality": modality_logits,   # (m*B, m), σειρά modality-major
        }

    def forward(self, signals: List[torch.Tensor]) -> torch.Tensor:
        """Test phase: CT -> LCAN feature module -> emotion classifier."""
        return self.lcan(self.encode(signals))["emotion"]


def grl_schedule(progress: float, max_lambda: float = 1.0) -> float:
    """DANN schedule 2/(1+exp(-10p)) - 1 (δεν ορίζεται στο paper)."""
    return max_lambda * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)
