"""Stacked LSTM branches and decision fusion for the Saffaryazdi baseline."""

from __future__ import annotations

import torch
from torch import nn


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, coefficient):
        ctx.coefficient = coefficient
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.coefficient * gradient, None


def gradient_reverse(value: torch.Tensor, coefficient: float) -> torch.Tensor:
    return GradientReverse.apply(value, coefficient)


class StackedLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden1: int = 80,
        hidden2: int = 30,
        dropout: float = 0.0,
        pooling: str = "last",
        bidirectional: bool = False,
        temporal_encoder: str = "lstm",
        spatial_cnn: bool = False,
    ):
        super().__init__()
        if pooling not in {"last", "mean", "attention", "attention_residual"}:
            raise ValueError(f"Unsupported pooling mode: {pooling}")
        if temporal_encoder not in {"lstm", "cnn_lstm", "spatial_cnn_lstm"}:
            raise ValueError(f"Unsupported temporal encoder: {temporal_encoder}")
        directions = 2 if bidirectional else 1
        self.lstm1 = nn.LSTM(
            hidden1
            if temporal_encoder in {"cnn_lstm", "spatial_cnn_lstm"}
            else input_dim,
            hidden1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.temporal_conv = (
            nn.Sequential(
                nn.Conv1d(input_dim, hidden1, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden1, hidden1, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            if temporal_encoder == "cnn_lstm"
            else None
        )
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(
            hidden1 * directions,
            hidden2,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout2 = nn.Dropout(dropout)
        representation_dim = hidden2 * directions
        self.attention = (
            nn.Linear(representation_dim, 1)
            if pooling in {"attention", "attention_residual"}
            else None
        )
        self.classifier = nn.Linear(representation_dim, 2)
        self.pooling = pooling
        self.temporal_encoder = temporal_encoder
        self.spatial_cnn = spatial_cnn
        self.spatial_conv = (
            nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 8, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            if spatial_cnn
            else None
        )
        self.spatial_projection = (
            nn.Linear(8 * 32 * 5, hidden1) if spatial_cnn else None
        )

    def encode(self, sequence: torch.Tensor) -> torch.Tensor:
        output = self.encode_sequence(sequence)
        if self.pooling == "last":
            representation = output[:, -1]
        elif self.pooling == "mean":
            representation = output.mean(dim=1)
        else:
            scores = torch.softmax(self.attention(output).squeeze(-1), dim=1)
            attention_representation = torch.sum(
                output * scores.unsqueeze(-1), dim=1
            )
            if self.pooling == "attention_residual":
                representation = attention_representation + output.mean(dim=1)
            else:
                representation = attention_representation
        return representation

    def encode_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        if self.spatial_cnn:
            batch, steps, features = sequence.shape
            if features != 160:
                raise ValueError(
                    f"Spatial EEG encoder expects 160 features, got {features}."
                )
            sequence = sequence.view(batch, steps, 5, 32).transpose(2, 3)
            sequence = sequence.reshape(batch * steps, 1, 32, 5)
            sequence = self.spatial_conv(sequence).flatten(1)
            sequence = self.spatial_projection(sequence)
            sequence = sequence.view(batch, steps, -1)
        elif self.temporal_encoder == "cnn_lstm":
            sequence = self.temporal_conv(sequence.transpose(1, 2)).transpose(1, 2)
        output, _ = self.lstm1(sequence)
        output = self.dropout1(output)
        output, _ = self.lstm2(output)
        output = self.dropout2(output)
        return output

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        representation = self.encode(sequence)
        return self.classifier(representation)


class SaffaryazdiTaskModel(nn.Module):
    """Independent EEG and physiology branches with late probability fusion."""

    def __init__(
        self,
        input_dim: int,
        hidden1: int = 80,
        hidden2: int = 30,
        phys_dim: int = 6,
        dropout: float = 0.0,
        pooling: str = "last",
        fusion_mode: str = "late",
        bidirectional: bool = False,
        temporal_encoder: str = "lstm",
        subject_classes: int = 0,
    ):
        super().__init__()
        if fusion_mode not in {"late", "learned", "gated", "cross_modal"}:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")
        self.eeg = StackedLSTMClassifier(
            input_dim,
            hidden1,
            hidden2,
            dropout,
            pooling,
            bidirectional,
            temporal_encoder,
            spatial_cnn=temporal_encoder == "spatial_cnn_lstm",
        )
        self.phys = StackedLSTMClassifier(
            phys_dim,
            hidden1,
            hidden2,
            dropout,
            pooling,
            bidirectional,
            "lstm" if temporal_encoder == "spatial_cnn_lstm" else temporal_encoder,
            spatial_cnn=False,
        )
        self.fusion_mode = fusion_mode
        self.fusion_classifier = (
            nn.Linear(hidden2 * (4 if bidirectional else 2), 2)
            if fusion_mode == "learned"
            else None
        )
        representation_dim = hidden2 * (2 if bidirectional else 1)
        self.eeg_projection = (
            nn.Linear(representation_dim, representation_dim)
            if fusion_mode == "gated"
            else None
        )
        self.phys_projection = (
            nn.Linear(representation_dim, representation_dim)
            if fusion_mode == "gated"
            else None
        )
        self.fusion_gate = (
            nn.Linear(representation_dim * 2, representation_dim)
            if fusion_mode == "gated"
            else None
        )
        self.gated_classifier = (
            nn.Linear(representation_dim, 2) if fusion_mode == "gated" else None
        )
        self.cross_attention = (
            nn.MultiheadAttention(
                representation_dim, num_heads=2, batch_first=True
            )
            if fusion_mode == "cross_modal"
            else None
        )
        self.cross_classifier = (
            nn.Linear(representation_dim, 2)
            if fusion_mode == "cross_modal"
            else None
        )
        representation_dim = hidden2 * (2 if bidirectional else 1)
        self.subject_classifier = (
            nn.Sequential(
                nn.Linear(representation_dim * 2, representation_dim),
                nn.ReLU(),
                nn.Linear(representation_dim, subject_classes),
            )
            if subject_classes > 0
            else None
        )

    def forward(self, eeg: torch.Tensor, phys: torch.Tensor):
        eeg_logits = self.eeg(eeg)
        phys_logits = self.phys(phys)
        return eeg_logits, phys_logits

    def forward_learned(self, eeg: torch.Tensor, phys: torch.Tensor):
        eeg_representation = self.eeg.encode(eeg)
        phys_representation = self.phys.encode(phys)
        eeg_logits = self.eeg.classifier(eeg_representation)
        phys_logits = self.phys.classifier(phys_representation)
        fused_logits = self.fusion_classifier(
            torch.cat([eeg_representation, phys_representation], dim=1)
        )
        return eeg_logits, phys_logits, fused_logits

    def forward_gated(self, eeg: torch.Tensor, phys: torch.Tensor):
        eeg_representation = self.eeg.encode(eeg)
        phys_representation = self.phys.encode(phys)
        eeg_logits = self.eeg.classifier(eeg_representation)
        phys_logits = self.phys.classifier(phys_representation)
        eeg_projected = self.eeg_projection(eeg_representation)
        phys_projected = self.phys_projection(phys_representation)
        gate = torch.sigmoid(
            self.fusion_gate(torch.cat([eeg_projected, phys_projected], dim=1))
        )
        fused_representation = gate * eeg_projected + (1.0 - gate) * phys_projected
        fused_logits = self.gated_classifier(fused_representation)
        return eeg_logits, phys_logits, fused_logits

    def forward_cross_modal(self, eeg: torch.Tensor, phys: torch.Tensor):
        eeg_states = self.eeg.encode_sequence(eeg)
        phys_states = self.phys.encode_sequence(phys)
        eeg_representation = self.eeg.encode(eeg)
        phys_representation = self.phys.encode(phys)
        eeg_logits = self.eeg.classifier(eeg_representation)
        phys_logits = self.phys.classifier(phys_representation)
        attended_eeg, _ = self.cross_attention(
            query=eeg_states, key=phys_states, value=phys_states
        )
        fused_representation = (attended_eeg + eeg_states).mean(dim=1)
        fused_logits = self.cross_classifier(fused_representation)
        return eeg_logits, phys_logits, fused_logits

    def forward_adversarial(self, eeg, phys, coefficient):
        eeg_representation = self.eeg.encode(eeg)
        phys_representation = self.phys.encode(phys)
        eeg_logits = self.eeg.classifier(eeg_representation)
        phys_logits = self.phys.classifier(phys_representation)
        subject_input = torch.cat([eeg_representation, phys_representation], dim=1)
        subject_logits = self.subject_classifier(
            gradient_reverse(subject_input, coefficient)
        )
        return eeg_logits, phys_logits, subject_logits

    @staticmethod
    def fuse_probabilities(
        eeg_logits: torch.Tensor,
        phys_logits: torch.Tensor,
        eeg_weight: float,
    ) -> torch.Tensor:
        eeg_probs = torch.softmax(eeg_logits, dim=-1)
        phys_probs = torch.softmax(phys_logits, dim=-1)
        return eeg_weight * eeg_probs + (1.0 - eeg_weight) * phys_probs
