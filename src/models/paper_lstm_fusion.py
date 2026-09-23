"""
paper_lstm_fusion.py

Reproduction / physiological adaptation του μοντέλου που περιγράφεται
στο:

    Saffaryazdi et al. (2022), "Using Facial Micro-Expressions in
    Combination With EEG and Physiological Signals for Emotion
    Recognition", Frontiers in Psychology, 13:864047.

Architecture (βλ. σημειώσεις χρήστη #8-13)
-------------------------------------------
Για κάθε modality (EEG, EDA, PPG) ένα ξεχωριστό, 2-layer stacked LSTM
("LSTM 1" -> 80 units, "LSTM 2" -> 30 units, όπως αναφέρει το paper)
επεξεργάζεται την ακολουθία των per-window features του modality και
παράγει ξεχωριστές valence/arousal προβλέψεις (auxiliary heads, ώστε
κάθε branch να εκπαιδεύεται με ουσιαστική supervision -- ακριβώς όπως
το paper εκπαιδεύει EEG/PPG/GSR LSTMs "παράλληλα").

Η αρχιτεκτονική του branch μπορεί να χρησιμοποιεί είτε το τελευταίο
BiLSTM hidden representation είτε temporal attention πάνω σε όλα τα
hidden states, ανάλογα με το πείραμα.

Fusion: αντί για σταθερά modality weights, το Experiment C χρησιμοποιεί
sample-dependent softmax gates πάνω στις τρεις modality representations.
Έτσι το μοντέλο μπορεί να δίνει διαφορετική βαρύτητα σε EEG, EDA και PPG
ανά trial, πριν από τα τελικά valence/arousal heads.

Deviations από το paper (τεκμηριωμένες, βλ. συζήτηση με τον χρήστη)
--------------------------------------------------------------------
* Δεν χρησιμοποιείται facial-micro-expression branch/ROI-selection.
* Windowing πάνω σε ΟΛΟΚΛΗΡΟ το trial (60 x 1s windows) αντί για το
  15s ROI γύρω από το micro-expression apex.
* Evaluation: subject-independent train/val/test split (ίδιο με το
  υπόλοιπο project) αντί για 6-fold leave-some-subject-out.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityLSTMBranch(nn.Module):
    """
    2-layer stacked BiLSTM (80 -> 30 units, όπως στο paper) πάνω σε μία
    ακολουθία window-features ενός modality, με δύο ξεχωριστά binary
    heads (valence, arousal).
    """

    def __init__(
        self,
        input_dim: int,
        hidden1: int = 80,
        hidden2: int = 30,
        dropout: float = 0.30,
        num_classes: int = 2,
    ):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_dim, hidden1, batch_first=True, bidirectional=True
        )
        self.lstm2 = nn.LSTM(
            hidden1 * 2, hidden2, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)

        self.valence_head = nn.Linear(hidden2 * 2, num_classes)
        self.arousal_head = nn.Linear(hidden2 * 2, num_classes)

    def forward(self, sequence: torch.Tensor):
        """
        sequence : (batch, seq_len, input_dim)

        Returns
        -------
        valence_logits : (batch, num_classes)
        arousal_logits : (batch, num_classes)
        embedding      : (batch, hidden2 * 2) -- final bidirectional state
        """

        out, _ = self.lstm1(sequence)
        _, (h_n, _) = self.lstm2(out)
        embedding = torch.cat([h_n[-2], h_n[-1]], dim=-1)

        embedding = self.dropout(embedding)

        valence_logits = self.valence_head(embedding)
        arousal_logits = self.arousal_head(embedding)

        return valence_logits, arousal_logits, embedding


class PaperLSTMFusionModel(nn.Module):
    """
    Multimodal EEG + EDA + PPG LSTM fusion model, reproduction του
    Saffaryazdi et al. (2022).
    """

    def __init__(
        self,
        eeg_feature_dim: int = 160,
        eda_feature_dim: int = 4,
        ppg_feature_dim: int = 2,
        hidden1: int = 80,
        hidden2: int = 30,
        dropout: float = 0.30,
        num_classes: int = 2,
    ):
        super().__init__()

        self.eeg_branch = ModalityLSTMBranch(
            eeg_feature_dim, hidden1, hidden2, dropout, num_classes
        )
        self.eda_branch = ModalityLSTMBranch(
            eda_feature_dim, hidden1, hidden2, dropout, num_classes
        )
        self.ppg_branch = ModalityLSTMBranch(
            ppg_feature_dim, hidden1, hidden2, dropout, num_classes
        )

        representation_dim = hidden2 * 2
        fusion_input_dim = representation_dim * 3

        self.valence_gate = nn.Linear(fusion_input_dim, 3)
        self.arousal_gate = nn.Linear(fusion_input_dim, 3)
        self.valence_fusion_head = nn.Linear(representation_dim, num_classes)
        self.arousal_fusion_head = nn.Linear(representation_dim, num_classes)

    def forward(self, eeg: torch.Tensor, eda: torch.Tensor, ppg: torch.Tensor) -> dict:
        """
        eeg : (batch, seq_len, eeg_feature_dim)
        eda : (batch, seq_len, eda_feature_dim)
        ppg : (batch, seq_len, ppg_feature_dim)
        """

        v_eeg, a_eeg, emb_eeg = self.eeg_branch(eeg)
        v_eda, a_eda, emb_eda = self.eda_branch(eda)
        v_ppg, a_ppg, emb_ppg = self.ppg_branch(ppg)

        modality_embeddings = torch.stack(
            [emb_eeg, emb_eda, emb_ppg], dim=1
        )  # (batch, 3, representation_dim)
        gate_input = torch.cat([emb_eeg, emb_eda, emb_ppg], dim=-1)

        valence_gate = F.softmax(self.valence_gate(gate_input), dim=-1)
        arousal_gate = F.softmax(self.arousal_gate(gate_input), dim=-1)

        fused_valence_embedding = (
            modality_embeddings * valence_gate.unsqueeze(-1)
        ).sum(dim=1)
        fused_arousal_embedding = (
            modality_embeddings * arousal_gate.unsqueeze(-1)
        ).sum(dim=1)

        fused_valence_logits = self.valence_fusion_head(
            fused_valence_embedding
        )
        fused_arousal_logits = self.arousal_fusion_head(
            fused_arousal_embedding
        )

        return {
            "valence_probs": F.softmax(fused_valence_logits, dim=-1),
            "arousal_probs": F.softmax(fused_arousal_logits, dim=-1),
            "valence_branch_logits": {"eeg": v_eeg, "eda": v_eda, "ppg": v_ppg},
            "arousal_branch_logits": {"eeg": a_eeg, "eda": a_eda, "ppg": a_ppg},
            "valence_gate_weights": valence_gate,
            "arousal_gate_weights": arousal_gate,
        }
