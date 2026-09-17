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

Fusion: αντί να εκπαιδεύσουμε 3 (ή 6, valence+arousal) εντελώς χωριστά
μοντέλα και μετά να τα συνδυάσουμε offline (όπως κάνει το paper), εδώ
υλοποιείται ένα ενιαίο, end-to-end εκπαιδεύσιμο "weighted probability
fusion" (Section 14.Β του paper) μέσω ενός μικρού softmax-normalized
learnable βάρους ανά modality/task -- πρακτικά ισοδύναμο αλλά πιο
αποδοτικό (ένα training run αντί για 6).

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
    2-layer stacked LSTM (80 -> 30 units, όπως στο paper) πάνω σε μία
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

        self.lstm1 = nn.LSTM(input_dim, hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)

        self.dropout = nn.Dropout(dropout)

        self.valence_head = nn.Linear(hidden2, num_classes)
        self.arousal_head = nn.Linear(hidden2, num_classes)

    def forward(self, sequence: torch.Tensor):
        """
        sequence : (batch, seq_len, input_dim)

        Returns
        -------
        valence_logits : (batch, num_classes)
        arousal_logits : (batch, num_classes)
        embedding      : (batch, hidden2)  -- τελικό hidden state
        """

        out, _ = self.lstm1(sequence)
        out, (h_n, _) = self.lstm2(out)

        # Τελικό hidden state του 2ου LSTM layer (ισοδύναμο με
        # "Dense layer πάνω στην τελευταία χρονική στιγμή" του paper).
        embedding = h_n[-1]

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

        # Weighted probability fusion (Section 14.Β): ένα learnable
        # βάρος ανά modality, ξεχωριστά για valence και arousal,
        # κανονικοποιημένο με softmax ώστε να αθροίζει σε 1.
        self.valence_fusion_weights = nn.Parameter(torch.ones(3))
        self.arousal_fusion_weights = nn.Parameter(torch.ones(3))

    def forward(self, eeg: torch.Tensor, eda: torch.Tensor, ppg: torch.Tensor) -> dict:
        """
        eeg : (batch, seq_len, eeg_feature_dim)
        eda : (batch, seq_len, eda_feature_dim)
        ppg : (batch, seq_len, ppg_feature_dim)
        """

        v_eeg, a_eeg, emb_eeg = self.eeg_branch(eeg)
        v_eda, a_eda, emb_eda = self.eda_branch(eda)
        v_ppg, a_ppg, emb_ppg = self.ppg_branch(ppg)

        valence_probs_per_branch = torch.stack(
            [F.softmax(v_eeg, dim=-1), F.softmax(v_eda, dim=-1), F.softmax(v_ppg, dim=-1)],
            dim=1,
        )  # (batch, 3, num_classes)

        arousal_probs_per_branch = torch.stack(
            [F.softmax(a_eeg, dim=-1), F.softmax(a_eda, dim=-1), F.softmax(a_ppg, dim=-1)],
            dim=1,
        )

        valence_weights = F.softmax(self.valence_fusion_weights, dim=0)  # (3,)
        arousal_weights = F.softmax(self.arousal_fusion_weights, dim=0)

        fused_valence_probs = (
            valence_probs_per_branch * valence_weights.view(1, 3, 1)
        ).sum(dim=1)  # (batch, num_classes)

        fused_arousal_probs = (
            arousal_probs_per_branch * arousal_weights.view(1, 3, 1)
        ).sum(dim=1)

        return {
            "valence_probs": fused_valence_probs,
            "arousal_probs": fused_arousal_probs,
            "valence_branch_logits": {"eeg": v_eeg, "eda": v_eda, "ppg": v_ppg},
            "arousal_branch_logits": {"eeg": a_eeg, "eda": a_eda, "ppg": a_ppg},
            "valence_fusion_weights": valence_weights,
            "arousal_fusion_weights": arousal_weights,
        }
