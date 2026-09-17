"""
paper_lstm_dataset.py

PyTorch Dataset για τα per-trial feature sequences που παράγει το
PaperLSTMDataPipeline (βλ. src/pipeline/paper_lstm_pipeline.py).

Κάθε δείγμα αντιστοιχεί σε ΕΝΑ ολόκληρο trial (60 x 1-δευτερόλεπτο
windows), όχι σε ένα μεμονωμένο window -- το LSTM βλέπει ολόκληρη την
ακολουθία πριν κάνει την τελική πρόβλεψη valence/arousal.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class PaperLSTMDataset(Dataset):

    def __init__(self, arrays: dict):

        self.eeg = arrays["eeg"]
        self.eda = arrays["eda"]
        self.ppg = arrays["ppg"]
        self.valence = arrays["valence"]
        self.arousal = arrays["arousal"]
        self.subjects = arrays["subjects"]
        self.trials = arrays["trials"]

    def __len__(self):
        return len(self.valence)

    def __getitem__(self, idx):

        return {
            "eeg": torch.tensor(self.eeg[idx], dtype=torch.float32),
            "eda": torch.tensor(self.eda[idx], dtype=torch.float32),
            "ppg": torch.tensor(self.ppg[idx], dtype=torch.float32),
            "valence": torch.tensor(self.valence[idx], dtype=torch.long),
            "arousal": torch.tensor(self.arousal[idx], dtype=torch.long),
            "subject": str(self.subjects[idx]),
            "trial": int(self.trials[idx]),
        }
