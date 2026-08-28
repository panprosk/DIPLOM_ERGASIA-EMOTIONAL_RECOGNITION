import torch

from src.datasets.base_dataset import BaseDataset


class MultimodalDataset(BaseDataset):

    """
    Dataset για όλα τα υπόλοιπα μοντέλα (π.χ. Cross-Attention
    Transformer, Mamba, GNN, AGAT-AGS).

    Επεκτείνει το BaseDataset (ίδιο pattern με το HybridDataset) ώστε
    να αποθηκεύει subjects/trials/windows metadata -- απαραίτητο για
    domain-adversarial training (χρειάζεται το subject id ανά δείγμα)
    και για οποιαδήποτε per-subject ανάλυση αποτελεσμάτων. Τα raw
    eeg/eda/ppg arrays αποθηκεύονται σε float16 (βλ. data_pipeline.py)
    και μετατρέπονται σε float32 tensor ανά-δείγμα μέσω του
    BaseDataset.to_tensor(), ίδια λογική μνήμης με το HybridDataset.
    """

    def __init__(
        self,
        eeg_windows,
        eda_windows,
        ppg_windows,
        labels,
        subjects=None,
        trials=None,
        windows=None,
    ):

        super().__init__(
            labels=labels,
            subjects=subjects,
            trials=trials,
            windows=windows,
        )

        self.eeg = eeg_windows

        self.eda = eda_windows

        self.ppg = ppg_windows

    def __getitem__(self, idx):

        sample = {

            "eeg": self.to_tensor(self.eeg[idx]),

            "eda": self.to_tensor(self.eda[idx]),

            "ppg": self.to_tensor(self.ppg[idx]),

            "label": torch.tensor(
                self.labels[idx], dtype=torch.long,
            ),

        }

        sample.update(self.get_metadata(idx))

        return sample