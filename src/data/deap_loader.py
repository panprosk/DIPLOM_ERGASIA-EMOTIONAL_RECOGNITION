from pathlib import Path
import pickle
import numpy as np


class DEAPLoader:
    """
    Loader για το DEAP dataset.

    Κάθε αρχείο subject (s01.dat ... s32.dat)
    περιέχει:
        data   : (40, 40, 8064)
        labels : (40, 4)
    """

    def __init__(self, dataset_path: str):

        self.dataset_path = Path(dataset_path)

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset path not found: {self.dataset_path}"
            )

    def get_subject_files(self):
        """
        Επιστρέφει όλα τα αρχεία sXX.dat
        """

        return sorted(self.dataset_path.glob("s*.dat"))

    def load_subject(self, subject_file):
        """
        Φόρτωση ενός subject.
        """

        with open(subject_file, "rb") as f:
            subject = pickle.load(f, encoding="latin1")

        return subject

    def load_all_subjects(self):
        """
        Φορτώνει όλους τους subjects.
        """

        dataset = {}

        subject_files = self.get_subject_files()

        for subject_file in subject_files:

            subject_name = subject_file.stem

            subject = self.load_subject(subject_file)

            dataset[subject_name] = {

                "data": np.asarray(subject["data"], dtype=np.float32),

                "labels": np.asarray(subject["labels"], dtype=np.float32)

            }

        return dataset