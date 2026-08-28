import numpy as np


class LabelGenerator:
    """
    Δημιουργία labels.

    Αρχικά μόνο Binary Valence.
    """

    def __init__(self, threshold=5.0):

        self.threshold = threshold

    def binary_valence(self, labels):

        """
        labels

        (40,4)

        επιστρέφει

        (40,)
        """

        valence = labels[:, 0]

        binary = (valence > self.threshold).astype(np.int64)

        return binary

    def add_binary_labels(self, dataset):

        """
        Προσθέτει labels
        σε όλους τους subjects.
        """

        for subject in dataset.values():

            subject["binary_valence"] = self.binary_valence(
                subject["labels"]
            )

        return dataset

    def generate(self, dataset):

        """
        Alias της add_binary_labels για συμβατότητα με κλήσεις generate().
        """

        return self.add_binary_labels(dataset)