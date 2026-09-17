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

    def binary_arousal(self, labels):

        """
        labels

        (40,4), στήλη 1 = arousal (DEAP: valence, arousal, dominance,
        liking).

        Θρεσχόλδ ίδιο με το valence (5.0 στην κλίμακα SAM 1-9), όπως
        στο Saffaryazdi et al. (2022): arousal < 5 -> LOW, >= 5 -> HIGH.
        Εδώ χρησιμοποιούμε ">" (ίδια σύμβαση με το binary_valence) ώστε
        να μείνει συνεπές με το υπόλοιπο pipeline.

        επιστρέφει

        (40,)
        """

        arousal = labels[:, 1]

        binary = (arousal > self.threshold).astype(np.int64)

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

    def add_binary_arousal_labels(self, dataset):

        """
        Προσθέτει binary arousal labels σε όλους τους subjects.
        """

        for subject in dataset.values():

            subject["binary_arousal"] = self.binary_arousal(
                subject["labels"]
            )

        return dataset

    def generate(self, dataset):

        """
        Alias της add_binary_labels για συμβατότητα με κλήσεις generate().
        """

        return self.add_binary_labels(dataset)