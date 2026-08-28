"""
data_statistics.py

Computes useful statistics for the DEAP pipeline.

Useful before training.
"""

import numpy as np


class DataStatistics:

    def __init__(self):

        pass

    @staticmethod
    def label_distribution(labels):

        labels = np.asarray(labels)

        unique, counts = np.unique(

            labels,

            return_counts=True

        )

        print()

        print("Label Distribution")

        for u, c in zip(unique, counts):

            print(

                f"Class {u}: {c}"

            )

        print()

    @staticmethod
    def dataset_statistics(dataset):

        print()

        print("Dataset Statistics")

        print("------------------------")

        print(

            "Samples:",

            len(dataset)

        )

        sample = dataset[0]

        print(

            "EEG:",

            sample["eeg"].shape

        )

        print(

            "EDA:",

            sample["eda"].shape

        )

        print(

            "PPG:",

            sample["ppg"].shape

        )

        print(

            "Features:",

            sample["features"].shape

        )

        print()

    @staticmethod
    def dataloader_statistics(loader):

        batches = len(loader)

        samples = len(loader.dataset)

        print()

        print("DataLoader Statistics")

        print("----------------------")

        print(

            "Samples:",

            samples

        )

        print(

            "Batches:",

            batches

        )

        print()

    @staticmethod
    def print_all(

        train_dataset,

        val_dataset,

        test_dataset,

        train_loader,

        val_loader,

        test_loader,

        train_labels,

    ):

        print()

        print("=" * 70)

        print("DATASET SUMMARY")

        print("=" * 70)

        DataStatistics.dataset_statistics(

            train_dataset

        )

        DataStatistics.dataset_statistics(

            val_dataset

        )

        DataStatistics.dataset_statistics(

            test_dataset

        )

        DataStatistics.dataloader_statistics(

            train_loader

        )

        DataStatistics.dataloader_statistics(

            val_loader

        )

        DataStatistics.dataloader_statistics(

            test_loader

        )

        DataStatistics.label_distribution(

            train_labels

        )