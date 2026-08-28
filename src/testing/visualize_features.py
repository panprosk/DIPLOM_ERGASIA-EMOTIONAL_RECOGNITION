"""
visualize_features.py

Visualization utilities for handcrafted features.

Used for debugging and validating the extracted feature vectors
before training the neural networks.
"""

import numpy as np
import matplotlib.pyplot as plt


class FeatureVisualizer:

    """
    Visualize handcrafted features.
    """

    @staticmethod
    def plot_feature_vector(features):

        plt.figure(figsize=(14,4))

        plt.plot(features)

        plt.title("Handcrafted Feature Vector")

        plt.xlabel("Feature Index")

        plt.ylabel("Value")

        plt.grid(True)

        plt.tight_layout()

        plt.show()

    @staticmethod
    def plot_histogram(features):

        plt.figure(figsize=(8,5))

        plt.hist(
            features,
            bins=40
        )

        plt.title("Feature Distribution")

        plt.xlabel("Feature Value")

        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.show()

    @staticmethod
    def print_statistics(features):

        features = np.asarray(features)

        print()

        print("=" * 60)
        print("FEATURE STATISTICS")
        print("=" * 60)

        print(f"Dimension : {features.shape[0]}")
        print(f"Mean      : {features.mean():.5f}")
        print(f"Std       : {features.std():.5f}")
        print(f"Minimum   : {features.min():.5f}")
        print(f"Maximum   : {features.max():.5f}")

        print("=" * 60)

    @staticmethod
    def visualize_dataset(dataset, sample_index=0):

        sample = dataset[sample_index]

        features = sample["features"]

        print()

        print("Visualizing handcrafted features...")

        FeatureVisualizer.print_statistics(features)

        FeatureVisualizer.plot_feature_vector(features)

        FeatureVisualizer.plot_histogram(features)