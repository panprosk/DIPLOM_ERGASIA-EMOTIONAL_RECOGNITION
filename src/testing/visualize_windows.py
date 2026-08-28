"""
Visualization of segmented windows.
"""

import matplotlib.pyplot as plt

import numpy as np


class WindowVisualizer:

    @staticmethod
    def visualize_window(signal, title):

        plt.figure(figsize=(12,3))

        plt.plot(signal)

        plt.title(title)

        plt.grid(True)

        plt.tight_layout()

        plt.show()

    @staticmethod
    def visualize_dataset(dataset):

        sample = dataset[np.random.randint(len(dataset))]

        eeg = sample["eeg"]

        eda = sample["eda"]

        ppg = sample["ppg"]

        print()

        print("Visualizing Random Window")

        WindowVisualizer.visualize_window(

            eeg[0],

            "EEG Window"

        )

        WindowVisualizer.visualize_window(

            eda[0],

            "EDA Window"

        )

        WindowVisualizer.visualize_window(

            ppg[0],

            "PPG Window"

        )