"""
Visualization of EEG / EDA / PPG signals.
"""

import matplotlib.pyplot as plt


class SignalVisualizer:

    @staticmethod
    def visualize_sample(dataset, index=0):

        sample = dataset[index]

        eeg = sample["eeg"]

        eda = sample["eda"]

        ppg = sample["ppg"]

        plt.figure(figsize=(15,8))

        plt.subplot(311)
        plt.plot(eeg[0])
        plt.title("EEG (Channel 1)")

        plt.subplot(312)
        plt.plot(eda[0])
        plt.title("EDA")

        plt.subplot(313)
        plt.plot(ppg[0])
        plt.title("PPG")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_dataset(dataset):

        print()

        print("Visualizing Signals...")

        SignalVisualizer.visualize_sample(dataset)