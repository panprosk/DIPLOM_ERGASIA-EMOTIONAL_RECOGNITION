"""
weighted_sampler.py

Creates a WeightedRandomSampler for the training dataset.

The sampler is used only on the training split in order to reduce
possible class imbalance during optimization.

Validation and Test loaders always use the natural data distribution.
"""

from collections import Counter

import numpy as np

import torch

from torch.utils.data import WeightedRandomSampler


class WeightedSamplerBuilder:
    """
    Builds a WeightedRandomSampler based on class frequencies.
    """

    def __init__(self):

        pass

    def build(self, labels):

        """
        Parameters
        ----------
        labels : array-like

        Returns
        -------
        WeightedRandomSampler
        """

        labels = np.asarray(labels)

        class_counts = Counter(labels)

        class_weights = {

            cls: 1.0 / count

            for cls, count in class_counts.items()

        }

        sample_weights = [

            class_weights[label]

            for label in labels

        ]

        sample_weights = torch.DoubleTensor(sample_weights)

        sampler = WeightedRandomSampler(

            weights=sample_weights,

            num_samples=len(sample_weights),

            replacement=True,

        )

        return sampler