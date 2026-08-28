"""
dataloader_factory.py

Creates all PyTorch DataLoaders.

Train
Validation
Test

Supports:

- Weighted Sampling
- Shuffle
- Multi-worker loading
- Pin memory
"""

from torch.utils.data import DataLoader

from src.dataloaders.weighted_sampler import WeightedSamplerBuilder


class DataLoaderFactory:

    """
    Factory responsible for creating
    train / validation / test dataloaders.
    """

    def __init__(

        self,

        batch_size,

        num_workers=4,

        pin_memory=True,

        weighted_sampling=True,

    ):

        self.batch_size = batch_size

        self.num_workers = num_workers

        self.pin_memory = pin_memory

        self.weighted_sampling = weighted_sampling

    def build_train_loader(

        self,

        dataset,

        labels,

    ):

        """
        Creates the Train DataLoader.
        """

        if self.weighted_sampling:

            sampler = WeightedSamplerBuilder().build(labels)

            loader = DataLoader(

                dataset,

                batch_size=self.batch_size,

                sampler=sampler,

                num_workers=self.num_workers,

                pin_memory=self.pin_memory,

                drop_last=False,

            )

        else:

            loader = DataLoader(

                dataset,

                batch_size=self.batch_size,

                shuffle=True,

                num_workers=self.num_workers,

                pin_memory=self.pin_memory,

                drop_last=False,

            )

        return loader

    def build_validation_loader(

        self,

        dataset,

    ):

        """
        Creates Validation DataLoader.
        """

        loader = DataLoader(

            dataset,

            batch_size=self.batch_size,

            shuffle=False,

            num_workers=self.num_workers,

            pin_memory=self.pin_memory,

            drop_last=False,

        )

        return loader

    def build_test_loader(

        self,

        dataset,

    ):

        """
        Creates Test DataLoader.
        """

        loader = DataLoader(

            dataset,

            batch_size=self.batch_size,

            shuffle=False,

            num_workers=self.num_workers,

            pin_memory=self.pin_memory,

            drop_last=False,

        )

        return loader

    def build(

        self,

        train_dataset,

        val_dataset,

        test_dataset,

        train_labels,

    ):

        """
        Returns all dataloaders.
        """

        train_loader = self.build_train_loader(

            train_dataset,

            train_labels,

        )

        validation_loader = self.build_validation_loader(

            val_dataset,

        )

        test_loader = self.build_test_loader(

            test_dataset,

        )

        return {

            "train": train_loader,

            "validation": validation_loader,

            "test": test_loader,

        }