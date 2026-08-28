"""
main_data_pipeline.py

Main entry point of the DEAP Data Pipeline.

Pipeline

Load Dataset
      ↓
Subject Split
      ↓
Preprocessing
      ↓
Windowing
      ↓
Feature Extraction
      ↓
Dataset
      ↓
DataLoaders
      ↓
Validation
      ↓
Statistics
      ↓
Visualization
"""

from src.config.config import Config

from src.pipeline.data_pipeline import DataPipeline

from src.testing.pipeline_validator import PipelineValidator

from src.testing.data_statistics import DataStatistics

from src.testing.visualize_signals import SignalVisualizer

from src.testing.visualize_windows import WindowVisualizer


def main():

    print("=" * 80)
    print("DEAP DATA PIPELINE")
    print("=" * 80)

    config = Config()

    pipeline = DataPipeline(config)

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        train_loader,
        validation_loader,
        test_loader,
        train_labels,
        train_subjects,
    ) = pipeline.run()

    PipelineValidator.validate(
        train_dataset=train_dataset,
        val_dataset=validation_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=validation_loader,
        test_loader=test_loader,
        train_labels=train_labels,
    )

    DataStatistics.print_all(
        train_dataset=train_dataset,
        val_dataset=validation_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=validation_loader,
        test_loader=test_loader,
        train_labels=train_labels,
    )

    SignalVisualizer.visualize_dataset(
        train_dataset
    )

    WindowVisualizer.visualize_dataset(
        train_dataset
    )

    print()

    print("=" * 80)
    print("DATA PIPELINE FINISHED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":

    main()