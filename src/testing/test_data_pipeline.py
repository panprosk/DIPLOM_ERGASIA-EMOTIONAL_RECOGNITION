"""
test_data_pipeline.py

Complete testing script for the DEAP Data Pipeline.

Checks:
✔ Loader
✔ Subject Split
✔ Preprocessing
✔ Windowing
✔ Feature Extraction
✔ Dataset
✔ DataLoader
✔ Validation
✔ Statistics
✔ Visualizations
"""

import warnings
warnings.filterwarnings("ignore")

import logging
import sys

# Configure logging to see output in the console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

from src.config.config import Config
from src.pipeline.data_pipeline import DataPipeline
from src.testing.pipeline_validator import PipelineValidator
from src.testing.data_statistics import DataStatistics
from src.testing.visualize_signals import SignalVisualizer
from src.testing.visualize_windows import WindowVisualizer
from src.testing.visualize_features import FeatureVisualizer

def test_data_pipeline_execution():
    main()

def main():

    print()
    print("=" * 80)
    print("TESTING COMPLETE DATA PIPELINE")
    print("=" * 80)

    config = Config()

    # [TEMPORARY FOR TESTING] Limit to 10 subjects to speed up verification
    logger.info("Initializing pipeline (limiting to 10 subjects for speed)...")

    pipeline = DataPipeline(config)
    
    if hasattr(pipeline.loader, 'get_subject_files'):
        original_get_files = pipeline.loader.get_subject_files
        pipeline.loader.get_subject_files = lambda: original_get_files()[:10]
        logger.info("Limited loader to first 10 subjects.")

    # Unpacking των 8 τιμών με την ακριβή σειρά που επιστρέφονται από το DataPipeline.run()
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

    print()
    print("=" * 80)
    print("PIPELINE VALIDATION")
    print("=" * 80)

    PipelineValidator.validate(
        train_dataset,
        validation_dataset,
        test_dataset,
        train_loader,
        validation_loader,
        test_loader,
        train_labels,
    )

    print()
    print("=" * 80)
    print("PIPELINE STATISTICS")
    print("=" * 80)

    DataStatistics.print_all(
        train_dataset,
        validation_dataset,
        test_dataset,
        train_loader,
        validation_loader,
        test_loader,
        train_labels,
    )

    print()
    print("=" * 80)
    print("SIGNAL VISUALIZATION")
    print("=" * 80)

    SignalVisualizer.visualize_dataset(
        train_dataset
    )

    print()
    print("=" * 80)
    print("WINDOW VISUALIZATION")
    print("=" * 80)

    WindowVisualizer.visualize_dataset(
        train_dataset
    )

    print()
    print("=" * 80)
    print("FEATURE VISUALIZATION")
    print("=" * 80)

    FeatureVisualizer.visualize_dataset(
        train_dataset
    )

    print()
    print("=" * 80)
    print("PIPELINE TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":

    main()