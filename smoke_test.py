import warnings
warnings.filterwarnings("ignore")
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

from src.config.config import Config
from src.pipeline.data_pipeline import DataPipeline
from src.testing.pipeline_validator import PipelineValidator
from src.testing.data_statistics import DataStatistics


def main():

    config = Config()
    config.NUM_WORKERS = 0  # avoid Windows multiprocessing spawn overhead in this quick check

    pipeline = DataPipeline(config)

    original = pipeline.loader.get_subject_files
    pipeline.loader.get_subject_files = lambda: original()[:8]

    # For a fast correctness smoke-test only: trim to first 5 trials/subject
    _original_load_subject = pipeline.loader.load_subject

    def _trimmed_load_subject(subject_file):
        subject = _original_load_subject(subject_file)
        subject["data"] = subject["data"][:5]
        subject["labels"] = subject["labels"][:5]
        return subject

    pipeline.loader.load_subject = _trimmed_load_subject

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
    print("SAMPLE CHECK")
    s = train_dataset[0]
    for k, v in s.items():
        print(k, getattr(v, "shape", v))

    PipelineValidator.validate(
        train_dataset, validation_dataset, test_dataset,
        train_loader, validation_loader, test_loader, train_labels
    )
    DataStatistics.print_all(
        train_dataset, validation_dataset, test_dataset,
        train_loader, validation_loader, test_loader, train_labels
    )
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
