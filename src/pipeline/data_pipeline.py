from __future__ import annotations
import gc
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np

from src.config.config import Config
from src.config.signals import EDA_CHANNEL, PPG_CHANNEL
from src.data.deap_loader import DEAPLoader
from src.data.subject_manager import SubjectManager
from src.data.label_generator import LabelGenerator
from src.split.subject_split import SubjectSplitter
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from src.preprocessing.baseline_correction import remove_trial_baseline
from src.segmentation.window_segmenter import WindowSegmenter
from src.features.feature_extractor import FeatureExtractor
from src.features.feature_scaler import FeatureScaler
from src.datasets.dataset_factory import build_dataset
from src.dataloaders.dataloader_factory import DataLoaderFactory

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Complete DEAP Pipeline.

    This class is responsible for building
    the complete data pipeline once.

    Every model and every experiment
    will reuse this pipeline.

    Returns:
        train_loader
        validation_loader
        test_loader
    """

    def __init__(self, config: Config):

        self.config = config

        logger.info("Initializing Data Pipeline...")

        ####################################################
        # Loader
        ####################################################

        self.loader = DEAPLoader(
            dataset_path=config.DEAP_PATH
        )

        ####################################################
        # Subject Manager
        ####################################################

        self.subject_manager = SubjectManager()

        ####################################################
        # Labels
        ####################################################

        self.label_generator = LabelGenerator(
            threshold=config.VALENCE_THRESHOLD
        )

        ####################################################
        # Split
        ####################################################

        self.splitter = SubjectSplitter(
            random_seed=config.SEED
        )

        ####################################################
        # Preprocessing
        ####################################################

        self.preprocessing = PreprocessingPipeline(
            sampling_rate=config.SAMPLING_RATE
        )

        ####################################################
        # Windowing
        ####################################################

        self.segmenter = WindowSegmenter(
            sampling_rate=config.SAMPLING_RATE,
            window_seconds=config.WINDOW_SIZE,
            overlap=config.OVERLAP
        )

        ####################################################
        # Features
        ####################################################

        self.feature_extractor = FeatureExtractor(
            sampling_rate=config.SAMPLING_RATE
        )

        ####################################################
        # Feature Scaler
        ####################################################

        self.feature_scaler = FeatureScaler()

        self.physio_feature_scaler = FeatureScaler()

        logger.info("Pipeline successfully initialized.")

    @staticmethod
    def _fit_transform_feature_key(
        train_windows: List[Dict],
        validation_windows: List[Dict],
        test_windows: List[Dict],
        key: str,
        scaler,
    ) -> None:
        """
        Fits a StandardScaler on a given feature key using the train
        windows only, then transforms train / validation / test in
        bulk (vectorized) and writes the scaled values back in place.
        """

        train_matrix = np.stack(
            [sample[key] for sample in train_windows]
        )

        # Δεύτερη γραμμή άμυνας: αν κάποιο NaN/Inf ξέφυγε από την
        # εξαγωγή χαρακτηριστικών, ένα και μόνο NaN σε μία στήλη θα
        # μόλυνε το mean/std του StandardScaler και θα έκανε NaN ΟΛΗ
        # τη στήλη για ΟΛΑ τα samples (train/val/test) μετά το
        # transform. Καθαρίζουμε εδώ πριν το fit ως ασφάλεια.
        train_matrix = np.nan_to_num(
            train_matrix, nan=0.0, posinf=0.0, neginf=0.0
        )

        scaler.fit(train_matrix)

        for windows in (train_windows, validation_windows, test_windows):

            if len(windows) == 0:
                continue

            matrix = np.stack([sample[key] for sample in windows])

            matrix = np.nan_to_num(
                matrix, nan=0.0, posinf=0.0, neginf=0.0
            )

            scaled = scaler.transform(matrix).astype(np.float32)

            for sample, row in zip(windows, scaled):
                sample[key] = row

    @staticmethod
    def _windows_to_arrays(windows: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Converts a flat list of window-dicts (as produced by the
        WindowSegmenter / FeatureExtractor) into stacked NumPy arrays
        ready to be consumed by the PyTorch Dataset classes.

        Τα raw σήματα (eeg/eda/ppg) αποθηκεύονται σε float16 αντί για
        float32 -- αυτά είναι κατά πολύ ο μεγαλύτερος καταναλωτής
        μνήμης (π.χ. ένα πλήρες 32-subject EEG train array μπορεί να
        φτάσει τα 3+ GB σε float32), και σε μηχάνημα με μόνο 8GB RAM
        αυτό οδηγεί σε βαρύ OS-level swapping που επιδεινώνεται
        προοδευτικά epoch με epoch (παρατηρήθηκε: ~0.8s/batch στο
        epoch 1 -> ~7.5s/batch στο epoch 9, δηλαδή ~10x πιο αργό).
        Τα δεδομένα είναι ήδη z-scored (τυπικές τιμές μέσα στο [-5, 5]),
        άρα η μειωμένη ακρίβεια/εύρος του float16 δεν επηρεάζει
        ουσιαστικά την ποιότητα εκπαίδευσης. Μετατρέπονται ξανά σε
        float32 tensor ανά-δείγμα μέσα στο Dataset.__getitem__
        (αμελητέο κόστος, γίνεται μόνο για το batch, όχι για όλο το
        dataset).
        """

        return {
            "eeg": np.stack([w["eeg"] for w in windows]).astype(np.float16),
            "eda": np.stack([w["eda"] for w in windows]).astype(np.float16),
            "ppg": np.stack([w["ppg"] for w in windows]).astype(np.float16),
            "features": np.stack([w["features"] for w in windows]),
            "physio_features": np.stack([w["physio_features"] for w in windows]),
            "labels": np.asarray([w["label"] for w in windows], dtype=np.int64),
            "subjects": np.asarray([w["subject_id"] for w in windows]),
            "trials": np.asarray([w["trial_id"] for w in windows], dtype=np.int64),
            "windows": np.asarray([w["window_id"] for w in windows], dtype=np.int64),
        }

    def run(self):

        logger.info("")
        logger.info("======================================")
        logger.info("Running Data Pipeline")
        logger.info("======================================")

        ###############################################
        # STEP 1
        ###############################################

        logger.info("Loading DEAP dataset...")

        if hasattr(self.loader, "load_dataset"):
            subjects = self.loader.load_dataset()
        elif hasattr(self.loader, "load_data"):
            subjects = self.loader.load_data()
        elif hasattr(self.loader, "load_all_subjects"):
            subjects = self.loader.load_all_subjects()
        else:
            raise AttributeError("DEAPLoader has no known load method (load_dataset/load_data/load_all_subjects).")

        logger.info(
            f"Loaded {len(subjects)} subjects."
        )

        ###############################################
        # STEP 2
        ###############################################

        logger.info("Selecting EEG / EDA / PPG channels...")

        # Τα modalities επιλέγονται/διαμορφώνονται ήδη από τον Loader
        logger.info("Signal selection complete.")

        ###############################################
        # STEP 3
        ###############################################

        logger.info("Generating binary labels...")

        # [ΔΙΟΡΘΩΣΗ 1]: Κλήση add_binary_labels
        subjects = self.label_generator.add_binary_labels(
            subjects
        )

        logger.info("Labels generated.")

        ###############################################
        # STEP 4
        ###############################################

        logger.info("Organizing subjects...")

        # [ΔΙΟΡΘΩΣΗ 3]: Modalities Separation πριν την προσθήκη στον SubjectManager
        if isinstance(subjects, dict):
            for sid, data in subjects.items():
                if isinstance(data, dict):
                    raw_data = data.get("data")
                    if raw_data is not None and isinstance(raw_data, np.ndarray) and raw_data.ndim == 3:
                        eeg_data = raw_data[:, :32, :]
                        eda_data = raw_data[:, EDA_CHANNEL:EDA_CHANNEL + 1, :]
                        ppg_data = raw_data[:, PPG_CHANNEL:PPG_CHANNEL + 1, :]

                        # Per-trial pre-stimulus baseline correction: αφαιρεί
                        # τον μέσο όρο του pre-stimulus (πρώτα BASELINE_SAMPLES)
                        # τμήματος από κάθε trial και πετάει έξω αυτό το
                        # τμήμα (δεν αντιστοιχεί στο emotion label του trial).
                        baseline_samples = self.config.BASELINE_SAMPLES

                        eeg_data = remove_trial_baseline(eeg_data, baseline_samples)
                        eda_data = remove_trial_baseline(eda_data, baseline_samples)
                        ppg_data = remove_trial_baseline(ppg_data, baseline_samples)
                    else:
                        eeg_data = data.get("eeg")
                        eda_data = data.get("eda")
                        ppg_data = data.get("ppg")

                    self.subject_manager.add_subject(
                        subject_id=sid,
                        eeg=eeg_data,
                        eda=eda_data,
                        ppg=ppg_data,
                        labels=data.get("labels"),
                        binary_valence=data.get("binary_valence"),
                    )
            subject_dict = self.subject_manager.subjects if self.subject_manager.subjects else subjects
        else:
            subject_dict = subjects

        logger.info(
            f"{len(subject_dict)} subjects organized."
        )

        ###############################################
        # STEP 5
        ###############################################

        logger.info("Splitting subjects...")

        # [ΔΙΟΡΘΩΣΗ 2]: Χρήση train_val_test_split με subject keys
        subject_keys = list(subject_dict.keys())
        splits = self.splitter.train_val_test_split(
            subject_ids=subject_keys,
            train_ratio=self.config.TRAIN_RATIO,
            val_ratio=self.config.VAL_RATIO,
            test_ratio=self.config.TEST_RATIO
        )

        train_subjects = {k: subject_dict[k] for k in splits["train"]}
        validation_subjects = {k: subject_dict[k] for k in splits["validation"]}
        test_subjects = {k: subject_dict[k] for k in splits["test"]}

        logger.info(
            f"Train Subjects: {len(train_subjects)}"
        )

        logger.info(
            f"Validation Subjects: {len(validation_subjects)}"
        )

        logger.info(
            f"Test Subjects: {len(test_subjects)}"
        )

        ############################################################
        # STEP 6
        # Compute Subject Statistics
        ############################################################

        logger.info("")
        logger.info("======================================")
        logger.info("Computing Subject Statistics")
        logger.info("======================================")

        self.preprocessing.compute_subject_statistics(
            train_subjects
        )

        logger.info("Subject statistics computed successfully.")

        ############################################################
        # STEP 7
        # PREPROCESS TRAIN SUBJECTS
        ############################################################

        logger.info("")
        logger.info("======================================")
        logger.info("Preprocessing Train Subjects")
        logger.info("======================================")

        processed_train_subjects = {}

        for subject_id, subject_data in train_subjects.items():

            logger.info(f"Processing Subject {subject_id}")

            processed_train_subjects[subject_id] = \
                self.preprocessing.process_subject(
                    subject_data
                )

        logger.info(
            f"Finished preprocessing "
            f"{len(processed_train_subjects)} train subjects."
        )

        ############################################################
        # STEP 8
        # PREPROCESS VALIDATION SUBJECTS
        ############################################################

        logger.info("")
        logger.info("======================================")
        logger.info("Preprocessing Validation Subjects")
        logger.info("======================================")

        processed_validation_subjects = {}

        for subject_id, subject_data in validation_subjects.items():

            logger.info(f"Processing Subject {subject_id}")

            processed_validation_subjects[subject_id] = \
                self.preprocessing.process_subject(
                    subject_data
                )

        logger.info(
            f"Finished preprocessing "
            f"{len(processed_validation_subjects)} validation subjects."
        )

        ############################################################
        # STEP 9
        # PREPROCESS TEST SUBJECTS
        ############################################################

        logger.info("")
        logger.info("======================================")
        logger.info("Preprocessing Test Subjects")
        logger.info("======================================")

        processed_test_subjects = {}

        for subject_id, subject_data in test_subjects.items():

            logger.info(f"Processing Subject {subject_id}")

            processed_test_subjects[subject_id] = \
                self.preprocessing.process_subject(
                    subject_data
                )

        logger.info(
            f"Finished preprocessing "
            f"{len(processed_test_subjects)} test subjects."
        )

        ############################################################
        # CHECKS
        ############################################################

        logger.info("")
        logger.info("======================================")
        logger.info("Preprocessing Summary")
        logger.info("======================================")

        logger.info(
            f"Train Subjects : {len(processed_train_subjects)}"
        )

        logger.info(
            f"Validation Subjects : {len(processed_validation_subjects)}"
        )

        logger.info(
            f"Test Subjects : {len(processed_test_subjects)}"
        )

        ############################################################
        # MEMORY CLEANUP
        # Οι raw (μη επεξεργασμένες) υπογραφές subjects καθώς και το
        # εσωτερικό dict του SubjectManager δεν χρειάζονται πλέον
        # (έχουμε ήδη τα processed_*_subjects). Σε μηχανήματα με
        # περιορισμένη RAM (π.χ. 8GB) το πλήρες 32-subject DEAP
        # dataset μπορεί εύκολα να εξαντλήσει τη μνήμη αν κρατάμε
        # πολλαπλά αντίγραφα ζωντανά ταυτόχρονα.
        ############################################################

        self.subject_manager.subjects.clear()

        num_train_subjects = len(train_subjects)
        num_validation_subjects = len(validation_subjects)
        num_test_subjects = len(test_subjects)

        # Κρατάμε μόνο τα (ελαφριά) subject IDs, όχι ολόκληρα τα
        # δεδομένα τους, για το τελικό return value του pipeline.
        train_subject_ids = list(train_subjects.keys())

        del subjects, subject_dict
        del train_subjects, validation_subjects, test_subjects

        gc.collect()

        ############################################################
        # Safety Checks
        ############################################################

        if len(processed_train_subjects) == 0:
            raise RuntimeError(
                "No train subjects after preprocessing."
            )

        if len(processed_validation_subjects) == 0:
            raise RuntimeError(
                "No validation subjects after preprocessing."
            )

        if len(processed_test_subjects) == 0:
            raise RuntimeError(
                "No test subjects after preprocessing."
            )

        logger.info("")
        logger.info("All preprocessing steps completed successfully.")

        ############################################################
        # STEP 10
        # WINDOW SEGMENTATION
        ############################################################

        logger.info("")
        logger.info("======================================")
        logger.info("Window Segmentation")
        logger.info("======================================")

        train_windows = self.segmenter.process_subjects(
            processed_train_subjects
        )

        validation_windows = self.segmenter.process_subjects(
            processed_validation_subjects
        )

        test_windows = self.segmenter.process_subjects(
            processed_test_subjects
        )

        logger.info(
            f"Train Windows : {len(train_windows)}"
        )

        logger.info(
            f"Validation Windows : {len(validation_windows)}"
        )

        logger.info(
            f"Test Windows : {len(test_windows)}"
        )

        num_train_windows = len(train_windows)
        num_validation_windows = len(validation_windows)
        num_test_windows = len(test_windows)

        ############################################################
        # MEMORY CLEANUP
        # Τα processed_*_subjects (πλήρη προεπεξεργασμένα σήματα ανά
        # trial) δεν χρειάζονται πλέον — τα windows κρατάνε ήδη τα
        # τεμαχισμένα (windowed) segments τους. Λόγω του overlap
        # (75%), τα windows είναι ~4x μεγαλύτερα σε συνολικό μέγεθος
        # από τα αρχικά trials, οπότε είναι κρίσιμο να ελευθερώσουμε
        # τα processed_*_subjects πριν προχωρήσουμε.
        ############################################################

        del processed_train_subjects
        del processed_validation_subjects
        del processed_test_subjects

        gc.collect()

        ############################################################
        # STEP 11
        # FEATURE EXTRACTION
        ############################################################

        logger.info("")
        logger.info("======================================")
        logger.info("Extracting Handcrafted Features")
        logger.info("======================================")

        train_windows = self.feature_extractor.process_dataset(
            train_windows
        )

        validation_windows = self.feature_extractor.process_dataset(
            validation_windows
        )

        test_windows = self.feature_extractor.process_dataset(
            test_windows
        )

        logger.info("Feature extraction completed.")

        ############################################################
        # STEP 11b
        # FEATURE SCALING (fit on Train, apply on Val/Test)
        ############################################################

        logger.info("")
        logger.info("======================================")
        logger.info("Scaling Handcrafted Features")
        logger.info("======================================")

        self._fit_transform_feature_key(
            train_windows, validation_windows, test_windows,
            key="features", scaler=self.feature_scaler,
        )

        self._fit_transform_feature_key(
            train_windows, validation_windows, test_windows,
            key="physio_features", scaler=self.physio_feature_scaler,
        )

        logger.info("Feature scaling completed (StandardScaler, fit on train).")

        ############################################################
        # STEP 12
        # BUILD DATASETS
        ############################################################

        logger.info("")
        logger.info("======================================")
        logger.info("Building PyTorch Datasets")
        logger.info("======================================")

        model_name = getattr(self.config, "MODEL_NAME", "hybrid")

        # Επεξεργαζόμαστε ένα split τη φορά και ελευθερώνουμε αμέσως
        # τη λίστα windows (dicts) μόλις έχει μετατραπεί σε stacked
        # arrays, ώστε να μην έχουμε ταυτόχρονα στη μνήμη τόσο τη
        # λίστα-πηγή όσο και το stacked array (διπλασιασμός μνήμης),
        # πόσο μάλλον και για τα 3 splits μαζί. Κρίσιμο σε μηχανήματα
        # με περιορισμένη RAM (π.χ. 8GB) όταν τρέχει το πλήρες
        # 32-subject DEAP dataset.

        train_arrays = self._windows_to_arrays(train_windows)
        del train_windows
        gc.collect()

        train_dataset = build_dataset(
            model_name=model_name,
            eeg=train_arrays["eeg"],
            eda=train_arrays["eda"],
            ppg=train_arrays["ppg"],
            features=train_arrays["features"],
            physio_features=train_arrays["physio_features"],
            labels=train_arrays["labels"],
            subjects=train_arrays["subjects"],
            trials=train_arrays["trials"],
            windows=train_arrays["windows"],
        )

        del train_arrays
        gc.collect()

        validation_arrays = self._windows_to_arrays(validation_windows)
        del validation_windows
        gc.collect()

        validation_dataset = build_dataset(
            model_name=model_name,
            eeg=validation_arrays["eeg"],
            eda=validation_arrays["eda"],
            ppg=validation_arrays["ppg"],
            features=validation_arrays["features"],
            physio_features=validation_arrays["physio_features"],
            labels=validation_arrays["labels"],
            subjects=validation_arrays["subjects"],
            trials=validation_arrays["trials"],
            windows=validation_arrays["windows"],
        )

        del validation_arrays
        gc.collect()

        test_arrays = self._windows_to_arrays(test_windows)
        del test_windows
        gc.collect()

        test_dataset = build_dataset(
            model_name=model_name,
            eeg=test_arrays["eeg"],
            eda=test_arrays["eda"],
            ppg=test_arrays["ppg"],
            features=test_arrays["features"],
            physio_features=test_arrays["physio_features"],
            labels=test_arrays["labels"],
            subjects=test_arrays["subjects"],
            trials=test_arrays["trials"],
            windows=test_arrays["windows"],
        )

        del test_arrays
        gc.collect()

        logger.info(
            f"Train Dataset : {len(train_dataset)} samples"
        )

        logger.info(
            f"Validation Dataset : {len(validation_dataset)} samples"
        )

        logger.info(
            f"Test Dataset : {len(test_dataset)} samples"
        )

        ############################################################
        # SANITY CHECKS
        ############################################################

        if len(train_dataset) == 0:
            raise RuntimeError(
                "Train dataset is empty."
            )

        if len(validation_dataset) == 0:
            raise RuntimeError(
                "Validation dataset is empty."
            )

        if len(test_dataset) == 0:
            raise RuntimeError(
                "Test dataset is empty."
            )

        logger.info("")
        logger.info("Datasets created successfully.")

        ############################################################
        # STEP 13
        # BUILD DATALOADERS
        ############################################################

        logger.info("")
        logger.info("======================================")
        logger.info("Building DataLoaders")
        logger.info("======================================")

        loader_factory = DataLoaderFactory(
            batch_size=self.config.BATCH_SIZE,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=getattr(self.config, "PIN_MEMORY", True),
            weighted_sampling=getattr(self.config, "WEIGHTED_SAMPLING", True),
        )

        train_labels = train_dataset.labels

        loaders = loader_factory.build(
            train_dataset=train_dataset,
            val_dataset=validation_dataset,
            test_dataset=test_dataset,
            train_labels=train_labels,
        )

        train_loader = loaders["train"]
        validation_loader = loaders["validation"]
        test_loader = loaders["test"]

        logger.info("DataLoaders successfully created.")

        ############################################################
        # PIPELINE SUMMARY
        ############################################################

        logger.info("")
        logger.info("======================================")
        logger.info("DATA PIPELINE SUMMARY")
        logger.info("======================================")

        logger.info(f"Train Subjects      : {num_train_subjects}")
        logger.info(f"Validation Subjects : {num_validation_subjects}")
        logger.info(f"Test Subjects       : {num_test_subjects}")

        logger.info("")

        logger.info(f"Train Windows       : {num_train_windows}")
        logger.info(f"Validation Windows  : {num_validation_windows}")
        logger.info(f"Test Windows        : {num_test_windows}")

        logger.info("")

        logger.info(f"Train Samples       : {len(train_dataset)}")
        logger.info(f"Validation Samples  : {len(validation_dataset)}")
        logger.info(f"Test Samples        : {len(test_dataset)}")

        logger.info("")

        logger.info(f"Batch Size          : {self.config.BATCH_SIZE}")
        logger.info(f"Window Size         : {self.config.WINDOW_SIZE}")
        logger.info(f"Overlap             : {self.config.OVERLAP}")
        logger.info(f"Sampling Rate       : {self.config.SAMPLING_RATE}")

        logger.info("")

        logger.info("Pipeline finished successfully.")

        ############################################################
        # RETURN
        # [ΔΙΟΡΘΩΣΗ 4]: Επιστροφή 8-tuple για το test_data_pipeline.py
        ############################################################

        return (
            train_dataset,
            validation_dataset,
            test_dataset,
            train_loader,
            validation_loader,
            test_loader,
            train_labels,
            train_subject_ids,
        )