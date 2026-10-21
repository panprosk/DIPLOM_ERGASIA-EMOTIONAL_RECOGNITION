from __future__ import annotations
import gc
import logging
import pickle
from collections import defaultdict
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
    def _subject_normalize_features(windows: List[Dict]) -> None:
        """
        Label-free per-subject z-score of the handcrafted features
        (statistics from ALL windows of the same subject, in place).
        Removes subject-specific offsets/scales, a standard
        cross-subject normalization; no labels and no other subjects
        are involved, so there is no train/test leakage.
        """

        by_subject: Dict[str, List[Dict]] = defaultdict(list)
        for window in windows:
            by_subject[str(window["subject_id"])].append(window)

        for subject_windows in by_subject.values():
            for key in ("features", "physio_features"):
                matrix = np.stack([w[key] for w in subject_windows]).astype(np.float64)
                matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
                mean = matrix.mean(axis=0, keepdims=True)
                std = matrix.std(axis=0, keepdims=True)
                normalized = ((matrix - mean) / np.maximum(std, 1e-6)).astype(np.float32)
                for window, row in zip(subject_windows, normalized):
                    window[key] = row

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

        # ΣΗΜΑΝΤΙΚΟ (memory): np.stack([...]).astype(float16), ακόμη
        # και np.stack([...astype(float16)...]), χτίζει ΠΡΩΤΑ μια
        # Python list με ΟΛΑ τα per-window arrays (ίδιο μέγεθος με το
        # τελικό stacked array) ΚΑΙ ταυτόχρονα κρατάει ζωντανά τα
        # αρχικά float32 per-window arrays μέσα στα window dicts, πριν
        # καν ξεκινήσει το np.stack -- δηλαδή peak μνήμη έως και ~4x
        # το τελικό μέγεθος (π.χ. για EEG (32560,32,768): float32
        # originals 2.98 GiB + float16 list 1.49 GiB + τελικό stacked
        # 1.49 GiB ταυτόχρονα -> MemoryError σε μηχάνημα με 8GB RAM).
        #
        # Λύση: preallocate το τελικό float16 array και γράφουμε ένα-
        # ένα window μέσα του, ΑΔΕΙΑΖΟΝΤΑΣ (w[key] = None) το αρχικό
        # float32 reference αμέσως μετά -- έτσι ο Python garbage
        # collector μπορεί να ελευθερώσει σταδιακά τα αρχικά arrays
        # καθώς προχωράμε, αντί να τα κρατάει όλα ζωντανά ταυτόχρονα.
        # Peak μνήμη πέφτει σε ~1x το τελικό μέγεθος (+ ένα window τη
        # φορά), δηλαδή ~3-4x λιγότερη μνήμη από πριν.
        def _stack_raw_signal(key: str) -> np.ndarray:

            first_shape = windows[0][key].shape

            out = np.empty((len(windows),) + first_shape, dtype=np.float16)

            for i, w in enumerate(windows):
                out[i] = w[key]
                w[key] = None  # ελευθερώνει το float32 original νωρίς

            return out

        eeg_arr = _stack_raw_signal("eeg")
        eda_arr = _stack_raw_signal("eda")
        ppg_arr = _stack_raw_signal("ppg")

        return {
            "eeg": eeg_arr,
            "eda": eda_arr,
            "ppg": ppg_arr,
            "features": np.stack([w["features"] for w in windows]),
            "physio_features": np.stack([w["physio_features"] for w in windows]),
            "labels": np.asarray([w["label"] for w in windows], dtype=np.int64),
            "subjects": np.asarray([w["subject_id"] for w in windows]),
            "trials": np.asarray([w["trial_id"] for w in windows], dtype=np.int64),
            "windows": np.asarray([w["window_id"] for w in windows], dtype=np.int64),
        }

    def _cache_protocol_name(self) -> str:
        return (
            "deap_official_v2"
            if getattr(self.config, "PREPROCESSING_VERSION", "v1") == "v2"
            else "legacy"
        )

    def run(self, cache_path: str | None = None):

        if cache_path is not None and Path(cache_path).exists():
            logger.info("")
            logger.info("======================================")
            logger.info(f"Loading cached pipeline datasets <- {cache_path}")
            logger.info("======================================")

            with open(cache_path, "rb") as f:
                cached = pickle.load(f)

            protocol = cached.pop("preprocessing_protocol", None) or {}
            cached_name = protocol.get("name", "legacy")
            if cached_name != self._cache_protocol_name():
                raise RuntimeError(
                    f"Cache '{cache_path}' was built with preprocessing "
                    f"'{cached_name}' but config requests "
                    f"'{self._cache_protocol_name()}'. Use another --cache-path."
                )
            return self._build_loaders_and_return(**cached)

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

        if getattr(self.config, "PREPROCESSING_VERSION", "v1") == "v2":
            return self._run_v2(subjects, cache_path)
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

                    trial_labels = data.get("labels")
                    trial_binary_valence = data.get("binary_valence")

                    # --- Valence "νεκρή ζώνη" filtering (label-noise reduction) ---
                    # Αποκλείει trials με valence πολύ κοντά στο decision
                    # threshold (ασαφή/οριακά ground-truth labels) πριν το
                    # windowing, ώστε ΚΑΝΕΝΑ downstream στάδιο (train/val/test)
                    # να μην βλέπει ποτέ αυτά τα trials.
                    margin = getattr(self.config, "VALENCE_MARGIN", 0.0)

                    if margin and margin > 0.0 and trial_labels is not None:

                        valence = trial_labels[:, 0]
                        threshold = self.config.VALENCE_THRESHOLD

                        keep_mask = np.abs(valence - threshold) >= margin

                        n_dropped = int((~keep_mask).sum())

                        if n_dropped > 0:
                            logger.info(
                                f"  Subject {sid}: dropping {n_dropped}/"
                                f"{len(valence)} trials with valence in "
                                f"[{threshold - margin:.1f}, {threshold + margin:.1f}] "
                                f"(ασαφή/οριακά labels)."
                            )

                        eeg_data = eeg_data[keep_mask]
                        eda_data = eda_data[keep_mask]
                        ppg_data = ppg_data[keep_mask]
                        trial_labels = trial_labels[keep_mask]

                        if trial_binary_valence is not None:
                            trial_binary_valence = trial_binary_valence[keep_mask]

                    self.subject_manager.add_subject(
                        subject_id=sid,
                        eeg=eeg_data,
                        eda=eda_data,
                        ppg=ppg_data,
                        labels=trial_labels,
                        binary_valence=trial_binary_valence,
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

        # Fit normalization only on training subjects. The same statistics
        # are then applied to validation/test subjects, avoiding the
        # train-vs-unseen-subject protocol mismatch of per-trial fallback.
        if getattr(self.config, "TRAIN_ONLY_POPULATION_NORMALIZATION", False):
            self.preprocessing.fit_population_statistics(train_subjects)

        ############################################################
        # STEP 6
        # Normalization protocol
        ############################################################

        # ΔΙΟΡΘΩΣΗ (cross-subject generalization): πριν, μόνο τα train
        # subjects έπαιρναν subject-level z-score (mean/std πάνω σε ΟΛΑ
        # τα trials τους), ενώ τα validation/test subjects (άγνωστα, δεν
        # υπάρχουν subject-level στατιστικά γι' αυτά) έκαναν fallback σε
        # per-trial normalization. Αυτό δημιουργούσε ασυνέπεια πρωτοκόλλου
        # ανάμεσα σε train και eval -- το μοντέλο έβλεπε στο training μια
        # συστηματικά διαφορετική κατανομή scale/offset από αυτή που θα
        # συναντήσει σε πραγματικά άγνωστα subjects. Η per-trial
        # normalization είναι ήδη το ΜΟΝΑΔΙΚΟ πρωτόκολλο που δουλεύει σε
        # unseen subjects (δεν χρειάζεται ιστορικό του subject), άρα
        # εφαρμόζεται πλέον ομοιόμορφα σε train/validation/test -- ίδιο
        # rationale με το GroupNorm αντί για BatchNorm στο EEG CNN
        # encoder (subject-invariant, όχι population-statistics based).
        logger.info("")
        logger.info("======================================")
        normalization_name = (
            "train-only population z-score"
            if getattr(self.config, "TRAIN_ONLY_POPULATION_NORMALIZATION", False)
            else "per-trial"
        )
        logger.info(
            f"Normalization: {normalization_name} "
            "(train/val/test protocol)"
        )
        logger.info("======================================")

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

        if cache_path is not None:
            logger.info(f"Saving pipeline cache -> {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "train_dataset": train_dataset,
                    "validation_dataset": validation_dataset,
                    "test_dataset": test_dataset,
                    "num_train_subjects": num_train_subjects,
                    "num_validation_subjects": num_validation_subjects,
                    "num_test_subjects": num_test_subjects,
                    "num_train_windows": num_train_windows,
                    "num_validation_windows": num_validation_windows,
                    "num_test_windows": num_test_windows,
                    "train_subject_ids": train_subject_ids,
                    "preprocessing_protocol": {
                        "name": "legacy",
                        "baseline_seconds": self.config.BASELINE_SECONDS,
                        "eeg_bandpass": (
                            self.config.EEG_LOWCUT,
                            self.config.EEG_HIGHCUT,
                        ),
                        "eeg_notch": self.config.EEG_NOTCH,
                        "normalization": (
                            "train_only_channel_population_zscore"
                            if getattr(
                                self.config,
                                "TRAIN_ONLY_POPULATION_NORMALIZATION",
                                False,
                            )
                            else "per_trial_zscore"
                        ),
                        "window_seconds": self.config.WINDOW_SIZE_SEC,
                        "overlap": self.config.OVERLAP,
                    },
                }, f)

        return self._build_loaders_and_return(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
            num_train_subjects=num_train_subjects,
            num_validation_subjects=num_validation_subjects,
            num_test_subjects=num_test_subjects,
            num_train_windows=num_train_windows,
            num_validation_windows=num_validation_windows,
            num_test_windows=num_test_windows,
            train_subject_ids=train_subject_ids,
        )

    def _v2_subject_windows(self, sid: str, data: Dict) -> List[Dict]:
        from src.preprocessing.deap_official_v2 import OfficialDeapV2Preprocessor

        if not hasattr(self, "_v2_preprocessor"):
            self._v2_preprocessor = OfficialDeapV2Preprocessor(
                sampling_rate=self.config.SAMPLING_RATE,
                window_size=self.config.WINDOW_SIZE,
                overlap=self.config.OVERLAP,
                baseline_samples=self.config.BASELINE_SAMPLES,
                eeg_norm=getattr(self.config, "EEG_NORM", "trial"),
            )

        logger.info(f"Processing Subject {sid} (official DEAP v2)")

        return self._v2_preprocessor.process_subject(
            sid,
            np.asarray(data["data"], dtype=np.float32),
            np.asarray(data["labels"]),
            data.get("binary_valence"),
        )

    def _run_v2(self, subjects: Dict, cache_path: str | None):
        """
        Preprocessing v2 (επίσημο preprocessed DEAP). Ίδιο subject split
        με το v1 (ίδιος splitter/seed), ώστε τα αποτελέσματα να είναι
        άμεσα συγκρίσιμα.
        """

        model_name = getattr(self.config, "MODEL_NAME", "hybrid")

        splits = self.splitter.train_val_test_split(
            subject_ids=list(subjects.keys()),
            train_ratio=self.config.TRAIN_RATIO,
            val_ratio=self.config.VAL_RATIO,
            test_ratio=self.config.TEST_RATIO,
        )

        logger.info(
            f"Split (v2): train={len(splits['train'])} "
            f"val={len(splits['validation'])} test={len(splits['test'])}"
        )

        window_sets = {}
        for split_name in ("train", "validation", "test"):
            windows: List[Dict] = []
            for sid in splits[split_name]:
                windows.extend(self._v2_subject_windows(sid, subjects[sid]))
            if not windows:
                raise RuntimeError(f"{split_name} split is empty after v2 preprocessing.")
            window_sets[split_name] = windows

        del subjects
        gc.collect()

        train_windows = window_sets["train"]
        validation_windows = window_sets["validation"]
        test_windows = window_sets["test"]

        num_windows = tuple(len(window_sets[k]) for k in ("train", "validation", "test"))
        train_subject_ids = list(splits["train"])

        logger.info("Scaling features (StandardScaler, fit on train windows only)...")

        self._fit_transform_feature_key(
            train_windows, validation_windows, test_windows,
            key="features", scaler=self.feature_scaler,
        )
        self._fit_transform_feature_key(
            train_windows, validation_windows, test_windows,
            key="physio_features", scaler=self.physio_feature_scaler,
        )

        datasets = []
        for windows in (train_windows, validation_windows, test_windows):
            arrays = self._windows_to_arrays(windows)
            datasets.append(build_dataset(
                model_name=model_name,
                eeg=arrays["eeg"],
                eda=arrays["eda"],
                ppg=arrays["ppg"],
                features=arrays["features"],
                physio_features=arrays["physio_features"],
                labels=arrays["labels"],
                subjects=arrays["subjects"],
                trials=arrays["trials"],
                windows=arrays["windows"],
            ))
            del arrays
            gc.collect()

        del window_sets, train_windows, validation_windows, test_windows
        gc.collect()

        train_dataset, validation_dataset, test_dataset = datasets

        payload = {
            "train_dataset": train_dataset,
            "validation_dataset": validation_dataset,
            "test_dataset": test_dataset,
            "num_train_subjects": len(splits["train"]),
            "num_validation_subjects": len(splits["validation"]),
            "num_test_subjects": len(splits["test"]),
            "num_train_windows": num_windows[0],
            "num_validation_windows": num_windows[1],
            "num_test_windows": num_windows[2],
            "train_subject_ids": train_subject_ids,
        }

        if cache_path is not None:
            logger.info(f"Saving pipeline cache -> {cache_path}")
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        **payload,
                        "preprocessing_protocol": {
                            **self._v2_preprocessor.metadata(),
                            "name": "deap_official_v2",
                        },
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

        return self._build_loaders_and_return(**payload)

    def _build_loaders_and_return(
        self,
        train_dataset,
        validation_dataset,
        test_dataset,
        num_train_subjects,
        num_validation_subjects,
        num_test_subjects,
        num_train_windows,
        num_validation_windows,
        num_test_windows,
        train_subject_ids,
    ):
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

    ################################################################
    # Subject-wise Cross-Validation support (LOSO / k-fold)
    ################################################################
    #
    # run() above always performs ONE internal train/val/test subject
    # split (via SubjectSplitter.train_val_test_split) and only
    # preprocesses those subjects. For a proper cross-subject
    # generalization evaluation (full Leave-One-Subject-Out over every
    # subject, or k-fold subject CV) every subject needs to be
    # preprocessed exactly once and then be freely reassignable to
    # different folds. Preprocessing itself is NOT split-dependent:
    # compute_subject_statistics() is never called anywhere in run(),
    # so every subject already only receives per-trial z-score
    # normalization (see STEP 6 log message above), independent of
    # which split a subject ends up in. The two methods below factor
    # out that expensive, split-independent part (load + label +
    # preprocess + baseline-correct + window + handcrafted features) so
    # a CV harness can compute it ONCE for all subjects and reuse it
    # across every fold -- only the cheap, fold-specific steps (feature
    # scaling fit-on-train + dataset/loader building) are repeated per
    # fold. Without this, a full 32-fold LOSO would redo the
    # multi-hour preprocessing/feature-extraction pass 32 times.

    def prepare_all_subject_windows(
        self,
        cache_path: str | None = None,
        max_subjects: int | None = None,
        trials_per_subject: int | None = None,
    ) -> List[Dict]:
        """
        Loads, labels, preprocesses, baseline-corrects, windows, and
        extracts handcrafted features for EVERY subject -- no
        train/validation/test split is applied. Returns a flat list of
        window-dicts (same format produced by WindowSegmenter +
        FeatureExtractor inside run()), each tagged with its own
        "subject_id". Use `build_fold_loaders` to turn an arbitrary
        subject-id partition of this list into train/validation/test
        datasets/loaders for one CV fold.
        """

        if cache_path is not None and Path(cache_path).exists():
            logger.info(f"Loading cached all-subject windows <- {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        logger.info("")
        logger.info("======================================")
        logger.info("Preparing ALL subjects (CV mode, no split)")
        logger.info("======================================")

        logger.info("Loading DEAP dataset...")

        if hasattr(self.loader, "load_dataset"):
            subjects = self.loader.load_dataset()
        elif hasattr(self.loader, "load_data"):
            subjects = self.loader.load_data()
        elif hasattr(self.loader, "load_all_subjects"):
            subjects = self.loader.load_all_subjects()
        else:
            raise AttributeError(
                "DEAPLoader has no known load method "
                "(load_dataset/load_data/load_all_subjects)."
            )

        logger.info(f"Loaded {len(subjects)} subjects.")

        subjects = self.label_generator.add_binary_labels(subjects)

        subject_ids = sorted(subjects.keys())
        if max_subjects is not None:
            subject_ids = subject_ids[:max_subjects]
        subjects = {sid: subjects[sid] for sid in subject_ids}

        if getattr(self.config, "PREPROCESSING_VERSION", "v1") == "v2":
            v2_windows: List[Dict] = []
            for sid, data in subjects.items():
                if trials_per_subject is not None:
                    data = {
                        **data,
                        "data": data["data"][:trials_per_subject],
                        "labels": data["labels"][:trials_per_subject],
                        "binary_valence": (
                            data["binary_valence"][:trials_per_subject]
                            if data.get("binary_valence") is not None else None
                        ),
                    }
                v2_windows.extend(self._v2_subject_windows(sid, data))
            if cache_path is not None:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(v2_windows, f, protocol=pickle.HIGHEST_PROTOCOL)
            return v2_windows

        margin = getattr(self.config, "VALENCE_MARGIN", 0.0)
        baseline_samples = self.config.BASELINE_SAMPLES

        all_windows: List[Dict] = []

        for sid, data in subjects.items():

            logger.info(f"Processing Subject {sid}")

            raw_data = data.get("data")
            eeg_data = raw_data[:, :32, :]
            eda_data = raw_data[:, EDA_CHANNEL:EDA_CHANNEL + 1, :]
            ppg_data = raw_data[:, PPG_CHANNEL:PPG_CHANNEL + 1, :]
            trial_labels = data.get("labels")
            trial_binary_valence = data.get("binary_valence")

            if trials_per_subject is not None:
                eeg_data = eeg_data[:trials_per_subject]
                eda_data = eda_data[:trials_per_subject]
                ppg_data = ppg_data[:trials_per_subject]
                trial_labels = trial_labels[:trials_per_subject]
                if trial_binary_valence is not None:
                    trial_binary_valence = trial_binary_valence[:trials_per_subject]

            eeg_data = remove_trial_baseline(eeg_data, baseline_samples)
            eda_data = remove_trial_baseline(eda_data, baseline_samples)
            ppg_data = remove_trial_baseline(ppg_data, baseline_samples)

            if margin and margin > 0.0 and trial_labels is not None:

                valence = trial_labels[:, 0]
                threshold = self.config.VALENCE_THRESHOLD
                keep_mask = np.abs(valence - threshold) >= margin

                eeg_data = eeg_data[keep_mask]
                eda_data = eda_data[keep_mask]
                ppg_data = ppg_data[keep_mask]
                trial_labels = trial_labels[keep_mask]
                if trial_binary_valence is not None:
                    trial_binary_valence = trial_binary_valence[keep_mask]

            subject_data = {
                "subject_id": sid,
                "eeg": eeg_data,
                "eda": eda_data,
                "ppg": ppg_data,
                "labels": trial_labels,
                "binary_valence": trial_binary_valence,
            }

            processed = self.preprocessing.process_subject(subject_data)

            windows = self.segmenter.segment_subject(processed)

            all_windows.extend(windows)

            del processed, subject_data
            gc.collect()

        logger.info(f"Total windows (all subjects): {len(all_windows)}")

        logger.info("Extracting handcrafted features for all windows...")
        all_windows = self.feature_extractor.process_dataset(all_windows)
        logger.info("Feature extraction completed.")

        if cache_path is not None:
            logger.info(f"Saving all-subject window cache -> {cache_path}")
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(all_windows, f, protocol=pickle.HIGHEST_PROTOCOL)

        return all_windows

    def build_fold_loaders(
        self,
        all_windows: List[Dict],
        split: Dict[str, List[str]],
        model_name: str | None = None,
    ):
        """
        Turns a subject-id partition
        ``{"train": [...], "validation": [...], "test": [...]}`` of
        `all_windows` (as returned by `prepare_all_subject_windows`)
        into fold-specific PyTorch datasets/loaders. A FRESH feature
        scaler is fit on this fold's train windows only and never
        reused across folds, so there is no cross-fold leakage of
        scaling statistics.
        """

        model_name = model_name or getattr(self.config, "MODEL_NAME", "hybrid")

        by_subject: Dict[str, List[Dict]] = defaultdict(list)
        for window in all_windows:
            by_subject[str(window["subject_id"])].append(window)

        def _collect(subject_ids) -> List[Dict]:
            collected = []
            for sid in subject_ids:
                for window in by_subject.get(str(sid), []):
                    # Shallow copy: eeg/eda/ppg arrays are shared by
                    # reference (never mutated in place below) but
                    # "features"/"physio_features" are copied because
                    # _fit_transform_feature_key overwrites them
                    # in-place per fold with THIS fold's scaler output.
                    collected.append({
                        **window,
                        "features": window["features"].copy(),
                        "physio_features": window["physio_features"].copy(),
                    })
            return collected

        train_windows = _collect(split["train"])
        validation_windows = _collect(split["validation"])
        test_windows = _collect(split["test"])

        if not train_windows or not validation_windows or not test_windows:
            raise RuntimeError(
                "Empty split while building fold loaders (train="
                f"{len(train_windows)}, validation={len(validation_windows)}, "
                f"test={len(test_windows)})."
            )

        if getattr(self.config, "SUBJECT_FEATURE_NORM", False):
            self._subject_normalize_features(
                train_windows + validation_windows + test_windows
            )

        self._fit_transform_feature_key(
            train_windows, validation_windows, test_windows,
            key="features", scaler=FeatureScaler(),
        )
        self._fit_transform_feature_key(
            train_windows, validation_windows, test_windows,
            key="physio_features", scaler=FeatureScaler(),
        )

        train_arrays = self._windows_to_arrays(train_windows)
        del train_windows
        gc.collect()

        validation_arrays = self._windows_to_arrays(validation_windows)
        del validation_windows
        gc.collect()

        test_arrays = self._windows_to_arrays(test_windows)
        del test_windows
        gc.collect()

        train_dataset = build_dataset(model_name=model_name, **train_arrays)
        validation_dataset = build_dataset(model_name=model_name, **validation_arrays)
        test_dataset = build_dataset(model_name=model_name, **test_arrays)

        return self._build_loaders_and_return(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
            num_train_subjects=len(split["train"]),
            num_validation_subjects=len(split["validation"]),
            num_test_subjects=len(split["test"]),
            num_train_windows=len(train_dataset),
            num_validation_windows=len(validation_dataset),
            num_test_windows=len(test_dataset),
            train_subject_ids=list(split["train"]),
        )