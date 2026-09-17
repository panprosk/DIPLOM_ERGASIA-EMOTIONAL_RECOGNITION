"""
paper_lstm_pipeline.py

Data pipeline για την αναπαραγωγή του Saffaryazdi et al. (2022),
"Using Facial Micro-Expressions in Combination With EEG and
Physiological Signals for Emotion Recognition", Frontiers in
Psychology, 13:864047 (χωρίς το facial-micro-expression κομμάτι, βλ.
σημειώσεις χρήστη σημείο 22).

Διαφορές από το src/pipeline/data_pipeline.py (το κύριο pipeline του
Hybrid CNN-MLP / Cross-Attention Transformer)
------------------------------------------------------------------
1. Windowing: 1-δευτερόλεπτο (128 δείγματα), χωρίς επικάλυψη -- αντί
   για το 6s/75% overlap του κύριου pipeline. 60 windows/trial.
2. Features: FFT band powers (EEG) + απλά στατιστικά (EDA/PPG), βλ.
   src/features/paper_lstm_features.py -- πολύ πιο ελαφριά από τα
   NeuroKit2-based features του κύριου pipeline.
3. Labels: ΔΥΟ ξεχωριστά binary προβλήματα (valence, arousal), αντί
   για ένα.
4. Έξοδος: sequences ανά trial (60 windows -> 1 sequence) και όχι
   ανεξάρτητα classified windows -- ώστε να τροφοδοτηθεί το 2-layer
   stacked LSTM (per-modality) του paper, το οποίο βλέπει ολόκληρη
   την χρονική ακολουθία ενός trial πριν κάνει την πρόβλεψη.

Evaluation protocol: subject-independent train/validation/test split
(ίδιο SubjectSplitter με το υπόλοιπο project) -- αυστηρότερο από το
6-fold leave-some-subject-out του paper, όπως συμφωνήθηκε με τον
χρήστη.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List

import numpy as np

from src.config.config import Config
from src.config.signals import EDA_CHANNEL, PPG_CHANNEL
from src.data.deap_loader import DEAPLoader
from src.data.label_generator import LabelGenerator
from src.split.subject_split import SubjectSplitter
from src.preprocessing.eeg_preprocessing import EEGPreprocessor
from src.preprocessing.eda_preprocessing import EDAPreprocessor
from src.preprocessing.ppg_preprocessing import PPGPreprocessor
from src.preprocessing.baseline_correction import remove_trial_baseline
from src.preprocessing.subject_statistics import SubjectStatistics
from src.segmentation.window_segmenter import WindowSegmenter
from src.features.paper_lstm_features import (
    PaperEEGFeatureExtractor,
    PaperEDAFeatureExtractor,
    PaperPPGFeatureExtractor,
)
from src.features.feature_scaler import FeatureScaler

logger = logging.getLogger(__name__)


class PaperLSTMDataPipeline:
    """
    Χτίζει per-trial sequences (EEG/EDA/PPG feature sequences) έτοιμες
    για το PaperLSTMFusionModel.
    """

    def __init__(
        self,
        config: Config,
        window_seconds: float = 1.0,
        overlap: float = 0.0,
        subject_aware_normalization: bool = False,
    ):

        self.config = config
        self.subject_aware_normalization = subject_aware_normalization

        self.loader = DEAPLoader(dataset_path=config.DEAP_PATH)

        self.label_generator = LabelGenerator(threshold=config.VALENCE_THRESHOLD)

        self.splitter = SubjectSplitter(random_seed=config.SEED)

        self.eeg_pre = EEGPreprocessor(
            sampling_rate=config.SAMPLING_RATE,
            lowcut=config.EEG_LOWCUT,
            highcut=config.EEG_HIGHCUT,
            notch=config.EEG_NOTCH,
        )
        self.eda_pre = EDAPreprocessor(sampling_rate=config.SAMPLING_RATE)
        self.ppg_pre = PPGPreprocessor(sampling_rate=config.SAMPLING_RATE)

        self.segmenter = WindowSegmenter(
            sampling_rate=config.SAMPLING_RATE,
            window_seconds=window_seconds,
            overlap=overlap,
        )

        self.eeg_feat = PaperEEGFeatureExtractor(sampling_rate=config.SAMPLING_RATE)
        self.eda_feat = PaperEDAFeatureExtractor(sampling_rate=config.SAMPLING_RATE)
        self.ppg_feat = PaperPPGFeatureExtractor()

        self.eeg_scaler = FeatureScaler()
        self.eda_scaler = FeatureScaler()
        self.ppg_scaler = FeatureScaler()

        self.expected_seq_len = None

    # ------------------------------------------------------------
    # Step 1: load + organize raw subjects
    # ------------------------------------------------------------

    def _load_and_organize(self, max_subjects: int | None = None) -> Dict[str, dict]:

        raw = self.loader.load_all_subjects()

        raw = self.label_generator.add_binary_labels(raw)
        raw = self.label_generator.add_binary_arousal_labels(raw)

        if max_subjects is not None:
            raw = dict(list(raw.items())[:max_subjects])

        organized = {}

        baseline_samples = self.config.BASELINE_SAMPLES

        for subject_id, subject in raw.items():

            data = subject["data"]  # (n_trials, 40, n_samples)

            eeg_data = data[:, :32, :]
            eda_data = data[:, EDA_CHANNEL:EDA_CHANNEL + 1, :]
            ppg_data = data[:, PPG_CHANNEL:PPG_CHANNEL + 1, :]

            eeg_data = remove_trial_baseline(eeg_data, baseline_samples)
            eda_data = remove_trial_baseline(eda_data, baseline_samples)
            ppg_data = remove_trial_baseline(ppg_data, baseline_samples)

            organized[subject_id] = {
                "subject_id": subject_id,
                "eeg": list(eeg_data),   # list of (32, n_samples) per trial
                "eda": list(eda_data),   # list of (1, n_samples) per trial
                "ppg": list(ppg_data),   # list of (1, n_samples) per trial
                "labels": subject["labels"],
                "binary_valence": subject["binary_valence"],
                "binary_arousal": subject["binary_arousal"],
            }

        return organized

    # ------------------------------------------------------------
    # Step 2: signal-level preprocessing (per subject)
    # ------------------------------------------------------------

    def _preprocess_subject(self, subject_data: dict, stats: dict | None) -> dict:

        eeg_mean, eeg_std = stats["eeg"] if stats else (None, None)
        eda_mean, eda_std = stats["eda"] if stats else (None, None)
        ppg_mean, ppg_std = stats["ppg"] if stats else (None, None)

        processed_eeg = [
            self.eeg_pre.preprocess(trial, subject_mean=eeg_mean, subject_std=eeg_std)
            for trial in subject_data["eeg"]
        ]
        processed_eda = [
            self.eda_pre.preprocess(trial, subject_mean=eda_mean, subject_std=eda_std)
            for trial in subject_data["eda"]
        ]
        processed_ppg = [
            self.ppg_pre.preprocess(trial, subject_mean=ppg_mean, subject_std=ppg_std)
            for trial in subject_data["ppg"]
        ]

        return {
            "subject_id": subject_data["subject_id"],
            "eeg": processed_eeg,
            "eda": processed_eda,
            "ppg": processed_ppg,
            "labels": subject_data["labels"],
            "binary_valence": subject_data["binary_valence"],
            "binary_arousal": subject_data["binary_arousal"],
        }

    # ------------------------------------------------------------
    # Step 3: windowing + feature extraction -> flat list of windows
    # ------------------------------------------------------------

    def _windows_with_features(self, processed_subjects: Dict[str, dict]) -> List[dict]:

        all_windows = []

        for subject_id, subject in processed_subjects.items():

            windows = self.segmenter.segment_subject(subject)

            arousal_labels = subject["binary_arousal"]

            for w in windows:

                w["label_valence"] = w["label"]
                w["label_arousal"] = int(arousal_labels[w["trial_id"]])

                w["eeg_features"] = self.eeg_feat.extract(w["eeg"])
                w["eda_features"] = self.eda_feat.extract(w["eda"])
                w["ppg_features"] = self.ppg_feat.extract(w["ppg"])

                # Δεν χρειαζόμαστε πλέον τα raw σήματα (μόνο τα features
                # τροφοδοτούν το LSTM) -- ελευθερώνουμε μνήμη νωρίς.
                w["eeg"] = None
                w["eda"] = None
                w["ppg"] = None

                all_windows.append(w)

        return all_windows

    # ------------------------------------------------------------
    # Step 4: fit/transform feature scalers (fit on train only)
    # ------------------------------------------------------------

    def _fit_transform_scaler(self, key: str, scaler: FeatureScaler,
                               train_windows, val_windows, test_windows):

        train_matrix = np.stack([w[key] for w in train_windows])
        train_matrix = np.nan_to_num(train_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        scaler.fit(train_matrix)

        for windows in (train_windows, val_windows, test_windows):

            if len(windows) == 0:
                continue

            matrix = np.stack([w[key] for w in windows])
            matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
            scaled = scaler.transform(matrix).astype(np.float32)

            for w, row in zip(windows, scaled):
                w[key] = row

    # ------------------------------------------------------------
    # Step 5: group windows into per-trial sequences
    # ------------------------------------------------------------

    @staticmethod
    def _windows_to_sequences(windows: List[dict]) -> Dict[str, np.ndarray]:

        grouped: Dict[tuple, List[dict]] = {}

        for w in windows:
            key = (str(w["subject_id"]), int(w["trial_id"]))
            grouped.setdefault(key, []).append(w)

        eeg_seqs, eda_seqs, ppg_seqs = [], [], []
        valence_labels, arousal_labels = [], []
        subjects, trials = [], []

        expected_len = None

        for (subject_id, trial_id), items in sorted(grouped.items()):

            items = sorted(items, key=lambda x: x["window_id"])

            if expected_len is None:
                expected_len = len(items)

            if len(items) != expected_len:
                # Ασυνεπές trial (π.χ. λόγω μελλοντικού label filtering) --
                # παραλείπεται αντί να σπάσει το σταθερό-μήκους tensor.
                logger.warning(
                    f"Skipping subject={subject_id} trial={trial_id}: "
                    f"{len(items)} windows (expected {expected_len})."
                )
                continue

            eeg_seqs.append(np.stack([it["eeg_features"] for it in items]))
            eda_seqs.append(np.stack([it["eda_features"] for it in items]))
            ppg_seqs.append(np.stack([it["ppg_features"] for it in items]))

            valence_labels.append(items[0]["label_valence"])
            arousal_labels.append(items[0]["label_arousal"])
            subjects.append(subject_id)
            trials.append(trial_id)

        return {
            "eeg": np.stack(eeg_seqs).astype(np.float32) if eeg_seqs else np.empty((0,)),
            "eda": np.stack(eda_seqs).astype(np.float32) if eda_seqs else np.empty((0,)),
            "ppg": np.stack(ppg_seqs).astype(np.float32) if ppg_seqs else np.empty((0,)),
            "valence": np.asarray(valence_labels, dtype=np.int64),
            "arousal": np.asarray(arousal_labels, dtype=np.int64),
            "subjects": np.asarray(subjects),
            "trials": np.asarray(trials, dtype=np.int64),
        }

    # ------------------------------------------------------------
    # Caching helpers -- επιτρέπει να τρέξουμε το (αργό, CPU-bound)
    # preprocessing/feature-extraction σε ένα μηχάνημα και να
    # τρέξουμε το training (GPU-bound) σε άλλο, χωρίς να ξαναγίνει
    # όλη η προεπεξεργασία στο δεύτερο μηχάνημα.
    # ------------------------------------------------------------

    @staticmethod
    def save_cache(data: Dict[str, Dict[str, np.ndarray]], cache_path: str) -> None:

        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)

        payload = {}
        for split in ("train", "validation", "test"):
            for key, value in data[split].items():
                payload[f"{split}__{key}"] = value

        payload["train_subject_ids"] = np.asarray(data["train_subject_ids"])

        np.savez_compressed(cache_path, **payload)
        logger.info(f"Saved pipeline cache -> {cache_path}")

    @staticmethod
    def load_cache(cache_path: str) -> Dict[str, Dict[str, np.ndarray]]:

        npz = np.load(cache_path, allow_pickle=False)

        data: Dict[str, Dict[str, np.ndarray]] = {
            "train": {}, "validation": {}, "test": {}
        }

        for full_key in npz.files:
            if full_key == "train_subject_ids":
                continue
            split, key = full_key.split("__", 1)
            data[split][key] = npz[full_key]

        data["train_subject_ids"] = [str(s) for s in npz["train_subject_ids"]]

        logger.info(f"Loaded pipeline cache <- {cache_path}")

        return data

    # ------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------

    def run(self, max_subjects: int | None = None,
            cache_path: str | None = None) -> Dict[str, Dict[str, np.ndarray]]:

        if cache_path is not None and os.path.exists(cache_path):
            return self.load_cache(cache_path)

        logger.info("=" * 60)
        logger.info("Paper-Reproduction LSTM Data Pipeline "
                     "(Saffaryazdi et al. 2022)")
        logger.info("=" * 60)

        logger.info("Loading + organizing DEAP subjects...")
        organized = self._load_and_organize(max_subjects=max_subjects)
        logger.info(f"Loaded {len(organized)} subjects.")

        subject_keys = list(organized.keys())
        splits = self.splitter.train_val_test_split(
            subject_ids=subject_keys,
            train_ratio=self.config.TRAIN_RATIO,
            val_ratio=self.config.VAL_RATIO,
            test_ratio=self.config.TEST_RATIO,
        )

        logger.info(
            f"Train subjects: {len(splits['train'])} | "
            f"Val subjects: {len(splits['validation'])} | "
            f"Test subjects: {len(splits['test'])}"
        )

        train_subjects = {k: organized[k] for k in splits["train"]}
        val_subjects = {k: organized[k] for k in splits["validation"]}
        test_subjects = {k: organized[k] for k in splits["test"]}

        # The default protocol fits normalization statistics on train
        # subjects and normalizes unseen subjects per trial. The optional
        # subject-aware experiment estimates unlabeled signal statistics
        # separately for each subject before feature extraction.
        stats = {}
        subjects_for_stats = (
            organized if self.subject_aware_normalization else train_subjects
        )
        for sid, data in subjects_for_stats.items():
            stats[sid] = {
                "eeg": SubjectStatistics.compute(data["eeg"]),
                "eda": SubjectStatistics.compute(data["eda"]),
                "ppg": SubjectStatistics.compute(data["ppg"]),
            }

        logger.info("Preprocessing signals (EEG/EDA/PPG)...")

        processed_train = {sid: self._preprocess_subject(d, stats.get(sid))
                            for sid, d in train_subjects.items()}
        processed_val = {sid: self._preprocess_subject(
                             d, stats.get(sid) if self.subject_aware_normalization else None)
                           for sid, d in val_subjects.items()}
        processed_test = {sid: self._preprocess_subject(
                              d, stats.get(sid) if self.subject_aware_normalization else None)
                           for sid, d in test_subjects.items()}

        logger.info("Windowing (1s, no overlap) + feature extraction...")

        train_windows = self._windows_with_features(processed_train)
        val_windows = self._windows_with_features(processed_val)
        test_windows = self._windows_with_features(processed_test)

        logger.info(
            f"Windows -> train={len(train_windows)} "
            f"val={len(val_windows)} test={len(test_windows)}"
        )

        logger.info("Fitting feature scalers on train windows...")

        self._fit_transform_scaler("eeg_features", self.eeg_scaler,
                                    train_windows, val_windows, test_windows)
        self._fit_transform_scaler("eda_features", self.eda_scaler,
                                    train_windows, val_windows, test_windows)
        self._fit_transform_scaler("ppg_features", self.ppg_scaler,
                                    train_windows, val_windows, test_windows)

        logger.info("Grouping windows into per-trial sequences...")

        train_arrays = self._windows_to_sequences(train_windows)
        val_arrays = self._windows_to_sequences(val_windows)
        test_arrays = self._windows_to_sequences(test_windows)

        logger.info(
            f"Trial sequences -> train={len(train_arrays['valence'])} "
            f"val={len(val_arrays['valence'])} test={len(test_arrays['valence'])} "
            f"(seq_len={train_arrays['eeg'].shape[1] if len(train_arrays['valence']) else 'NA'})"
        )

        result = {
            "train": train_arrays,
            "validation": val_arrays,
            "test": test_arrays,
            "train_subject_ids": sorted(train_subjects.keys()),
        }

        if cache_path is not None:
            self.save_cache(result, cache_path)

        return result
