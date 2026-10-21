"""Paper-faithful DEAP preprocessing for Saffaryazdi et al. (2022).

This module is intentionally separate from the existing experimental Paper
LSTM pipeline.  It removes the first eight seconds, creates 55 non-overlapping
one-second windows, and extracts the feature dimensions described in the
paper: 160 EEG band-power features and six physiological features.
"""

from __future__ import annotations

import logging
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, medfilt, sosfiltfilt

from src.config.config import Config
from src.data.deap_loader import DEAPLoader
from src.split.subject_split import SubjectSplitter

logger = logging.getLogger(__name__)


class SaffaryazdiPipeline:
    """Build and cache subject-independent per-trial feature sequences."""

    FS = 128
    DROP_SECONDS = 8
    WINDOW_SAMPLES = 128
    EDA_MEDIAN_KERNEL = 5
    EEG_BANDS = ((1.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 30.0), (30.0, 45.0))
    EEG_BANDS_NO_DELTA = ((4.0, 8.0), (8.0, 12.0), (12.0, 30.0), (30.0, 45.0))

    def __init__(
        self,
        config: Config,
        seed: int = 42,
        feature_mode: str = "absolute",
        eeg_normalization: str = "global",
        phys_mode: str = "all",
        modality: str = "all",
        window_seconds: float = 1.0,
        eeg_band_mode: str = "five",
        eeg_filter: str = "none",
        drop_seconds: int = 8,
        filter_before_trim: bool = False,
    ):
        if feature_mode not in {"absolute", "log", "absolute_relative", "hann_absolute"}:
            raise ValueError(f"Unsupported EEG feature mode: {feature_mode}")
        if eeg_normalization not in {"global", "subject", "feature_global"}:
            raise ValueError(
                f"Unsupported EEG normalization mode: {eeg_normalization}"
            )
        if phys_mode not in {"all", "eda", "ppg"}:
            raise ValueError(f"Unsupported physiology mode: {phys_mode}")
        if modality not in {"all", "eeg", "phys"}:
            raise ValueError(f"Unsupported modality mode: {modality}")
        if eeg_band_mode not in {"five", "four"}:
            raise ValueError(f"Unsupported EEG band mode: {eeg_band_mode}")
        if eeg_filter not in {"none", "1-45"}:
            raise ValueError(f"Unsupported EEG filter: {eeg_filter}")
        if not 0 <= drop_seconds < 63:
            raise ValueError("drop_seconds must be in [0, 63).")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive.")
        self.config = config
        self.seed = seed
        self.feature_mode = feature_mode
        self.eeg_normalization = eeg_normalization
        self.phys_mode = phys_mode
        self.modality = modality
        self.window_seconds = window_seconds
        self.eeg_band_mode = eeg_band_mode
        self.eeg_filter = eeg_filter
        self.drop_seconds = drop_seconds
        self.filter_before_trim = filter_before_trim
        self.eeg_bands = (
            self.EEG_BANDS if eeg_band_mode == "five" else self.EEG_BANDS_NO_DELTA
        )
        self.window_samples = int(round(window_seconds * self.FS))
        if self.window_samples > (63 - self.drop_seconds) * self.FS:
            raise ValueError("window_seconds exceeds the post-drop trial duration.")
        self.loader = DEAPLoader(dataset_path=config.DEAP_PATH)

    @staticmethod
    def _filter(signal: np.ndarray, low: float, high: float) -> np.ndarray:
        sos = butter(4, [low, high], btype="bandpass", fs=128, output="sos")
        return sosfiltfilt(sos, signal, axis=-1).astype(np.float32)

    @staticmethod
    def _fit_zscore(train_values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if train_values[0].ndim == 2:
            stacked = np.concatenate(train_values, axis=1)
            return stacked.mean(axis=1, keepdims=True), stacked.std(axis=1, keepdims=True)
        stacked = np.concatenate([x.reshape(-1) for x in train_values])
        return np.asarray([stacked.mean()]), np.asarray([stacked.std()])

    @staticmethod
    def _apply_zscore(
        values: np.ndarray, mean: np.ndarray, std: np.ndarray
    ) -> np.ndarray:
        return ((values - mean) / (std + 1e-6)).astype(np.float32)

    @classmethod
    def _eeg_features(
        cls,
        window: np.ndarray,
        feature_mode: str = "absolute",
        bands: tuple[tuple[float, float], ...] | None = None,
    ) -> np.ndarray:
        if feature_mode == "hann_absolute":
            window = window * np.hanning(window.shape[-1])[None, :]
        spectrum = np.fft.rfft(window, axis=-1)
        power = (np.abs(spectrum) ** 2) / window.shape[-1]
        freqs = np.fft.rfftfreq(window.shape[-1], d=1.0 / cls.FS)
        features = []
        for low, high in bands or cls.EEG_BANDS:
            mask = (freqs >= low) & (freqs < high)
            features.append(power[:, mask].mean(axis=-1))
        band_power = np.stack(features, axis=-1)
        if feature_mode == "log":
            band_power = np.concatenate(
                [band_power, np.log1p(np.maximum(band_power, 0.0))], axis=-1
            )
        elif feature_mode == "absolute_relative":
            total_power = band_power.sum(axis=-1, keepdims=True)
            relative_power = band_power / (total_power + 1e-6)
            band_power = np.concatenate([band_power, relative_power], axis=-1)
        return band_power.reshape(-1).astype(np.float32)

    @staticmethod
    def _eda_features(window: np.ndarray) -> np.ndarray:
        # Median filtering is applied once to the complete EDA trial before
        # normalization and windowing in _preprocess_eda.
        diff1 = np.diff(window)
        diff2 = np.diff(diff1)
        return np.asarray(
            [
                window.mean(),
                window.std(),
                np.mean(np.abs(diff1)),
                np.mean(np.abs(diff2)),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _ppg_features(window: np.ndarray) -> np.ndarray:
        return np.asarray([window.mean(), window.std()], dtype=np.float32)

    @classmethod
    def _preprocess_eda(cls, signal: np.ndarray) -> np.ndarray:
        filtered = cls._filter(signal[None, :], 0.1, 15.0)[0]
        return medfilt(filtered, kernel_size=cls.EDA_MEDIAN_KERNEL).astype(np.float32)

    @classmethod
    def _trial_features(
        cls,
        eeg: np.ndarray,
        eda: np.ndarray,
        ppg: np.ndarray,
        feature_mode: str = "absolute",
        phys_mode: str = "all",
        window_samples: int = 128,
        bands: tuple[tuple[float, float], ...] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_windows = eeg.shape[-1] // window_samples
        eeg_seq, phys_seq = [], []
        for index in range(n_windows):
            start = index * window_samples
            stop = start + window_samples
            eeg_seq.append(
                cls._eeg_features(eeg[:, start:stop], feature_mode, bands)
            )
            eda_features = cls._eda_features(eda[start:stop])
            ppg_features = cls._ppg_features(ppg[start:stop])
            if phys_mode == "eda":
                phys_seq.append(eda_features)
            elif phys_mode == "ppg":
                phys_seq.append(ppg_features)
            else:
                phys_seq.append(np.concatenate([eda_features, ppg_features]))
        return np.stack(eeg_seq), np.stack(phys_seq)

    def _load_subjects(self) -> dict[str, dict]:
        cached = getattr(self, "_subjects_cache", None)
        if cached is not None:
            return cached
        subjects = self.loader.load_all_subjects()
        result = {}
        for subject_id, subject in subjects.items():
            if subject["data"].shape != (40, 40, 8064):
                raise ValueError(
                    f"Unexpected DEAP shape for {subject_id}: "
                    f"{subject['data'].shape}; expected (40, 40, 8064)."
                )
            raw_data = subject["data"]
            if self.filter_before_trim:
                eda = np.stack(
                    [
                        self._preprocess_eda(raw_data[trial, 36])
                        for trial in range(raw_data.shape[0])
                    ]
                )
                ppg = np.stack(
                    [
                        self._filter(
                            raw_data[trial, 38][None, :], 0.7, 2.5
                        )[0]
                        for trial in range(raw_data.shape[0])
                    ]
                )
            else:
                eda = raw_data[:, 36, :]
                ppg = raw_data[:, 38, :]
            data = raw_data[:, :, self.drop_seconds * self.FS :]
            eeg = data[:, :32, :]
            if self.eeg_filter == "1-45":
                eeg = np.stack(
                    [self._filter(trial, 1.0, 45.0) for trial in eeg]
                )
            labels = subject["labels"]
            result[subject_id] = {
                "eeg": eeg,
                "eda": eda[:, self.drop_seconds * self.FS :],
                "ppg": ppg[:, self.drop_seconds * self.FS :],
                "valence": (labels[:, 0] >= 5.0).astype(np.int64),
                "arousal": (labels[:, 1] >= 5.0).astype(np.int64),
            }
        self._subjects_cache = result
        return result

    def _build_arrays(self, subjects: dict[str, dict], train_ids: list[str]) -> dict:
        train_eeg, train_eda, train_ppg = [], [], []
        for sid in train_ids:
            item = subjects[sid]
            for trial in range(40):
                train_eeg.append(item["eeg"][trial])
                train_eda.append(
                    item["eda"][trial]
                    if self.filter_before_trim
                    else self._preprocess_eda(item["eda"][trial])
                )
                train_ppg.append(
                    item["ppg"][trial]
                    if self.filter_before_trim
                    else self._filter(item["ppg"][trial][None, :], 0.7, 2.5)[0]
                )
        eeg_mean, eeg_std = self._fit_zscore(train_eeg)
        eda_mean, eda_std = self._fit_zscore(train_eda)
        ppg_mean, ppg_std = self._fit_zscore(train_ppg)
        subject_eeg_stats = {}
        if self.eeg_normalization == "subject":
            for sid in train_ids:
                item = subjects[sid]
                subject_eeg_stats[sid] = self._fit_zscore(
                    [item["eeg"][trial] for trial in range(40)]
                )

        split_arrays = {}
        for split_name, subject_ids in self.splits.items():
            eeg_sequences, phys_sequences = [], []
            valence, arousal, subject_values, trial_values = [], [], [], []
            for sid in subject_ids:
                item = subjects[sid]
                for trial in range(40):
                    # DEAP EEG is already preprocessed by the dataset provider;
                    # the paper does not apply a second band-pass filter.
                    eeg = item["eeg"][trial]
                    eda = (
                        item["eda"][trial]
                        if self.filter_before_trim
                        else self._preprocess_eda(item["eda"][trial])
                    )
                    ppg = (
                        item["ppg"][trial]
                        if self.filter_before_trim
                        else self._filter(item["ppg"][trial][None, :], 0.7, 2.5)[0]
                    )
                    if self.eeg_normalization == "feature_global":
                        eeg_for_features = eeg
                    elif self.eeg_normalization == "subject":
                        if sid in subject_eeg_stats:
                            subject_mean, subject_std = subject_eeg_stats[sid]
                            eeg_for_features = self._apply_zscore(
                                eeg, subject_mean, subject_std
                            )
                        else:
                            # Strict inductive protocol: unseen subjects use
                            # only global statistics fitted on training subjects.
                            eeg_for_features = self._apply_zscore(
                                eeg, eeg_mean, eeg_std
                            )
                    else:
                        eeg_for_features = self._apply_zscore(
                            eeg, eeg_mean, eeg_std
                        )
                    eda = self._apply_zscore(eda, eda_mean, eda_std)
                    ppg = self._apply_zscore(ppg, ppg_mean, ppg_std)
                    eeg_seq, phys_seq = self._trial_features(
                        eeg_for_features, eda, ppg, self.feature_mode, self.phys_mode
                        , self.window_samples, self.eeg_bands
                    )
                    if self.modality == "eeg":
                        phys_seq = np.zeros_like(phys_seq)
                    elif self.modality == "phys":
                        eeg_seq = np.zeros_like(eeg_seq)
                    eeg_sequences.append(eeg_seq)
                    phys_sequences.append(phys_seq)
                    valence.append(item["valence"][trial])
                    arousal.append(item["arousal"][trial])
                    subject_values.append(sid)
                    trial_values.append(trial)
            split_arrays[split_name] = {
                "eeg": np.asarray(eeg_sequences, dtype=np.float32),
                "phys": np.asarray(phys_sequences, dtype=np.float32),
                "valence": np.asarray(valence, dtype=np.int64),
                "arousal": np.asarray(arousal, dtype=np.int64),
                "subjects": np.asarray(subject_values),
                "trials": np.asarray(trial_values, dtype=np.int64),
            }
        if self.eeg_normalization == "feature_global":
            feature_mean = split_arrays["train"]["eeg"].mean(axis=(0, 1), keepdims=True)
            feature_std = split_arrays["train"]["eeg"].std(axis=(0, 1), keepdims=True)
            for values in split_arrays.values():
                values["eeg"] = (
                    (values["eeg"] - feature_mean) / (feature_std + 1e-6)
                ).astype(np.float32)
        return split_arrays

    def subject_ids(self) -> list[str]:
        """Return the sorted list of available subject identifiers."""
        return sorted(self._load_subjects())

    def run(
        self,
        cache_path: str | None = None,
        max_subjects: int | None = None,
        splits: dict | None = None,
    ) -> dict:
        subjects = self._load_subjects()
        if max_subjects is not None:
            subjects = dict(list(subjects.items())[:max_subjects])
        subject_ids = sorted(subjects)
        if splits is None:
            split = SubjectSplitter(random_seed=self.seed).train_val_test_split(
                subject_ids, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
            )
        else:
            split = {name: list(ids) for name, ids in splits.items()}
        self.splits = split
        metadata = {
            "cache_version": 1,
            "feature_mode": self.feature_mode,
            "eeg_normalization": self.eeg_normalization,
            "phys_mode": self.phys_mode,
            "modality": self.modality,
            "window_seconds": self.window_seconds,
            "drop_seconds": self.drop_seconds,
            "eeg_band_mode": self.eeg_band_mode,
            "eeg_filter": self.eeg_filter,
            "filter_before_trim": self.filter_before_trim,
            "train_subjects": sorted(map(str, split["train"])),
            "validation_subjects": sorted(map(str, split["validation"])),
            "test_subjects": sorted(map(str, split["test"])),
            "max_subjects": max_subjects,
        }
        if cache_path and Path(cache_path).is_file():
            with np.load(cache_path, allow_pickle=False) as cached:
                cached_metadata = (
                    json.loads(str(cached["metadata"]))
                    if "metadata" in cached
                    else None
                )
                if cached_metadata == metadata:
                    logger.info("Loaded Saffaryazdi cache <- %s", cache_path)
                    return {
                        split_name: {
                            key: cached[f"{split_name}__{key}"]
                            for key in (
                                "eeg",
                                "phys",
                                "valence",
                                "arousal",
                                "subjects",
                                "trials",
                            )
                        }
                        for split_name in ("train", "validation", "test")
                    }
            logger.warning(
                "Ignoring incompatible Saffaryazdi cache; rebuilding <- %s",
                cache_path,
            )
        arrays = self._build_arrays(subjects, split["train"])
        train_subjects = set(arrays["train"]["subjects"].astype(str))
        validation_subjects = set(arrays["validation"]["subjects"].astype(str))
        test_subjects = set(arrays["test"]["subjects"].astype(str))
        if (
            train_subjects & validation_subjects
            or train_subjects & test_subjects
            or validation_subjects & test_subjects
        ):
            raise RuntimeError("Subject leakage detected in Saffaryazdi split.")
        logger.info(
            "Saffaryazdi sequences: train=%d validation=%d test=%d; shape=%s",
            len(arrays["train"]["valence"]),
            len(arrays["validation"]["valence"]),
            len(arrays["test"]["valence"]),
            arrays["train"]["eeg"].shape[1:],
        )
        if cache_path:
            payload = {
                f"{split_name}__{key}": value
                for split_name, values in arrays.items()
                for key, value in values.items()
            }
            payload["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, **payload)
            logger.info("Saved Saffaryazdi cache -> %s", cache_path)
        return arrays
