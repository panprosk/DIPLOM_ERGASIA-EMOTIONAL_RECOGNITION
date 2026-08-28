from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """
    Global configuration of the thesis.
    """

    # ==========================================================
    # PROJECT
    # ==========================================================

    PROJECT_NAME: str = "Cross-Subject Emotion Recognition"

    RANDOM_SEED: int = 42

    SEED: int = 42  # Alias για συμβατότητα

    DEVICE: str = "cuda"

    # ==========================================================
    # DATASET
    # ==========================================================

    ROOT_DIR: Path = Path(__file__).resolve().parents[2]

    DATASET_DIR: Path = ROOT_DIR / "data" / "deap"

    DEAP_PATH: Path = ROOT_DIR / "data" / "deap"

    DEAP_DIR: Path = ROOT_DIR / "data" / "deap"  # Alias για συμβατότητα

    OUTPUT_DIR: Path = ROOT_DIR / "outputs"

    LOG_DIR: Path = OUTPUT_DIR / "logs"

    EXPERIMENT_DIR: Path = OUTPUT_DIR / "experiments"

    # ==========================================================
    # SIGNALS
    # ==========================================================

    SAMPLING_RATE: int = 128

    EEG_CHANNELS: int = 32

    EDA_CHANNELS: int = 1

    PPG_CHANNELS: int = 1

    # ==========================================================
    # LABELS
    # ==========================================================

    VALENCE_THRESHOLD: float = 5.0

    NUM_CLASSES: int = 2

    # ==========================================================
    # MODEL
    # ==========================================================

    MODEL_NAME: str = "hybrid"

    EMBEDDING_DIM: int = 128

    PHYSIO_FEATURE_DIM: int = 22  # EDA (12) + PPG (10) handcrafted features

    # Διάσταση των handcrafted EEG χαρακτηριστικών (band powers,
    # στατιστικά ροπών κ.λπ., 13 features x 32 κανάλια). Χρησιμοποιείται
    # για να εμπλουτίσει το raw-EEG-CNN embedding μέσα στο EEG branch
    # (βλ. EEGHandcraftedEncoder) — βοηθά τη γενίκευση καθώς αυτά τα
    # χαρακτηριστικά είναι πιο σταθερά cross-subject από το raw waveform.
    EEG_HANDCRAFTED_DIM: int = 416

    # Μέγιστη ισχύς (lambda) του Domain-Adversarial subject classifier
    # (Gradient Reversal Layer). Χρησιμοποιείται με γραμμικό ramp-up
    # στα πρώτα epochs (DANN-style schedule) ώστε ο encoder να μην
    # "μπερδεύεται" υπερβολικά νωρίς στην εκπαίδευση.
    ADVERSARIAL_LAMBDA_MAX: float = 0.30

    # ----------------------------------------------------------------
    # Cross-Attention Transformer (2ο μοντέλο, βλ. ΣΕΙΡΑ_ΥΛΟΠΟΙΗΣΗΣ
    # Βήμα 12) -- υπερπαράμετροι σκόπιμα μικρές ώστε το training να
    # παραμένει εφικτό σε CPU-only μηχάνημα με μόνο 8GB RAM.
    # ----------------------------------------------------------------
    TRANSFORMER_D_MODEL: int = 64

    TRANSFORMER_NUM_HEADS: int = 4

    TRANSFORMER_SELF_ATTN_LAYERS: int = 2

    TRANSFORMER_CROSS_ATTN_LAYERS: int = 1

    TRANSFORMER_FF_DIM: int = 128

    TRANSFORMER_PATCH_SIZE: int = 32  # 768 samples / 32 = 24 patches ανά modality

    # ==========================================================
    # WINDOWING
    # ==========================================================

    WINDOW_SIZE_SEC: int = 6

    WINDOW_SIZE: int = WINDOW_SIZE_SEC * SAMPLING_RATE

    OVERLAP: float = 0.75

    STEP_SIZE: int = int(WINDOW_SIZE * (1 - OVERLAP))

    # ==========================================================
    # PREPROCESSING
    # ==========================================================

    EEG_LOWCUT: float = 4.0

    EEG_HIGHCUT: float = 45.0

    EEG_NOTCH: bool = False

    EDA_LOWPASS: float = 1.0

    EDA_GAUSSIAN_SIGMA: float = 1.0

    SUBJECT_WISE_NORMALIZATION: bool = True

    # ==========================================================
    # FEATURES
    # ==========================================================

    USE_HANDCRAFTED_FEATURES: bool = True

    USE_NEUROKIT_FEATURES: bool = True

    USE_SCIPY_FEATURES: bool = True

    # ==========================================================
    # DATA SPLIT
    # ==========================================================

    TRAIN_RATIO: float = 0.70

    VALIDATION_RATIO: float = 0.15

    VAL_RATIO: float = 0.15  # Alias για συμβατότητα

    TEST_RATIO: float = 0.15

    USE_LOSO: bool = False

    # ==========================================================
    # DATALOADER
    # ==========================================================

    BATCH_SIZE: int = 64

    # NOTE: Τα δεδομένα είναι ήδη πλήρως φορτωμένα στη μνήμη (in-memory
    # NumPy arrays/tensors), οπότε δεν υπάρχει disk I/O ανά sample να
    # παραλληλοποιηθεί. Σε Windows, num_workers>0 χρησιμοποιεί spawn και
    # κάνει pickle ολόκληρο το dataset object για να το στείλει σε κάθε
    # worker process· με το πλήρες 32-subject dataset αυτό ξεπερνά το
    # όριο μεγέθους του Windows anonymous pipe και προκαλεί
    # "OSError: [Errno 22] Invalid argument" / MemoryError. Γι' αυτό
    # 0 (single-process loading) είναι η ασφαλής προεπιλογή εδώ.
    NUM_WORKERS: int = 0

    PIN_MEMORY: bool = True

    DROP_LAST: bool = False

    WEIGHTED_SAMPLING: bool = True

    # ==========================================================
    # TRAINING
    # ==========================================================

    EPOCHS: int = 100

    LEARNING_RATE: float = 1e-3

    # Αυξήθηκε από 1e-4 -> 5e-4: παρατηρήθηκε έντονο overfitting
    # (Train Acc 93% έναντι Val/Test Acc ~50%) στο πλήρες 32-subject
    # dataset. Ισχυρότερο L2 regularization βοηθά να περιοριστεί.
    WEIGHT_DECAY: float = 5e-4

    DROPOUT: float = 0.30

    EARLY_STOPPING_PATIENCE: int = 15

    # ==========================================================
    # METRICS
    # ==========================================================

    PRIMARY_METRIC: str = "f1"

    COMPUTE_CONFIDENCE: bool = True

    COMPUTE_ENTROPY: bool = True

    COMPUTE_SUBJECT_METRICS: bool = True

    # ==========================================================
    # EXPERIMENTS
    # ==========================================================

    SAVE_CHECKPOINTS: bool = True

    SAVE_PREDICTIONS: bool = True

    SAVE_FEATURES: bool = False

    SAVE_LOGS: bool = True

    # ==========================================================
    # SHG AGENT
    # ==========================================================

    ENABLE_AGENT: bool = False

    MAX_AGENT_EXPERIMENTS: int = 500

    GENERALIZATION_WEIGHT: float = 0.40

    WORST_SUBJECT_WEIGHT: float = 0.20

    CONFIDENCE_WEIGHT: float = 0.20

    ENTROPY_WEIGHT: float = 0.10

    VARIANCE_WEIGHT: float = 0.10

    # ==========================================================
    # DIRECTORIES
    # ==========================================================

    def create_directories(self):
        """
        Creates all required output directories.
        """

        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

        self.EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    # ==========================================================
    # SUMMARY
    # ==========================================================

    def summary(self):
        """
        Prints a short summary of the current configuration.
        """

        print("=" * 60)
        print("PROJECT CONFIGURATION")
        print("=" * 60)

        print(f"Dataset      : {self.DATASET_DIR}")
        print(f"SamplingRate : {self.SAMPLING_RATE}")
        print(f"Window       : {self.WINDOW_SIZE_SEC} sec")
        print(f"Overlap      : {self.OVERLAP}")
        print(f"Batch Size   : {self.BATCH_SIZE}")
        print(f"Epochs       : {self.EPOCHS}")
        print(f"LearningRate : {self.LEARNING_RATE}")
        print(f"Device       : {self.DEVICE}")
        print(f"Agent Enabled: {self.ENABLE_AGENT}")

        print("=" * 60)