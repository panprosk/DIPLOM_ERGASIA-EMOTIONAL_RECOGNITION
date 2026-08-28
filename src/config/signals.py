"""
Signal configuration για το DEAP dataset.
"""

# -----------------------------
# EEG
# -----------------------------

EEG_CHANNELS = list(range(32))

# -----------------------------
# Peripheral Signals
# -----------------------------

EDA_CHANNEL = 36
PPG_CHANNEL = 38

# -----------------------------
# Συνολικά channels που θα
# χρησιμοποιηθούν στη διπλωματική
# -----------------------------

SELECTED_CHANNELS = EEG_CHANNELS + [
    EDA_CHANNEL,
    PPG_CHANNEL
]

# αριθμός σημάτων

N_EEG = len(EEG_CHANNELS)

N_SIGNALS = len(SELECTED_CHANNELS)