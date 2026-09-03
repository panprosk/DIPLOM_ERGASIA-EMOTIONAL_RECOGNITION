import numpy as np


def remove_trial_baseline(
    trials: np.ndarray,
    baseline_samples: int,
) -> np.ndarray:
    """
    Per-trial pre-stimulus baseline correction.

    Κάθε DEAP trial περιέχει τα πρώτα `baseline_samples` δείγματα ως
    "ήρεμη" (pre-stimulus) καταγραφή, πριν αρχίσει το video-stimulus.
    Αυτό το τμήμα δεν αντιπροσωπεύει το συναίσθημα-label του trial,
    αλλά χρησιμοποιείται καθιερωμένα στη βιβλιογραφία (DEAP-based
    emotion recognition) ως baseline reference: αφαιρείται ο μέσος όρος
    του ανά (trial, channel) από το υπόλοιπο σήμα του trial, ώστε να
    αφαιρεθεί η ατομική/per-trial στάθμη (π.χ. διαφορετικό skin
    conductance level ή EEG offset ανά υποκείμενο/trial) πριν προστεθεί
    οποιοδήποτε άλλο normalization. Αυτό μειώνει σημαντικά τον θόρυβο
    που προέρχεται από διαφορές μεταξύ subjects/trials, βελτιώνοντας
    τη γενίκευση (cross-subject generalization).

    Το baseline τμήμα αφαιρείται εντελώς από την έξοδο -- δεν έχει
    νόημα να παραμείνει windowed/labeled ως μέρος του trial, αφού δεν
    αντιστοιχεί στο συναισθηματικό stimulus.

    Parameters
    ----------
    trials : np.ndarray, shape (n_trials, n_channels, n_samples)
    baseline_samples : int, αριθμός δειγμάτων του pre-stimulus τμήματος.

    Returns
    -------
    np.ndarray, shape (n_trials, n_channels, n_samples - baseline_samples)
    """

    if trials.ndim != 3:
        raise ValueError(
            f"Expected 3D array (trials, channels, samples), got shape {trials.shape}"
        )

    if baseline_samples <= 0 or baseline_samples >= trials.shape[-1]:
        # Τίποτα να αφαιρεθεί / άκυρη τιμή -- επιστρέφουμε ως έχει.
        return trials

    baseline = trials[:, :, :baseline_samples]

    baseline_mean = baseline.mean(axis=-1, keepdims=True)

    stimulus = trials[:, :, baseline_samples:]

    corrected = stimulus - baseline_mean

    return corrected.astype(trials.dtype)
