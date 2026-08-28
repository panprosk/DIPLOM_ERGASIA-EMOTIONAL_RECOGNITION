"""Evaluation helpers for window-, trial-, and subject-level reporting."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def _classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    """Binary metrics with a fixed label order for comparable Macro-F1."""
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(
            f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0)
        ),
    }


def evaluate_hierarchical_predictions(
    probabilities: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    trials: np.ndarray,
) -> dict:
    """
    Computes evaluation metrics at three levels.

    Window-level scores treat each overlapping window as an observation.
    Trial-level scores first average the class probabilities of all windows
    belonging to the same ``(subject, trial)``. Subject-level scores are then
    calculated from those trial predictions, so a trial with many overlapping
    windows cannot dominate the reported subject performance.
    """
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels, dtype=np.int64)
    subjects = np.asarray(subjects).astype(str)
    trials = np.asarray(trials, dtype=np.int64)

    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("probabilities must have shape (n_samples, 2).")
    if not (len(probabilities) == len(labels) == len(subjects) == len(trials)):
        raise ValueError("probabilities, labels, subjects, and trials must have equal length.")

    window_predictions = probabilities.argmax(axis=1)
    window_metrics = _classification_metrics(labels, window_predictions)

    grouped = defaultdict(list)
    for idx, key in enumerate(zip(subjects, trials)):
        grouped[key].append(idx)

    trial_probabilities = []
    trial_labels = []
    trial_subjects = []
    trial_ids = []

    for (subject, trial), indices in sorted(grouped.items()):
        group_labels = labels[indices]
        if not np.all(group_labels == group_labels[0]):
            raise ValueError(f"Inconsistent labels in subject={subject}, trial={trial}.")

        trial_probabilities.append(probabilities[indices].mean(axis=0))
        trial_labels.append(group_labels[0])
        trial_subjects.append(subject)
        trial_ids.append(trial)

    trial_probabilities = np.asarray(trial_probabilities)
    trial_labels = np.asarray(trial_labels, dtype=np.int64)
    trial_predictions = trial_probabilities.argmax(axis=1)
    trial_subjects = np.asarray(trial_subjects)
    trial_ids = np.asarray(trial_ids, dtype=np.int64)
    trial_metrics = _classification_metrics(trial_labels, trial_predictions)

    subject_metrics = []
    for subject in np.unique(trial_subjects):
        mask = trial_subjects == subject
        metrics = _classification_metrics(trial_labels[mask], trial_predictions[mask])
        subject_metrics.append({
            "subject": str(subject),
            "n_trials": int(mask.sum()),
            **metrics,
        })

    subject_f1 = np.asarray([item["f1"] for item in subject_metrics], dtype=float)
    subject_accuracy = np.asarray([item["accuracy"] for item in subject_metrics], dtype=float)

    return {
        "window": window_metrics,
        "trial": trial_metrics,
        "subject_f1_mean": float(subject_f1.mean()),
        "subject_f1_std": float(subject_f1.std(ddof=0)),
        "worst_subject_f1": float(subject_f1.min()),
        "subject_accuracy_mean": float(subject_accuracy.mean()),
        "subject_metrics": subject_metrics,
        "trial_labels": trial_labels,
        "trial_predictions": trial_predictions,
        "trial_probabilities": trial_probabilities,
        "trial_subjects": trial_subjects,
        "trial_ids": trial_ids,
    }
