"""
metrics.py
Βήμα 13: Accuracy, F1, confidence, entropy + subject-wise metrics
"""

import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def compute_entropy(probs: np.ndarray) -> np.ndarray:
    """
    Shannon entropy ανά sample.
    probs: (N, n_classes)
    returns: (N,)
    """
    probs   = np.clip(probs, 1e-8, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=-1)
    return entropy


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    probs:  np.ndarray) -> dict:
    """
    Υπολογίζει όλα τα metrics.

    Args:
        y_true: (N,)  ground truth labels
        y_pred: (N,)  predicted labels
        probs:  (N, 2) softmax probabilities
    """
    confidence = probs.max(axis=-1).mean()
    entropy    = compute_entropy(probs).mean()

    return {
        "accuracy":   float(accuracy_score(y_true, y_pred)),
        "f1":         float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "confidence": float(confidence),
        "entropy":    float(entropy),
    }


def compute_subject_wise_metrics(y_true:      np.ndarray,
                                  y_pred:      np.ndarray,
                                  subject_ids: np.ndarray) -> dict:
    """
    Υπολογίζει F1 ανά subject.
    Χρησιμοποιείται από τον agent για cross-subject analysis.

    Returns:
        {
          subject_id: {"f1": ..., "accuracy": ..., "n_samples": ...},
          ...
          "mean_f1":  ...,
          "worst_f1": ...,
          "variance": ...,
        }
    """
    results   = {}
    f1_scores = []

    for sid in np.unique(subject_ids):
        mask = subject_ids == sid
        yt   = y_true[mask]
        yp   = y_pred[mask]

        f1  = f1_score(yt, yp, average="binary", zero_division=0)
        acc = accuracy_score(yt, yp)

        results[int(sid)] = {
            "f1":        round(float(f1),  4),
            "accuracy":  round(float(acc), 4),
            "n_samples": int(mask.sum()),
        }
        f1_scores.append(f1)

    f1_arr = np.array(f1_scores)
    results["mean_f1"]  = round(float(f1_arr.mean()), 4)
    results["worst_f1"] = round(float(f1_arr.min()),  4)
    results["variance"] = round(float(f1_arr.var()),  4)

    return results
