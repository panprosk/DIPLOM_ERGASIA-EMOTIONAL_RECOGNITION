"""
Βήμα 14-15: Τρέχει ένα πείραμα και αποθηκεύει τα αποτελέσματα.
"""

import time
import uuid
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from models.baseline_cnn_mlp import build_model
from training.trainer import train_model, evaluate
from training.metrics import compute_metrics, compute_subject_wise_metrics


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_generalization_score(results: dict) -> float:
    """
    Βήμα 3 του σεναρίου agent.
    Δεν επιλέγει μόνο από F1 — σταθμίζει cross-subject robustness.

    Score = 0.40 * mean_f1
           + 0.20 * worst_f1
           + 0.15 * confidence
           - 0.10 * entropy
           - 0.15 * variance
    """
    score = (
          0.40 * results.get("f1",         0.0)
        + 0.20 * results.get("worst_f1",   0.0)
        + 0.15 * results.get("confidence", 0.0)
        - 0.10 * results.get("entropy",    0.0)
        - 0.15 * results.get("variance",   0.0)
    )
    return round(float(score), 4)


def run_experiment(config:       dict,
                   train_loader: DataLoader,
                   val_loader:   DataLoader,
                   test_loader:  DataLoader,
                   subject_ids_test: np.ndarray = None) -> dict:
    """
    Εκτελεί ένα πλήρες experiment.

    Args:
        config:           experiment configuration dict
        train_loader:     DataLoader για training
        val_loader:       DataLoader για validation
        test_loader:      DataLoader για test
        subject_ids_test: array με subject id ανά sample (για subject-wise metrics)

    Returns:
        results dict με όλα τα metrics
    """
    experiment_id = str(uuid.uuid4())[:8]
    start_time    = time.time()

    print(f"\n{'='*55}")
    print(f"  Experiment {experiment_id}")
    print(f"  Model:   {config.get('model', 'cnn_mlp')}")
    print(f"  Signals: {config.get('signals', ['EEG','EDA','PPG'])}")
    print(f"{'='*55}")

    # ── Βήμα 1: Δημιουργία μοντέλου ──
    model     = build_model(config).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # ── Βήμα 2: Training ──
    train_results = train_model(
        model, train_loader, val_loader, config, DEVICE
    )

    # ── Βήμα 3: Evaluation στο test set ──
    test_metrics = evaluate(model, test_loader, criterion, DEVICE)

    # ── Βήμα 4: Subject-wise metrics ──
    subject_metrics = {}
    worst_f1        = test_metrics["f1"]
    variance        = 0.0

    if subject_ids_test is not None:
        # Collect predictions με subject ids
        model.eval()
        all_preds, all_labels, all_probs, all_sids = [], [], [], []

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                eeg   = batch["eeg"].to(DEVICE)
                eda   = batch["eda"].to(DEVICE)
                ppg   = batch["ppg"].to(DEVICE)
                label = batch["label"]

                out   = model({"eeg": eeg, "eda": eda, "ppg": ppg})
                preds = out["logits"].argmax(dim=-1).cpu().numpy()
                probs = out["probs"].cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(label.numpy())
                all_probs.extend(probs)

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs  = np.array(all_probs)

        # Subject-wise F1
        subject_metrics = compute_subject_wise_metrics(
            all_labels, all_preds, subject_ids_test[:len(all_preds)]
        )
        worst_f1 = subject_metrics.get("worst_f1", test_metrics["f1"])
        variance = subject_metrics.get("variance",  0.0)

    elapsed = round(time.time() - start_time, 2)

    # ── Συναρμολόγηση αποτελεσμάτων ──
    results = {
        "experiment_id":   experiment_id,
        "model":           config.get("model",   "cnn_mlp"),
        "signals":         config.get("signals", ["EEG", "EDA", "PPG"]),
        "accuracy":        round(test_metrics["accuracy"],   4),
        "f1":              round(test_metrics["f1"],         4),
        "confidence":      round(test_metrics["confidence"], 4),
        "entropy":         round(test_metrics["entropy"],    4),
        "worst_f1":        round(worst_f1,  4),
        "variance":        round(variance,  4),
        "best_val_f1":     round(train_results["best_val_f1"], 4),
        "training_time":   elapsed,
        "subject_metrics": subject_metrics,
        "config":          config,
    }

    results["generalization_score"] = compute_generalization_score(results)

    print(f"\n  Results:")
    print(f"    Accuracy:             {results['accuracy']:.4f}")
    print(f"    F1:                   {results['f1']:.4f}")
    print(f"    Worst Subject F1:     {results['worst_f1']:.4f}")
    print(f"    Subject Variance:     {results['variance']:.4f}")
    print(f"    Confidence:           {results['confidence']:.4f}")
    print(f"    Entropy:              {results['entropy']:.4f}")
    print(f"    Generalization Score: {results['generalization_score']:.4f}")
    print(f"    Training time:        {elapsed}s")

    return results
