"""
test_agent.py
Ελέγχει experiment_runner, logger, memory, agent με dummy data.
Τρέξε: python test_agent.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import numpy as np
from torch.utils.data import DataLoader

print("=" * 60)
print("  AGENT COMPONENTS - FULL TEST")
print("=" * 60)


# ── Dummy Dataset ──────────────────────────────────────────
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, n=200):
        self.n   = n
        self.eeg = torch.randn(n, 32, 512)
        self.eda = torch.randn(n,  1, 512)
        self.ppg = torch.randn(n,  1, 512)
        self.lbl = torch.randint(0, 2, (n,))

    def __len__(self): return self.n

    def __getitem__(self, i):
        return {
            "eeg":   self.eeg[i],
            "eda":   self.eda[i],
            "ppg":   self.ppg[i],
            "label": self.lbl[i],
        }

train_loader = DataLoader(DummyDataset(200), batch_size=32, shuffle=True)
val_loader   = DataLoader(DummyDataset(60),  batch_size=32)
test_loader  = DataLoader(DummyDataset(60),  batch_size=32)
subject_ids_test = np.repeat(np.arange(1, 21), 3)


# ──────────────────────────────────────────────
# TEST 1: ExperimentLogger
# ──────────────────────────────────────────────
print("\n[TEST 1] ExperimentLogger...")
try:
    from experiment_logging.experiment_logger import ExperimentLogger  # ΔΙΟΡΘΩΜΕΝΟ

    logger = ExperimentLogger()

    dummy_result = {
        "experiment_id":        "test001",
        "model":                "cnn_mlp",
        "signals":              ["EEG", "EDA"],
        "accuracy":             0.72,
        "f1":                   0.70,
        "worst_f1":             0.55,
        "variance":             0.08,
        "confidence":           0.76,
        "entropy":              0.41,
        "generalization_score": 0.58,
        "best_val_f1":          0.68,
        "training_time":        12.5,
        "config":               {"model": "cnn_mlp", "signals": ["EEG", "EDA"]},
        "subject_metrics":      {1: {"f1": 0.72}, 2: {"f1": 0.55}},
    }

    logger.log(dummy_result)
    df = logger.get_all()

    assert len(df) >= 1, "FAIL: no rows in DB"
    print(f"  ✓ Logger works — {len(df)} experiment(s) in DB")

    best = logger.get_best()
    print(f"  ✓ Best experiment: {best.get('model')} "
          f"score={best.get('generalization_score')}")

except Exception as e:
    print(f"  ✗ Logger error: {e}")
    raise


# ──────────────────────────────────────────────
# TEST 2: ExperimentMemory
# ──────────────────────────────────────────────
print("\n[TEST 2] ExperimentMemory...")
try:
    from agent.memory import ExperimentMemory, Hypothesis

    mem = ExperimentMemory()

    mem.add_result({
        "model": "cnn_mlp", "signals": ["EEG"],
        "f1": 0.65, "variance": 0.12, "confidence": 0.74,
        "entropy": 0.45, "generalization_score": 0.50,
    })
    mem.add_result({
        "model": "cnn_mlp", "signals": ["EEG", "EDA"],
        "f1": 0.70, "variance": 0.07, "confidence": 0.78,
        "entropy": 0.38, "generalization_score": 0.58,
    })
    mem.add_result({
        "model": "cnn_mlp", "signals": ["EEG", "PPG"],
        "f1": 0.67, "variance": 0.10, "confidence": 0.80,
        "entropy": 0.40, "generalization_score": 0.54,
    })

    best = mem.get_best_models(2)
    assert len(best) == 2
    print(f"  ✓ Memory stores {len(mem.experiments)} experiments")
    print(f"  ✓ Best: {best[0]['signals']} "
          f"score={best[0]['generalization_score']}")

    patterns = mem.get_patterns()
    print(f"  ✓ Patterns detected: {len(patterns)}")
    for p in patterns:
        print(f"    → {p['description']}")

    h = Hypothesis(
        statement="EDA improves cross-subject stability.",
        confidence=0.7,
        status="unverified"
    )
    mem.add_hypothesis(h)
    print(f"  ✓ Hypothesis added: '{h.statement}'")

except Exception as e:
    print(f"  ✗ Memory error: {e}")
    raise


# ──────────────────────────────────────────────
# TEST 3: experiment_runner
# ──────────────────────────────────────────────
print("\n[TEST 3] Experiment Runner...")
try:
    from training.experiment_runner import (
        run_experiment,
        compute_generalization_score,
    )

    config = {
        "model":    "cnn_mlp",
        "signals":  ["EEG", "EDA"],
        "epochs":   3,
        "lr":       1e-3,
        "patience": 3,
    }

    results = run_experiment(
        config,
        train_loader,
        val_loader,
        test_loader,
        subject_ids_test,
    )

    assert "f1"                   in results
    assert "generalization_score" in results
    assert "subject_metrics"      in results

    print(f"  ✓ Experiment ran OK")
    print(f"  ✓ F1:                   {results['f1']:.4f}")
    print(f"  ✓ Generalization Score: {results['generalization_score']:.4f}")
    print(f"  ✓ Subject metric keys:  "
          f"{list(results['subject_metrics'].keys())[:5]}")

    score = compute_generalization_score(results)
    assert isinstance(score, float)
    print(f"  ✓ Score formula OK: {score:.4f}")

except Exception as e:
    print(f"  ✗ Runner error: {e}")
    raise


# ──────────────────────────────────────────────
# TEST 4: SimpleMetaAgent
# ──────────────────────────────────────────────
print("\n[TEST 4] SimpleMetaAgent (mini run)...")
try:
    from agent.simple_meta_agent import SimpleMetaAgent

    SimpleMetaAgent.BASELINE_CONFIGS = [
        {"model": "cnn_mlp", "signals": ["EEG"],
         "epochs": 2, "lr": 1e-3, "patience": 2},
        {"model": "cnn_mlp", "signals": ["EEG", "EDA"],
         "epochs": 2, "lr": 1e-3, "patience": 2},
    ]

    agent = SimpleMetaAgent(
        train_loader, val_loader, test_loader,
        subject_ids_test=subject_ids_test,
    )

    report = agent.run(max_rounds=1)

    assert "best_models" in report
    assert len(report["best_models"]) > 0
    print(f"  ✓ Agent completed 1 round")
    print(f"  ✓ Best: {report['best_models'][0]['signals']} "
          f"score={report['best_models'][0].get('generalization_score', 0):.4f}")

except Exception as e:
    print(f"  ✗ Agent error: {e}")
    raise


# ──────────────────────────────────────────────
# TEST 5: Output files
# ──────────────────────────────────────────────
print("\n[TEST 5] Output files...")

csv_exists = os.path.exists("experiments/experiments.csv")
db_exists  = os.path.exists("experiments/experiments.db")

print(f"  {'✓' if csv_exists else '✗'} experiments/experiments.csv")
print(f"  {'✓' if db_exists  else '✗'} experiments/experiments.db")

if csv_exists:
    import pandas as pd
    df = pd.read_csv("experiments/experiments.csv")
    print(f"  ✓ CSV έχει {len(df)} rows")


# ──────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ✓ ΟΛΑ ΤΑ AGENT TESTS ΠΕΡΑΣΑΝ")
print("  Έτοιμο για run_agent.py με DEAP data!")
print("=" * 60)
