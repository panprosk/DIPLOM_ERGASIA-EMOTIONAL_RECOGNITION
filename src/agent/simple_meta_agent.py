"""
simple_meta_agent.py
Βήμα 17: Πρώτη έκδοση του Scientific Meta Agent.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.memory import ExperimentMemory, Hypothesis
from experiment_logging.experiment_logger import ExperimentLogger  # ΔΙΟΡΘΩΜΕΝΟ


class SimpleMetaAgent:

    BASELINE_CONFIGS = [
        {"model": "cnn_mlp", "signals": ["EEG"],
         "epochs": 30, "lr": 1e-3, "patience": 7},

        {"model": "cnn_mlp", "signals": ["EEG", "EDA"],
         "epochs": 30, "lr": 1e-3, "patience": 7},

        {"model": "cnn_mlp", "signals": ["EEG", "PPG"],
         "epochs": 30, "lr": 1e-3, "patience": 7},

        {"model": "cnn_mlp", "signals": ["EEG", "EDA", "PPG"],
         "epochs": 30, "lr": 1e-3, "patience": 7},
    ]

    def __init__(self,
                 train_loader,
                 val_loader,
                 test_loader,
                 subject_ids_test=None):

        self.train_loader     = train_loader
        self.val_loader       = val_loader
        self.test_loader      = test_loader
        self.subject_ids_test = subject_ids_test
        self.memory           = ExperimentMemory()
        self.logger           = ExperimentLogger()
        self.round            = 0

    def run(self, max_rounds: int = 2):
        print("\n" + "="*55)
        print("  SCIENTIFIC META AGENT — START")
        print("="*55)

        # Round 1: Baseline
        print("\n[Round 1] Baseline experiments...")
        self._run_configs(self.BASELINE_CONFIGS)

        patterns   = self.memory.get_patterns()
        hypotheses = self._generate_hypotheses(patterns)

        self.memory.summary()

        if max_rounds < 2 or not hypotheses:
            print("\n  Agent finished after baseline round.")
            return self._final_report()

        # Round 2: Targeted
        print("\n[Round 2] Hypothesis-driven experiments...")
        targeted = self._design_targeted_experiments(hypotheses)
        self._run_configs(targeted)

        self.memory.summary()
        return self._final_report()

    def _run_configs(self, configs: list):
        from training.experiment_runner import run_experiment

        for cfg in configs:
            results = run_experiment(
                cfg,
                self.train_loader,
                self.val_loader,
                self.test_loader,
                self.subject_ids_test,
            )
            self.memory.add_result(results)
            self.logger.log(results)

    def _generate_hypotheses(self, patterns: list) -> list:
        hypotheses = []

        for p in patterns:
            if p["type"] == "modality_stability" and p["signal"] == "EDA":
                h = Hypothesis(
                    statement=(
                        "EDA improves cross-subject stability by reducing "
                        "subject variance in emotion recognition."
                    ),
                    confidence=min(p["strength"] * 2, 0.9),
                    status="unverified",
                )
                self.memory.add_hypothesis(h)
                hypotheses.append(h)
                print(f"\n  Hypothesis generated:")
                print(f"  → '{h.statement}'")

            elif p["type"] == "modality_confidence" and p["signal"] == "PPG":
                h = Hypothesis(
                    statement=(
                        "PPG improves prediction confidence "
                        "without necessarily increasing F1."
                    ),
                    confidence=min(p["strength"] * 2, 0.9),
                    status="unverified",
                )
                self.memory.add_hypothesis(h)
                hypotheses.append(h)
                print(f"\n  Hypothesis generated:")
                print(f"  → '{h.statement}'")

        if not patterns:
            print("  No patterns detected yet — need more experiments.")

        return hypotheses

    def _design_targeted_experiments(self, hypotheses: list) -> list:
        configs = []

        for h in hypotheses:
            if "EDA" in h.statement and "stability" in h.statement:
                for window in [2, 6]:
                    configs.append({
                        "model":       "cnn_mlp",
                        "signals":     ["EEG", "EDA"],
                        "window_size": window,
                        "epochs":      20,
                        "lr":          1e-3,
                        "patience":    5,
                    })

        return configs if configs else self.BASELINE_CONFIGS[:2]

    def _final_report(self) -> dict:
        best_models  = self.memory.get_best_models(3)
        all_patterns = self.memory.get_patterns()

        print("\n" + "="*55)
        print("  AGENT FINAL REPORT")
        print("="*55)
        print(f"\n  Total experiments: {len(self.memory.experiments)}")
        print(f"  Hypotheses:        {len(self.memory.hypotheses)}")
        print(f"\n  Top 3 configurations:")

        for i, exp in enumerate(best_models, 1):
            print(f"    {i}. {exp['model']} {exp['signals']}")
            print(f"       F1={exp['f1']:.4f}  "
                  f"Worst={exp['worst_f1']:.4f}  "
                  f"Score={exp.get('generalization_score', 0):.4f}")

        print(f"\n  Patterns detected: {len(all_patterns)}")
        for p in all_patterns:
            print(f"    → {p['description']}")

        print(f"\n  Results saved in: experiments/")
        print("="*55)

        return {
            "best_models": best_models,
            "patterns":    all_patterns,
            "hypotheses":  [h.statement for h in self.memory.hypotheses],
        }
