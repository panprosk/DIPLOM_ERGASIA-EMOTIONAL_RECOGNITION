"""
memory.py
Βήμα 9: Η μνήμη του agent.
Κρατά history experiments, hypotheses, best/failed configs.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Hypothesis:
    """Μία επιστημονική υπόθεση του agent."""
    statement:               str
    supporting_experiments:  List[str] = field(default_factory=list)
    contradicting_experiments: List[str] = field(default_factory=list)
    confidence:              float  = 0.0
    status:                  str    = "unverified"  # unverified / supported / rejected / refined
    refined_statement:       Optional[str] = None


class ExperimentMemory:
    """
    Κεντρική μνήμη του agent.
    Αποθηκεύει experiments, hypotheses, patterns.
    """

    def __init__(self):
        self.experiments:  List[dict]       = []
        self.hypotheses:   List[Hypothesis] = []
        self.best_configs: List[dict]       = []
        self.failed_configs: List[dict]     = []

    def add_result(self, result: dict):
        self.experiments.append(result)
        print(f"  Memory: {len(self.experiments)} experiments stored")

    def get_best_models(self, n: int = 3, metric: str = "generalization_score") -> List[dict]:
        if not self.experiments:
            return []
        sorted_exps = sorted(
            self.experiments,
            key=lambda x: x.get(metric, 0.0),
            reverse=True
        )
        return sorted_exps[:n]

    def add_hypothesis(self, hypothesis: Hypothesis):
        self.hypotheses.append(hypothesis)
        print(f"  Memory: hypothesis added → '{hypothesis.statement}'")

    def get_patterns(self) -> List[dict]:
        """
        Απλή ανάλυση patterns από το history.
        Επιστρέφει list of pattern dicts.
        """
        if len(self.experiments) < 2:
            return []

        patterns = []

        # Pattern 1: EDA μειώνει variance
        eda_exps    = [e for e in self.experiments if "EDA" in e.get("signals", [])]
        no_eda_exps = [e for e in self.experiments if "EDA" not in e.get("signals", [])]

        if eda_exps and no_eda_exps:
            eda_var    = np.mean([e.get("variance", 1.0) for e in eda_exps])
            no_eda_var = np.mean([e.get("variance", 1.0) for e in no_eda_exps])

            if eda_var < no_eda_var * 0.9:  # 10% μείωση
                patterns.append({
                    "type":        "modality_stability",
                    "signal":      "EDA",
                    "description": f"EDA reduces subject variance ({no_eda_var:.3f} → {eda_var:.3f})",
                    "strength":    round((no_eda_var - eda_var) / no_eda_var, 3),
                })

        # Pattern 2: PPG βελτιώνει confidence
        ppg_exps    = [e for e in self.experiments if "PPG" in e.get("signals", [])]
        no_ppg_exps = [e for e in self.experiments if "PPG" not in e.get("signals", [])]

        if ppg_exps and no_ppg_exps:
            ppg_conf    = np.mean([e.get("confidence", 0.0) for e in ppg_exps])
            no_ppg_conf = np.mean([e.get("confidence", 0.0) for e in no_ppg_exps])

            if ppg_conf > no_ppg_conf * 1.02:  # 2% βελτίωση
                patterns.append({
                    "type":        "modality_confidence",
                    "signal":      "PPG",
                    "description": f"PPG improves confidence ({no_ppg_conf:.3f} → {ppg_conf:.3f})",
                    "strength":    round((ppg_conf - no_ppg_conf) / no_ppg_conf, 3),
                })

        return patterns

    def summary(self):
        print(f"\n  === Memory Summary ===")
        print(f"  Experiments:  {len(self.experiments)}")
        print(f"  Hypotheses:   {len(self.hypotheses)}")
        if self.experiments:
            best = self.get_best_models(1)[0]
            print(f"  Best so far:  {best['model']} {best['signals']} "
                  f"(score={best.get('generalization_score', 0):.4f})")
