"""
run_agent.py
Entry point για τον agent με πραγματικά DEAP data.
Τρέξε: python run_agent.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np

from data.deap_loader    import load_all_subjects
from data.preprocessing  import preprocess_subject
from data.dataset        import (subject_wise_split,
                                  build_dataloaders,
                                  segment_trials)
from agent.simple_meta_agent import SimpleMetaAgent


DATA_DIR = "data/deap"


def build_subject_ids_test(all_subjects: dict,
                            test_ids: list,
                            window_sec: float = 4.0,
                            overlap:    float = 0.5,
                            fs:         int   = 128) -> np.ndarray:
    """
    Δημιουργεί array με subject_id ανά window για το test set.
    Χρησιμοποιείται από τον agent για subject-wise metrics.
    """
    subject_ids = []
    for sid in test_ids:
        signals = all_subjects[sid]
        _, _, _, y = segment_trials(
            signals, window_sec=window_sec, overlap=overlap, fs=fs
        )
        subject_ids.extend([sid] * len(y))

    return np.array(subject_ids)


def main():
    print("=" * 55)
    print("  EMOTION RECOGNITION — DEAP PIPELINE")
    print("  Προσκεφαλάς Παναγιώτης — Διπλωματική")
    print("=" * 55)

    # ── 1. Load ──────────────────────────────
    print("\n[1] Loading DEAP subjects...")
    raw = load_all_subjects(DATA_DIR)

    if not raw:
        print("  ✗ Δεν βρέθηκαν subjects στο data/deap/")
        print("  Βάλε τα s01.dat - s32.dat στον φάκελο data/deap/")
        return

    # ── 2. Preprocess ────────────────────────
    print("\n[2] Preprocessing signals...")
    subjects = {}
    for sid, signals in raw.items():
        print(f"  Preprocessing subject {sid:02d}...")
        subjects[sid] = preprocess_subject(signals)

    # ── 3. Split ─────────────────────────────
    print("\n[3] Subject-wise split...")
    train_ids, val_ids, test_ids = subject_wise_split(subjects)

    # ── 4. DataLoaders ───────────────────────
    print("\n[4] Building DataLoaders...")
    train_loader, val_loader, test_loader = build_dataloaders(
        subjects,
        train_ids,
        val_ids,
        test_ids,
        batch_size=64,
        num_workers=0,    # Windows-safe
    )

    # ── 5. Subject IDs για test ──────────────
    print("\n[5] Building subject ID array for test set...")
    subject_ids_test = build_subject_ids_test(subjects, test_ids)
    print(f"  Test windows:  {len(subject_ids_test)}")
    print(f"  Test subjects: {np.unique(subject_ids_test).tolist()}")

    # ── 6. Agent ─────────────────────────────
    print("\n[6] Starting Scientific Meta Agent...")
    agent = SimpleMetaAgent(
        train_loader,
        val_loader,
        test_loader,
        subject_ids_test=subject_ids_test,
    )

    report = agent.run(max_rounds=2)

    # ── 7. Summary ───────────────────────────
    print("\n" + "=" * 55)
    print("  PIPELINE COMPLETE")
    print("=" * 55)
    print(f"  Experiments run:  {len(report.get('best_models', []))}")
    print(f"  Results saved in: experiments/")

    return report


if __name__ == "__main__":
    main()
