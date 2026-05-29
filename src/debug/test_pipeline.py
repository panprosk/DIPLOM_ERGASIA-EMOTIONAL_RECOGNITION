import sys
import os
sys.path.append(os.path.abspath("src"))

from data.deap_loader import load_all_subjects
from data.preprocessing import preprocess_subject
from data.dataset import subject_wise_split, build_dataloaders


def main():

    print("\n=== LOADING ===")
    subjects_raw = load_all_subjects("data/deap", subject_ids=[1, 2])

    print("\n=== PREPROCESSING ===")
    subjects_clean = {}

    for sid, signals in subjects_raw.items():
        out = preprocess_subject(signals)

        for k in ["eeg", "eda", "ppg"]:
            assert not (out[k] != out[k]).any(), f"NaN in {k}"
            assert not (out[k] == float("inf")).any(), f"Inf in {k}"

        subjects_clean[sid] = out
        print(f"Subject {sid} OK")

    print("\n=== SPLIT ===")
    train_ids, val_ids, test_ids = subject_wise_split(subjects_clean)

    print("\n=== DATALOADERS ===")
    train_loader, val_loader, test_loader = build_dataloaders(
        subjects_clean,
        train_ids,
        val_ids,
        test_ids,
        batch_size=16,
        num_workers=0   # 🔥 IMPORTANT FIX
    )

    print("\n=== BATCH CHECK ===")
    batch = next(iter(train_loader))

    print("EEG:", batch["eeg"].shape)
    print("EDA:", batch["eda"].shape)
    print("PPG:", batch["ppg"].shape)
    print("Label:", batch["label"].shape)

    print("\n✅ PIPELINE OK")


if __name__ == "__main__":
    main()