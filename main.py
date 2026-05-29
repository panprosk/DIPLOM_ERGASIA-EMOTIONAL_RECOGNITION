"""
main.py
Entry point — ενώνει όλα τα παραπάνω.
"""

from data.deap_loader   import load_all_subjects
from data.preprocessing import preprocess_subject
from data.dataset       import subject_wise_split, build_dataloaders


DATA_DIR    = "data/deap"
BATCH_SIZE  = 64
NUM_WORKERS = 4


def main():
    print("=" * 50)
    print("  DEAP Emotion Recognition Pipeline")
    print("=" * 50)

    # 1-3: Load
    print("\n[1] Loading subjects...")
    all_subjects_raw = load_all_subjects(DATA_DIR)

    # 6: Preprocess (signal cleaning)
    print("\n[2] Preprocessing signals...")
    all_subjects = {}
    for sid, signals in all_subjects_raw.items():
        print(f"  Preprocessing subject {sid:02d}...")
        all_subjects[sid] = preprocess_subject(signals)

    # 5: Split
    print("\n[3] Subject-wise split...")
    train_ids, val_ids, test_ids = subject_wise_split(all_subjects)

    # 7+9: Segment & build loaders
    print("\n[4] Building DataLoaders...")
    train_loader, val_loader, test_loader = build_dataloaders(
        all_subjects, train_ids, val_ids, test_ids,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # Verify
    print("\n[5] Verifying batch structure...")
    batch = next(iter(train_loader))
    print(f"  EEG batch: {batch['eeg'].shape}")   # (64, 32, 512)
    print(f"  EDA batch: {batch['eda'].shape}")   # (64, 1,  512)
    print(f"  PPG batch: {batch['ppg'].shape}")   # (64, 1,  512)
    print(f"  Labels:    {batch['label'].shape}") # (64,)
    print(f"\n✓ Pipeline ready. Train batches: {len(train_loader)}")


if __name__ == "__main__":
    main()
