"""
train_baseline.py

Fast baseline training WITHOUT agent.
Για γρήγορο experimentation και debugging.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from data.deap_loader import load_all_subjects
from data.preprocessing import preprocess_subject
from data.dataset import (
    subject_wise_split,
    build_dataloaders,
)

# IMPORTANT:
from models.baseline_cnn_mlp import HybridCNNMLP


# =========================================================
# CONFIG
# =========================================================

DATA_DIR = "data/deap"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 64
EPOCHS = 20

LR = 3e-4
WEIGHT_DECAY = 1e-3

SEED = 42


# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_seed(seed=42):

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# TRAIN
# =========================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
):

    model.train()

    running_loss = 0.0

    all_preds = []
    all_targets = []

    loop = tqdm(loader, leave=False)

    for batch in loop:

        batch = {
            k: v.to(DEVICE)
            for k, v in batch.items()
        }

        optimizer.zero_grad()

        outputs = model(batch)

        logits = outputs["logits"]

        loss = criterion(
            logits,
            batch["label"]
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_targets.extend(
            batch["label"].cpu().numpy()
        )

        loop.set_postfix(
            loss=loss.item()
        )

    acc = accuracy_score(
        all_targets,
        all_preds
    )

    f1 = f1_score(
        all_targets,
        all_preds
    )

    return (
        running_loss / len(loader),
        acc,
        f1,
    )


# =========================================================
# EVALUATION
# =========================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
):

    model.eval()

    running_loss = 0.0

    all_preds = []
    all_targets = []

    for batch in loader:

        batch = {
            k: v.to(DEVICE)
            for k, v in batch.items()
        }

        outputs = model(batch)

        logits = outputs["logits"]

        loss = criterion(
            logits,
            batch["label"]
        )

        running_loss += loss.item()

        preds = torch.argmax(
            logits,
            dim=1
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_targets.extend(
            batch["label"].cpu().numpy()
        )

    acc = accuracy_score(
        all_targets,
        all_preds
    )

    f1 = f1_score(
        all_targets,
        all_preds
    )

    return (
        running_loss / len(loader),
        acc,
        f1,
    )


# =========================================================
# MAIN
# =========================================================

def main():

    set_seed(SEED)

    print("=" * 60)
    print("DEAP BASELINE TRAINING")
    print("=" * 60)

    print(f"\nDevice: {DEVICE}")

    # -----------------------------------------------------
    # LOAD
    # -----------------------------------------------------

    print("\n[1] Loading subjects...")

    raw_subjects = load_all_subjects(DATA_DIR)

    if len(raw_subjects) == 0:

        print("\nERROR: No DEAP subjects found.")
        return

    # -----------------------------------------------------
    # PREPROCESS
    # -----------------------------------------------------

    print("\n[2] Preprocessing...")

    subjects = {}

    for sid, signals in raw_subjects.items():

        print(f"Subject {sid:02d}")

        subjects[sid] = preprocess_subject(signals)

    # -----------------------------------------------------
    # SPLIT
    # -----------------------------------------------------

    print("\n[3] Subject-wise split...")

    train_ids, val_ids, test_ids = subject_wise_split(subjects)

    # -----------------------------------------------------
    # DATALOADERS
    # -----------------------------------------------------

    print("\n[4] Building dataloaders...")

    train_loader, val_loader, test_loader = build_dataloaders(
        subjects,
        train_ids,
        val_ids,
        test_ids,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    print("\n[5] Building model...")

    model = HybridCNNMLP(

        signals=["EEG"],

        cnn_out_dim=64,

        hidden_dim=128,

        dropout=0.5,

    ).to(DEVICE)

    print(model)

    # -----------------------------------------------------
    # OPTIMIZER
    # -----------------------------------------------------

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LR,

        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )

    # -----------------------------------------------------
    # TRAINING
    # -----------------------------------------------------

    print("\n[6] Training...\n")

    best_val_f1 = 0.0

    for epoch in range(EPOCHS):

        train_loss, train_acc, train_f1 = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        val_loss, val_acc, val_f1 = evaluate(
            model,
            val_loader,
            criterion,
        )

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Train F1: {train_f1:.4f} || "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            torch.save(
                model.state_dict(),
                "best_baseline.pt"
            )

            print(
                f"  ✓ Best model saved "
                f"(F1={val_f1:.4f})"
            )

    # -----------------------------------------------------
    # TEST
    # -----------------------------------------------------

    print("\n[7] Final Test Evaluation...")

    model.load_state_dict(
        torch.load("best_baseline.pt")
    )

    test_loss, test_acc, test_f1 = evaluate(
        model,
        test_loader,
        criterion,
    )

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(f"Test Loss : {test_loss:.4f}")
    print(f"Test Acc  : {test_acc:.4f}")
    print(f"Test F1   : {test_f1:.4f}")

    print("\nSaved:")
    print("best_baseline.pt")


# =========================================================

if __name__ == "__main__":
    main()