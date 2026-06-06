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

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from data.deap_loader import load_all_subjects
from data.preprocessing import preprocess_subject
from data.dataset import (
    subject_wise_split,
    build_dataloaders,
)

from models.baseline_cnn_mlp import HybridCNNMLP


# =========================================================
# CONFIG
# =========================================================

DATA_DIR = "data/deap"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 64
EPOCHS = 50 # ΑΥΞΗΘΗΚΕ

LR = 1e-4 # Μειωμένο Learning Rate
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
    return_predictions=False,
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

    if return_predictions:

        return (
            running_loss / len(loader),
            acc,
            f1,
            np.array(all_targets),
            np.array(all_preds),
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
    # CLASS DISTRIBUTION CHECK
    # -----------------------------------------------------

    print("\nChecking class distribution...")

    train_labels = []
    for batch in train_loader:
        train_labels.extend(batch["label"].numpy())
    train_labels = np.array(train_labels)

    train_class0_count = (train_labels == 0).sum()
    train_class1_count = (train_labels == 1).sum()
    total_train_samples = train_class0_count + train_class1_count

    weight_class0 = total_train_samples / (2.0 * train_class0_count)
    weight_class1 = total_train_samples / (2.0 * train_class1_count)
    class_weights = torch.tensor([weight_class0, weight_class1], dtype=torch.float32).to(DEVICE)

    print(f"Train: class0={train_class0_count} class1={train_class1_count}")
    print(f"Class weights: {class_weights.tolist()}")

    for name, loader in [
        ("Val", val_loader),
        ("Test", test_loader),
    ]:
        labels = []
        for batch in loader:
            labels.extend(batch["label"].numpy())
        labels = np.array(labels)
        print(
            f"{name}: "
            f"class0={(labels==0).sum()} "
            f"class1={(labels==1).sum()}"
        )

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    print("\n[5] Building model...")

    feature_dims = {
        "eeg": 352,
        "eda": 8,
        "ppg": 8
    }

    model = HybridCNNMLP(
        signals=["EEG"],
        feature_dims=feature_dims,
        cnn_out_dim=128,
        hidden_dim=256,
        dropout=0.2,
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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,  # ΕΦΑΡΜΟΓΗ ΒΑΡΩΝ ΚΛΑΣΕΩΝ
        label_smoothing=0.1
    )

    # -----------------------------------------------------
    # TRAINING
    # -----------------------------------------------------

    print("\n[6] Training...\n")

    best_val_f1 = 0.0

    patience = 10  # ΑΥΞΗΘΗΚΕ
    patience_counter = 0

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

        scheduler.step(val_loss) # ΕΝΗΜΕΡΩΣΗ SCHEDULER

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
            patience_counter = 0

            torch.save(
                model.state_dict(),
                "best_baseline.pt"
            )

            print(
                f"  ✓ Best model saved "
                f"(F1={val_f1:.4f})"
            )

        else:

            patience_counter += 1

            print(
                f"  No improvement "
                f"({patience_counter}/{patience})"
            )

            if patience_counter >= patience:

                print("\nEarly stopping triggered.")
                break

    # -----------------------------------------------------
    # TEST
    # -----------------------------------------------------

    print("\n[7] Final Test Evaluation...")

    model.load_state_dict(
        torch.load(
            "best_baseline.pt",
            map_location=DEVICE
        )
    )

    (
        test_loss,
        test_acc,
        test_f1,
        y_true,
        y_pred,
    ) = evaluate(
        model,
        test_loader,
        criterion,
        return_predictions=True,
    )

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(f"Test Loss : {test_loss:.4f}")
    print(f"Test Acc  : {test_acc:.4f}")
    print(f"Test F1   : {test_f1:.4f}")

    print("\nClassification Report")
    print(
        classification_report(
            y_true,
            y_pred,
            digits=4
        )
    )

    print("\nConfusion Matrix")
    print(
        confusion_matrix(
            y_true,
            y_pred
        )
    )

    print("\nSaved:")
    print("best_baseline.pt")


# =========================================================

if __name__ == "__main__":
    main()