"""
trainer.py
Βήμα 11-12: Training loop + Validation/Evaluation loop
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device) -> dict:
    """
    Ένα epoch training.
    Returns: {'loss': float, 'accuracy': float}
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(loader, desc="  Train", leave=False):
        eeg   = batch["eeg"].to(device)
        eda   = batch["eda"].to(device)
        ppg   = batch["ppg"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad()

        out    = model({"eeg": eeg, "eda": eda, "ppg": ppg})
        logits = out["logits"]
        loss   = criterion(logits, label)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * label.size(0)
        preds       = logits.argmax(dim=-1)
        correct    += (preds == label).sum().item()
        total      += label.size(0)

    return {
        "loss":     total_loss / total,
        "accuracy": correct / total,
    }


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> dict:
    """
    Evaluation loop.
    Returns: loss, accuracy, f1, confidence, entropy
    """
    from sklearn.metrics import f1_score
    from .metrics import compute_entropy    # relative import

    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_confs, all_probs = [], [], [], []

    for batch in tqdm(loader, desc="  Eval ", leave=False):
        eeg   = batch["eeg"].to(device)
        eda   = batch["eda"].to(device)
        ppg   = batch["ppg"].to(device)
        label = batch["label"].to(device)

        out    = model({"eeg": eeg, "eda": eda, "ppg": ppg})
        logits = out["logits"]
        loss   = criterion(logits, label)

        total_loss += loss.item() * label.size(0)
        preds       = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())
        all_confs.extend(out["confidence"].cpu().numpy())
        all_probs.extend(out["probs"].cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confs  = np.array(all_confs)
    all_probs  = np.array(all_probs)

    n = len(all_labels)

    return {
        "loss":       total_loss / n,
        "accuracy":   float((all_preds == all_labels).mean()),
        "f1":         float(f1_score(all_labels, all_preds,
                                     average="binary", zero_division=0)),
        "confidence": float(all_confs.mean()),
        "entropy":    float(compute_entropy(all_probs).mean()),
    }


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader:   DataLoader,
                config: dict,
                device: torch.device) -> dict:
    """
    Full training loop με early stopping βάσει val F1.
    Returns: {'best_val_f1': float, 'history': list}
    """
    epochs   = config.get("epochs",       50)
    lr       = config.get("lr",           1e-3)
    patience = config.get("patience",     10)
    wd       = config.get("weight_decay", 1e-4)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    best_val_f1  = 0.0
    best_state   = None
    patience_ctr = 0
    history      = []

    for epoch in range(1, epochs + 1):
        train_m = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_m   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history.append({
            "epoch":      epoch,
            "train_loss": train_m["loss"],
            "train_acc":  train_m["accuracy"],
            "val_loss":   val_m["loss"],
            "val_acc":    val_m["accuracy"],
            "val_f1":     val_m["f1"],
        })

        print(f"  Epoch {epoch:3d} | "
              f"Train loss: {train_m['loss']:.4f} "
              f"acc: {train_m['accuracy']:.3f} | "
              f"Val loss: {val_m['loss']:.4f} "
              f"acc: {val_m['accuracy']:.3f} "
              f"F1: {val_m['f1']:.3f}")

        # Early stopping
        if val_m["f1"] > best_val_f1:
            best_val_f1  = val_m["f1"]
            best_state   = {k: v.clone()
                            for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)

    return {
        "best_val_f1": best_val_f1,
        "history":     history,
    }
