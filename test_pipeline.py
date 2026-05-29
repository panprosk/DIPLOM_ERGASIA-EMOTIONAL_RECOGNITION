"""
test_pipeline.py
Ελέγχει βήμα-βήμα ότι όλα δουλεύουν σωστά.
Τρέξε: python test_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import numpy as np

print("=" * 60)
print("  THESIS PIPELINE - FULL TEST")
print("=" * 60)


# ──────────────────────────────────────────────
# TEST 1: Imports
# ──────────────────────────────────────────────
print("\n[TEST 1] Imports...")
try:
    from models.baseline_cnn_mlp import HybridCNNMLP, build_model
    from training.metrics import compute_entropy, compute_metrics
    from training.trainer import train_one_epoch, evaluate
    print("  ✓ Όλα τα imports δουλεύουν")
except ImportError as e:
    print(f"  ✗ Import error: {e}")
    sys.exit(1)


# ──────────────────────────────────────────────
# TEST 2: Device
# ──────────────────────────────────────────────
print("\n[TEST 2] Device check...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  ✓ Device: {device}")
if device.type == "cuda":
    print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")


# ──────────────────────────────────────────────
# TEST 3: Model creation
# ──────────────────────────────────────────────
print("\n[TEST 3] Model creation...")

configs = [
    {"model": "cnn_mlp", "signals": ["EEG"]},
    {"model": "cnn_mlp", "signals": ["EEG", "EDA"]},
    {"model": "cnn_mlp", "signals": ["EEG", "PPG"]},
    {"model": "cnn_mlp", "signals": ["EEG", "EDA", "PPG"]},
]

for cfg in configs:
    m = build_model(cfg).to(device)
    n = m.get_num_params()
    print(f"  ✓ Signals {cfg['signals']} → {n:,} parameters")


# ──────────────────────────────────────────────
# TEST 4: Forward pass με dummy data
# ──────────────────────────────────────────────
print("\n[TEST 4] Forward pass (dummy data)...")

BATCH = 8
T     = 512   # 4 sec * 128 Hz

dummy = {
    "eeg": torch.randn(BATCH, 32, T).to(device),
    "eda": torch.randn(BATCH,  1, T).to(device),
    "ppg": torch.randn(BATCH,  1, T).to(device),
}

model = build_model({"signals": ["EEG", "EDA", "PPG"]}).to(device)
out   = model(dummy)

assert out["logits"].shape     == (BATCH, 2), "FAIL: logits shape"
assert out["probs"].shape      == (BATCH, 2), "FAIL: probs shape"
assert out["confidence"].shape == (BATCH,),   "FAIL: confidence shape"

print(f"  ✓ logits shape:     {tuple(out['logits'].shape)}")
print(f"  ✓ probs shape:      {tuple(out['probs'].shape)}")
print(f"  ✓ confidence shape: {tuple(out['confidence'].shape)}")
print(f"  ✓ features shape:   {tuple(out['features'].shape)}")


# ──────────────────────────────────────────────
# TEST 5: Probabilities αθροίζουν σε 1
# ──────────────────────────────────────────────
print("\n[TEST 5] Softmax probabilities...")

prob_sums = out["probs"].sum(dim=-1)
assert torch.allclose(prob_sums, torch.ones(BATCH, device=device), atol=1e-5)
print(f"  ✓ Prob sums: min={prob_sums.min():.6f}  max={prob_sums.max():.6f}")


# ──────────────────────────────────────────────
# TEST 6: Backward pass
# ──────────────────────────────────────────────
print("\n[TEST 6] Backward pass & gradients...")

import torch.nn as nn
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
labels    = torch.randint(0, 2, (BATCH,)).to(device)

optimizer.zero_grad()
out2  = model(dummy)
loss  = criterion(out2["logits"], labels)
loss.backward()
optimizer.step()

grad_norms = [
    p.grad.norm().item()
    for p in model.parameters()
    if p.grad is not None
]
assert all(np.isfinite(g) for g in grad_norms), "FAIL: NaN/Inf gradients"
print(f"  ✓ Loss: {loss.item():.4f}")
print(f"  ✓ Gradients OK  (max norm: {max(grad_norms):.4f})")


# ──────────────────────────────────────────────
# TEST 7: Metrics
# ──────────────────────────────────────────────
print("\n[TEST 7] Metrics computation...")

probs_np = out2["probs"].detach().cpu().numpy()
y_pred   = probs_np.argmax(axis=-1)
y_true   = labels.cpu().numpy()

metrics = compute_metrics(y_true, y_pred, probs_np)
entropy = compute_entropy(probs_np)

print(f"  ✓ Accuracy:   {metrics['accuracy']:.4f}")
print(f"  ✓ F1:         {metrics['f1']:.4f}")
print(f"  ✓ Confidence: {metrics['confidence']:.4f}")
print(f"  ✓ Entropy:    {metrics['entropy']:.4f}  (max={np.log(2):.4f})")


# ──────────────────────────────────────────────
# TEST 8: Mini training loop (5 steps)
# ──────────────────────────────────────────────
print("\n[TEST 8] Mini training loop (5 batches)...")

from torch.utils.data import DataLoader, TensorDataset

# Dummy dataset με 100 samples
N = 100
eeg_data = torch.randn(N, 32, T)
eda_data = torch.randn(N,  1, T)
ppg_data = torch.randn(N,  1, T)
lbl_data = torch.randint(0, 2, (N,))

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.N = N
    def __len__(self):
        return self.N
    def __getitem__(self, i):
        return {
            "eeg":   eeg_data[i],
            "eda":   eda_data[i],
            "ppg":   ppg_data[i],
            "label": lbl_data[i],
        }

loader    = DataLoader(DummyDataset(), batch_size=16, shuffle=True)
model2    = build_model({"signals": ["EEG", "EDA", "PPG"]}).to(device)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)

train_metrics = train_one_epoch(model2, loader, optimizer2, criterion, device)
val_metrics   = evaluate(model2, loader, criterion, device)

print(f"  ✓ Train loss: {train_metrics['loss']:.4f}  acc: {train_metrics['accuracy']:.4f}")
print(f"  ✓ Val   loss: {val_metrics['loss']:.4f}   acc: {val_metrics['accuracy']:.4f}  F1: {val_metrics['f1']:.4f}")


# ──────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ✓ ΟΛΑ ΤΑ TESTS ΠΕΡΑΣΑΝ ΕΠΙΤΥΧΩΣ")
print("  Pipeline είναι έτοιμο για DEAP data!")
print("=" * 60)
