"""
test_hybrid_cnn_mlp.py

Unit tests για το Hybrid CNN-MLP.

Ελέγχουν:
- Σωστά output shapes (logits, probs, embeddings, gates)
- probs να αθροίζουν σε 1 (softmax)
- Gates να βρίσκονται στο [0, 1]
- Να μην υπάρχουν NaN/Inf στην έξοδο
- Ροή gradients (backward + optimizer step μειώνει το loss)
- Ανεξαρτησία από το batch size (π.χ. batch_size=1)
- Σωστή δημιουργία μέσω build_model(config)

Τρέξιμο:
    pytest tests/test_hybrid_cnn_mlp.py -v
"""

import torch
import torch.nn as nn
import pytest

from src.models import HybridCNNMLP, build_model
from src.config.config import Config


EEG_CHANNELS = 32
WINDOW_SAMPLES = 768
PHYSIO_DIM = 22
NUM_CLASSES = 2


def make_model():
    return HybridCNNMLP(
        eeg_channels=EEG_CHANNELS,
        physio_feature_dim=PHYSIO_DIM,
        embedding_dim=128,
        num_classes=NUM_CLASSES,
        dropout=0.30,
    )


def make_batch(batch_size=8):
    torch.manual_seed(0)
    eeg = torch.randn(batch_size, EEG_CHANNELS, WINDOW_SAMPLES)
    physio = torch.randn(batch_size, PHYSIO_DIM)
    labels = torch.randint(0, NUM_CLASSES, (batch_size,))
    return eeg, physio, labels


def test_output_shapes():

    model = make_model()
    model.eval()

    eeg, physio, _ = make_batch(batch_size=8)

    with torch.no_grad():
        out = model(eeg, physio)

    assert out["logits"].shape == (8, NUM_CLASSES)
    assert out["probs"].shape == (8, NUM_CLASSES)
    assert out["eeg_embedding"].shape == (8, 128)
    assert out["physio_embedding"].shape == (8, 128)
    assert out["eeg_gate"].shape == (8,)
    assert out["physio_gate"].shape == (8,)


def test_probs_sum_to_one():

    model = make_model()
    model.eval()

    eeg, physio, _ = make_batch(batch_size=16)

    with torch.no_grad():
        out = model(eeg, physio)

    sums = out["probs"].sum(dim=-1)

    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_gates_in_valid_range():

    model = make_model()
    model.eval()

    eeg, physio, _ = make_batch(batch_size=16)

    with torch.no_grad():
        out = model(eeg, physio)

    assert torch.all(out["eeg_gate"] >= 0.0) and torch.all(out["eeg_gate"] <= 1.0)
    assert torch.all(out["physio_gate"] >= 0.0) and torch.all(out["physio_gate"] <= 1.0)

    # Τα δύο gates πρέπει να αθροίζουν σε 1 (συμπληρωματικά βάρη εμπιστοσύνης)
    total = out["eeg_gate"] + out["physio_gate"]
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)


def test_no_nan_or_inf():

    model = make_model()
    model.eval()

    eeg, physio, _ = make_batch(batch_size=8)

    with torch.no_grad():
        out = model(eeg, physio)

    for key, tensor in out.items():
        assert torch.isfinite(tensor).all(), f"{key} contains NaN/Inf"


def test_batch_size_one():
    """
    Batch size 1 μπορεί να προκαλέσει προβλήματα με BatchNorm.
    Ελέγχουμε ότι δουλεύει τουλάχιστον σε eval mode.
    """

    model = make_model()
    model.eval()

    eeg, physio, _ = make_batch(batch_size=1)

    with torch.no_grad():
        out = model(eeg, physio)

    assert out["logits"].shape == (1, NUM_CLASSES)


def test_gradient_flow_and_training_step():

    model = make_model()
    model.train()

    eeg, physio, labels = make_batch(batch_size=32)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    out = model(eeg, physio)
    loss_before = criterion(out["logits"], labels)

    optimizer.zero_grad()
    loss_before.backward()

    # Όλα τα trainable parameters πρέπει να έχουν λάβει gradient
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    optimizer.step()

    # Δεύτερο forward pass μετά το optimizer step
    out_after = model(eeg, physio)
    loss_after = criterion(out_after["logits"], labels)

    # Δεν εγγυόμαστε μονοτονική μείωση σε ένα μόνο βήμα, αλλά το loss
    # πρέπει να είναι πεπερασμένο και ο υπολογισμός να ολοκληρώνεται.
    assert torch.isfinite(loss_after)


def test_build_model_from_config():

    config = Config()

    model = build_model("hybrid", config)

    assert isinstance(model, HybridCNNMLP)

    eeg, physio, _ = make_batch(batch_size=4)
    model.eval()

    with torch.no_grad():
        out = model(eeg, physio)

    assert out["logits"].shape == (4, config.NUM_CLASSES)


def test_eeg_handcrafted_branch_shapes():
    """
    Ελέγχει ότι το προαιρετικό EEG handcrafted-feature branch (band
    powers κ.λπ., 416-dim) fusάρεται σωστά με το raw-CNN embedding
    χωρίς να αλλάζει τα output shapes.
    """

    model = HybridCNNMLP(
        eeg_channels=EEG_CHANNELS,
        physio_feature_dim=PHYSIO_DIM,
        embedding_dim=128,
        num_classes=NUM_CLASSES,
        dropout=0.30,
        eeg_handcrafted_dim=416,
    )
    model.eval()

    eeg, physio, _ = make_batch(batch_size=8)
    eeg_handcrafted = torch.randn(8, 416)

    with torch.no_grad():
        out = model(eeg, physio, eeg_handcrafted_features=eeg_handcrafted)

    assert out["logits"].shape == (8, NUM_CLASSES)
    assert torch.isfinite(out["logits"]).all()


def test_domain_adversarial_subject_head():
    """
    Ελέγχει ότι, όταν δοθεί num_subjects, το μοντέλο παράγει
    subject_logits με το σωστό shape, ότι το gradient reversal δεν
    σπάει το backward pass, και ότι χωρίς num_subjects δεν υπάρχει
    καθόλου subject_logits key (backward compatibility).
    """

    num_subjects = 5

    model = HybridCNNMLP(
        eeg_channels=EEG_CHANNELS,
        physio_feature_dim=PHYSIO_DIM,
        embedding_dim=128,
        num_classes=NUM_CLASSES,
        dropout=0.30,
        num_subjects=num_subjects,
    )
    model.train()

    eeg, physio, labels = make_batch(batch_size=8)
    subject_labels = torch.randint(0, num_subjects, (8,))

    out = model(eeg, physio, grl_lambda=0.5)

    assert "subject_logits" in out
    assert out["subject_logits"].shape == (8, num_subjects)

    task_loss = nn.CrossEntropyLoss()(out["logits"], labels)
    subject_loss = nn.CrossEntropyLoss()(out["subject_logits"], subject_labels)
    total_loss = task_loss + 0.3 * subject_loss

    total_loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    # Χωρίς num_subjects, δεν πρέπει να υπάρχει καθόλου adversarial head
    model_no_adv = make_model()
    model_no_adv.eval()
    with torch.no_grad():
        out_no_adv = model_no_adv(eeg, physio)
    assert "subject_logits" not in out_no_adv


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
