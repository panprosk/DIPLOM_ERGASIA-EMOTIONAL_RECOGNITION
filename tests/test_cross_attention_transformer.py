"""
test_cross_attention_transformer.py

Unit tests για το Multimodal Cross-Attention Transformer (2ο μοντέλο
της διπλωματικής, βλ. ΣΕΙΡΑ_ΥΛΟΠΟΙΗΣΗΣ.txt Βήμα 12).

Ελέγχουν:
- Σωστά output shapes (logits, probs, embeddings, cross-attention weight)
- probs να αθροίζουν σε 1 (softmax)
- Να μην υπάρχουν NaN/Inf στην έξοδο
- Ροή gradients (backward + optimizer step)
- Ανεξαρτησία από το batch size (π.χ. batch_size=1)
- Σωστή δημιουργία μέσω build_model(config)
- Domain-adversarial subject classifier (προαιρετικό head)

Τρέξιμο:
    pytest tests/test_cross_attention_transformer.py -v
"""

import torch
import torch.nn as nn
import pytest

from src.models import CrossAttentionTransformer, build_model
from src.config.config import Config


EEG_CHANNELS = 32
WINDOW_SAMPLES = 768
NUM_CLASSES = 2


def make_model(**kwargs):
    defaults = dict(
        eeg_channels=EEG_CHANNELS,
        physio_channels=2,
        d_model=64,
        num_heads=4,
        num_self_attn_layers=2,
        num_cross_attn_layers=1,
        dim_feedforward=128,
        patch_size=32,
        num_classes=NUM_CLASSES,
        dropout=0.30,
    )
    defaults.update(kwargs)
    return CrossAttentionTransformer(**defaults)


def make_batch(batch_size=8):
    torch.manual_seed(0)
    eeg = torch.randn(batch_size, EEG_CHANNELS, WINDOW_SAMPLES)
    eda = torch.randn(batch_size, 1, WINDOW_SAMPLES)
    ppg = torch.randn(batch_size, 1, WINDOW_SAMPLES)
    labels = torch.randint(0, NUM_CLASSES, (batch_size,))
    return eeg, eda, ppg, labels


def test_output_shapes():

    model = make_model()
    model.eval()

    eeg, eda, ppg, _ = make_batch(batch_size=8)

    with torch.no_grad():
        out = model(eeg, eda, ppg)

    assert out["logits"].shape == (8, NUM_CLASSES)
    assert out["probs"].shape == (8, NUM_CLASSES)
    assert out["eeg_embedding"].shape == (8, 64)
    assert out["physio_embedding"].shape == (8, 64)
    assert out["cross_attention_weight"].shape == (8,)


def test_probs_sum_to_one():

    model = make_model()
    model.eval()

    eeg, eda, ppg, _ = make_batch(batch_size=16)

    with torch.no_grad():
        out = model(eeg, eda, ppg)

    sums = out["probs"].sum(dim=-1)

    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_no_nan_or_inf():

    model = make_model()
    model.eval()

    eeg, eda, ppg, _ = make_batch(batch_size=8)

    with torch.no_grad():
        out = model(eeg, eda, ppg)

    for key, tensor in out.items():
        assert torch.isfinite(tensor).all(), f"{key} contains NaN/Inf"


def test_batch_size_one():

    model = make_model()
    model.eval()

    eeg, eda, ppg, _ = make_batch(batch_size=1)

    with torch.no_grad():
        out = model(eeg, eda, ppg)

    assert out["logits"].shape == (1, NUM_CLASSES)


def test_gradient_flow_and_training_step():

    model = make_model()
    model.train()

    eeg, eda, ppg, labels = make_batch(batch_size=32)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    out = model(eeg, eda, ppg)
    loss_before = criterion(out["logits"], labels)

    optimizer.zero_grad()
    loss_before.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    optimizer.step()

    out_after = model(eeg, eda, ppg)
    loss_after = criterion(out_after["logits"], labels)

    assert torch.isfinite(loss_after)


def test_build_model_from_config():

    config = Config()
    config.MODEL_NAME = "cross_attention_transformer"

    model = build_model("cross_attention_transformer", config)

    assert isinstance(model, CrossAttentionTransformer)

    eeg, eda, ppg, _ = make_batch(batch_size=4)
    model.eval()

    with torch.no_grad():
        out = model(eeg, eda, ppg)

    assert out["logits"].shape == (4, config.NUM_CLASSES)


def test_domain_adversarial_subject_head():
    """
    Ελέγχει ότι, όταν δοθεί num_subjects, το μοντέλο παράγει
    subject_logits με το σωστό shape και ότι το backward pass δεν
    σπάει (gradient reversal). Χωρίς num_subjects δεν πρέπει να
    υπάρχει καθόλου subject_logits key (backward compatibility).
    """

    num_subjects = 5

    model = make_model(num_subjects=num_subjects)
    model.train()

    eeg, eda, ppg, labels = make_batch(batch_size=8)
    subject_labels = torch.randint(0, num_subjects, (8,))

    out = model(eeg, eda, ppg, grl_lambda=0.5)

    assert "subject_logits" in out
    assert out["subject_logits"].shape == (8, num_subjects)

    task_loss = nn.CrossEntropyLoss()(out["logits"], labels)
    subject_loss = nn.CrossEntropyLoss()(out["subject_logits"], subject_labels)
    total_loss = task_loss + 0.3 * subject_loss

    total_loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    model_no_adv = make_model()
    model_no_adv.eval()
    with torch.no_grad():
        out_no_adv = model_no_adv(eeg, eda, ppg)
    assert "subject_logits" not in out_no_adv


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
