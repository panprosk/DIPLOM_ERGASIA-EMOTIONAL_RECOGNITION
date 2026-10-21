import numpy as np
import torch

from src.config.config import Config
from src.models.model_factory import build_model
from src.models.hybrid_cnn_mlp_v2 import HybridCNNMLPv2
from src.models.hybrid_cnn_mlp import HybridCNNMLP
from src.preprocessing.deap_official_v2 import (
    OfficialDeapV2Preprocessor,
    EEG_FEATURE_DIM,
    PHYSIO_FEATURE_DIM,
)


def _synthetic_trials(n_trials=2, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(8064) / 128.0
    data = rng.normal(size=(n_trials, 40, 8064)).astype(np.float32) * 10.0
    data[:, 36] = 5000 + 300 * np.sin(2 * np.pi * 0.05 * t) + rng.normal(size=8064)
    data[:, 38] = 3000 + 800 * np.sin(2 * np.pi * 1.2 * t) + rng.normal(size=8064)
    labels = np.array([[7.0, 5, 5, 5], [3.0, 5, 5, 5]])[:n_trials]
    return data, labels


def test_v2_preprocessing_shapes_and_values():
    data, labels = _synthetic_trials()
    windows = OfficialDeapV2Preprocessor().process_subject("s99", data, labels)

    assert len(windows) == 2 * 37
    w = windows[0]
    assert w["eeg"].shape == (32, 768)
    assert w["eda"].shape == (1, 768) and w["ppg"].shape == (1, 768)
    assert w["features"].shape == (EEG_FEATURE_DIM + PHYSIO_FEATURE_DIM,)
    assert w["physio_features"].shape == (PHYSIO_FEATURE_DIM,)
    assert np.isfinite(np.stack([x["features"] for x in windows])).all()
    assert {x["label"] for x in windows if x["trial_id"] == 0} == {1}
    assert {x["label"] for x in windows if x["trial_id"] == 1} == {0}


def test_v2_label_threshold_five_is_negative():
    data, _ = _synthetic_trials(1)
    windows = OfficialDeapV2Preprocessor().process_subject(
        "s99", data, np.array([[5.0, 5, 5, 5]])
    )
    assert windows[0]["label"] == 0


def test_v2_relative_power_sums_to_one_per_channel():
    data, labels = _synthetic_trials(1)
    w = OfficialDeapV2Preprocessor().process_subject("s99", data, labels)[0]
    relative = w["features"][128:256].reshape(32, 4)
    assert np.allclose(relative.sum(axis=1), 1.0, atol=1e-4)


def test_v2_model_forward_backward_and_factory():
    config = Config()
    config.HYBRID_VERSION = "v2"
    config.EEG_HANDCRAFTED_DIM = 256
    config.EMBEDDING_DIM = 96
    config.DROPOUT = 0.30

    model = build_model("hybrid", config, num_subjects=5)
    assert isinstance(model, HybridCNNMLPv2)

    out = model(
        torch.randn(4, 32, 768), torch.randn(4, 22), torch.randn(4, 256), grl_lambda=0.5
    )
    assert out["logits"].shape == (4, 2)
    assert out["subject_logits"].shape == (4, 5)
    assert out["contrastive_embedding"].shape == (4, 64)
    assert ((out["eeg_gate"] >= 0) & (out["eeg_gate"] <= 1)).all()
    assert torch.allclose(out["eeg_gate"] + out["physio_gate"], torch.ones(4))

    out["logits"].sum().backward()
    assert all(p.grad is not None for p in model.eeg_cnn.parameters())


def test_default_config_still_builds_v1_model():
    assert isinstance(build_model("hybrid", Config()), HybridCNNMLP)
