import argparse

import numpy as np
import torch

from src.models.ct_elcan import CTELCAN, ConditionalDiscriminator, ConditionalGenerator
from train_transformer_ctelcan import (
    make_labels, split_modalities, summarize, train_fold,
)


def _args(**kw):
    base = dict(
        target="valence", window=128, epochs=1, batch_size=16, lr=1e-4, gan_lr=1e-4,
        dropout=0.1, layers=2, heads=4, dim=32, grad_clip=10.0, weight_decay=0.0,
        noise_std=0.0, select_val=False, alpha=0.1, beta=0.3,
        gamma=1.0, phi=0.1, grl_max=1.0, gan_warmup_epochs=0, no_gan=False,
        no_adv=False, log_test_every=1,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_model_shapes():
    model = CTELCAN((32, 1, 1), 128, 2, 5, dim=32, layers=2, heads=4)
    x = torch.randn(6, 34, 128)
    feats = model.encode(split_modalities(x))
    assert len(feats) == 3 and feats[0].shape == (6, 32, 128)
    out = model.lcan(feats)
    assert out["emotion"].shape == (6, 2)
    assert out["subject"].shape == (6, 5)
    assert out["modality"].shape == (18, 3)
    assert model(split_modalities(x)).shape == (6, 2)

    gen = ConditionalGenerator(32, 2)
    disc = ConditionalDiscriminator(32, 2)
    y = torch.randint(0, 2, (6,))
    fake = gen(feats[0], y)
    assert fake.shape == feats[0].shape
    assert disc(fake, y).shape == (6,)


def test_train_fold_smoke():
    rng = np.random.default_rng(0)
    n_sub, trials, wins = 6, 4, 3
    x, subj, trial, ratings = [], [], [], []
    for s in range(n_sub):
        for t in range(trials):
            for _ in range(wins):
                x.append(rng.standard_normal((34, 128)).astype(np.float32))
                subj.append(s); trial.append(t)
                ratings.append([float(t % 2) * 4 + 3, 5.0])
    data = {
        "x": np.stack(x), "subject": np.asarray(subj), "trial": np.asarray(trial),
        "ratings": np.asarray(ratings, dtype=np.float32),
    }
    names = [f"s{i:02d}" for i in range(n_sub)]
    labels = make_labels(data["ratings"], "valence")
    fold = {"fold": 0, "train": names[:3], "validation": names[3:5], "test": names[5:]}
    res = train_fold(fold, data, labels, names, _args(), torch.device("cpu"), None)
    assert res["probabilities"].shape == (trials * wins, 2)
    summary = summarize([res], 2, names)
    assert 0.0 <= summary["subject_window_acc_mean"] <= 1.0
