"""Tests for Phase 1: data partitioning and detector training."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from meta_interoception.data import load_mnist_datasets
from meta_interoception.models import MLP256, ObserverHead, extract_features_and_errors
from meta_interoception.phase1_clean_detectors import train_observer_head


def test_dataset_splits():
    b_tr, held_out, te, dim = load_mnist_datasets(b_train_size=100, held_out_size=50, seed=42)
    assert len(b_tr) == 100
    assert len(held_out) == 50
    assert len(te) > 0


def test_extract_features_and_errors():
    model = MLP256(input_dim=64, hidden_dim=32, num_classes=10)
    x = torch.randn(20, 64)
    y = torch.randint(0, 10, (20,))
    loader = DataLoader(TensorDataset(x, y), batch_size=10)

    extracted = extract_features_and_errors(model, loader)
    assert extracted["h"].shape == (20, 32)
    assert extracted["stats"].shape == (20, 4)
    assert extracted["probs"].shape == (20, 10)
    assert extracted["conf"].shape == (20,)
    assert extracted["conf_error_score"].shape == (20,)
    assert extracted["errors"].shape == (20,)


def test_train_observer_head_smoke():
    h = torch.randn(30, 32)
    err = torch.tensor([0] * 25 + [1] * 5)
    head = train_observer_head(h, err, epochs=2, batch_size=10, seed=42)
    probs = head.predict_proba(h)
    assert probs.shape == (30, 1)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
