"""Tests for Phase 2: Permuted-MNIST drift and probe evaluation."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from meta_interoception.data import get_permuted_dataset
from meta_interoception.models import MLP256, extract_features_and_errors


def test_permuted_dataset_diversity():
    x = torch.randn(10, 64)
    y = torch.randint(0, 10, (10,))
    base_ds = TensorDataset(x, y)

    ds_m1 = get_permuted_dataset(base_ds, permutation_seed=42 + 1 * 17, input_dim=64)
    ds_m2 = get_permuted_dataset(base_ds, permutation_seed=42 + 2 * 17, input_dim=64)

    x1, _ = ds_m1[0]
    x2, _ = ds_m2[0]
    # Permutations should produce different pixel re-orderings
    assert not torch.equal(x1, x2)


def test_model_evaluation_under_permutation():
    model = MLP256(input_dim=64, hidden_dim=32, num_classes=10)
    x = torch.randn(20, 64)
    y = torch.randint(0, 10, (20,))
    base_ds = TensorDataset(x, y)
    perm_ds = get_permuted_dataset(base_ds, permutation_seed=999, input_dim=64)

    loader = DataLoader(perm_ds, batch_size=10)
    extracted = extract_features_and_errors(model, loader)
    assert extracted["h"].shape == (20, 32)
    assert extracted["conf_error_score"].shape == (20,)
    assert len(extracted["errors"]) == 20
