"""Unit tests for utils, metrics, and models in meta-interoception."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from meta_interoception.metrics import (
    compute_auc,
    compute_classification_metrics,
    compute_risk_coverage,
    find_coverage_at_accuracy,
)
from meta_interoception.models import MLP256, ObserverHead, SelfStatsProbe, extract_4_stats
from meta_interoception.utils import get_git_commit_hash, save_results, set_seed


def test_set_seed_reproducibility():
    set_seed(123)
    t1 = torch.randn(4, 4)
    set_seed(123)
    t2 = torch.randn(4, 4)
    assert torch.equal(t1, t2)


def test_compute_auc():
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.2, 0.7, 0.9]
    auc = compute_auc(y_true, y_scores)
    assert auc == 1.0

    # Single class degenerate case
    auc_single = compute_auc([0, 0], [0.1, 0.2])
    assert auc_single == 0.5


def test_risk_coverage_metrics():
    # 8 correct, 2 errors
    y_correct = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    conf = [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.50, 0.40]

    covs, accs = compute_risk_coverage(y_correct, conf, min_coverage=0.1, num_points=10)
    assert len(covs) == 10
    assert len(accs) == 10
    assert accs[0] == 1.0  # Top 10% is 100% accurate

    cov_99 = find_coverage_at_accuracy(covs, accs, target_accuracy=0.99)
    assert cov_99 > 0.0


def test_extract_4_stats():
    h = torch.tensor([[1.0, 2.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]])
    stats = extract_4_stats(h)
    assert stats.shape == (2, 4)
    # Row 0: mean=(1+2+0-1)/4=0.5, awake=(1>0 + 2>0)/4 = 0.5, mag=(1+2+0+1)/4 = 1.0
    assert pytest.approx(stats[0, 0].item(), 0.01) == 0.5
    assert pytest.approx(stats[0, 2].item(), 0.01) == 0.5
    assert pytest.approx(stats[0, 3].item(), 0.01) == 1.0
    # Row 1: all 0
    assert stats[1, 2].item() == 0.0


def test_model_and_probes_forward():
    model = MLP256(input_dim=64, hidden_dim=32, num_classes=10)
    x = torch.randn(5, 64)
    logits, h = model.forward_with_hidden(x)
    assert logits.shape == (5, 10)
    assert h.shape == (5, 32)

    obs_head = ObserverHead(input_dim=32, hidden_dim=16)
    out_obs = obs_head(h)
    prob_obs = obs_head.predict_proba(h)
    assert out_obs.shape == (5, 1)
    assert prob_obs.shape == (5, 1)
    assert (prob_obs >= 0.0).all() and (prob_obs <= 1.0).all()

    self_probe = SelfStatsProbe(input_dim=4, hidden_dim=8)
    stats = extract_4_stats(h)
    prob_self = self_probe.predict_proba(stats)
    assert prob_self.shape == (5, 1)
