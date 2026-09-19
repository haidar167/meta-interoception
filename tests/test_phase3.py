"""Tests for Phase 3: selective prediction and risk-coverage curves."""

import numpy as np
import pytest

from meta_interoception.metrics import compute_risk_coverage, find_coverage_at_accuracy


def test_compute_risk_coverage_shapes():
    y_correct = np.array([1, 1, 1, 1, 0, 1, 1, 0, 1, 0])
    confs = np.array([0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45])

    covs, accs = compute_risk_coverage(y_correct, confs, min_coverage=0.2, num_points=5)
    assert len(covs) == 5
    assert len(accs) == 5
    assert covs[0] == 0.2
    assert covs[-1] == 1.0
    # Top 20% is samples 0 and 1 -> both correct -> 100%
    assert accs[0] == 1.0


def test_find_coverage_at_target_accuracy():
    covs = np.array([0.1, 0.3, 0.5, 0.7, 1.0])
    accs = np.array([1.0, 0.995, 0.991, 0.985, 0.95])

    cov_99 = find_coverage_at_accuracy(covs, accs, target_accuracy=0.99)
    # Target 0.99 is met at 0.1, 0.3, 0.5 (acc 0.991) -> max is 0.5
    assert cov_99 == 0.5

    # Unattainable accuracy
    cov_impossible = find_coverage_at_accuracy(covs, accs, target_accuracy=1.01)
    assert cov_impossible == 0.0
