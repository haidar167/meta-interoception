"""Phase 3: Selective Prediction — Risk-Coverage curves and Learned Blending."""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

from .data import load_mnist_datasets
from .metrics import compute_risk_coverage, find_coverage_at_accuracy
from .models import MLP256, ObserverHead, extract_features_and_errors
from .phase1_clean_detectors import train_observer_head
from .utils import save_results, set_seed


def run_phase3(
    seed: int = 42,
    target_accuracy: float = 0.99,
    epochs_head: int = 20,
    output_dir: str = "results",
    figure_dir: str = "figures",
    checkpoint_dir: str = "checkpoints",
) -> Dict[str, Any]:
    """Execute Phase 3 selective prediction and risk-coverage benchmark."""
    set_seed(seed)
    print("=" * 85)
    print("PHASE 3: SELECTIVE PREDICTION — RISK-COVERAGE SHOWDOWN & LEARNED BLEND")
    print("=" * 85)

    # 1. Load datasets and trained Model B
    _, held_out_ds, test_ds, input_dim = load_mnist_datasets(seed=seed)
    ckpt_b = Path(checkpoint_dir) / "model_b.pt"
    if not ckpt_b.exists():
        raise FileNotFoundError(f"Model B checkpoint not found at {ckpt_b}. Run Phase 1 first.")

    model_b = MLP256(input_dim=input_dim)
    model_b.load_state_dict(torch.load(ckpt_b, map_location="cpu"))
    model_b.eval()

    held_out_loader = DataLoader(held_out_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    # 2. Extract signals on Held-Out set to train 4-stat probe, observer head, and blender
    print("\nExtracting calibration signals on Held-out set (10,000 samples)...")
    calib_data = extract_features_and_errors(model_b, held_out_loader)
    calib_h = calib_data["h"]
    calib_stats = calib_data["stats"].numpy()
    calib_errors = calib_data["errors"].numpy()
    calib_correct = (calib_errors == 0).astype(int)

    # (a) Confidence score
    conf_calib_a = calib_data["conf"].numpy()

    # (b) Fit 4-stat probe
    print("Fitting Detector (b): 4-Stat Probe...")
    probe_b = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    probe_b.fit(calib_stats, calib_errors)
    # Correctness confidence from probe b: 1 - P(error)
    conf_calib_b = probe_b.predict_proba(calib_stats)[:, 0]

    # (c) Train Observer Head
    print("Training Detector (c): Observer Head (256 -> 64 -> 1)...")
    obs_head = train_observer_head(calib_h, calib_data["errors"], epochs=epochs_head, lr=1e-3, seed=seed)
    with torch.no_grad():
        p_err_c = obs_head.predict_proba(calib_h).squeeze(1).numpy()
    conf_calib_c = 1.0 - p_err_c

    # 3. Train Learned Blender
    # Stack [conf_a, conf_b, conf_c] to predict correctness
    X_blend_calib = np.stack([conf_calib_a, conf_calib_b, conf_calib_c], axis=1)
    print("\nTraining Learned Blender on [Conf_a, Conf_b, Conf_c] to predict sample correctness...")
    blender = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=seed, max_iter=1000, C=1.0)
    )
    blender.fit(X_blend_calib, calib_correct)
    blend_weights = blender.named_steps["logisticregression"].coef_[0].tolist()
    print(f"Learned Blender Coefficients: Conf_a={blend_weights[0]:.3f}, Conf_b={blend_weights[1]:.3f}, Conf_c={blend_weights[2]:.3f}")

    # 4. Evaluate on Test Set (10,000 samples)
    print("\nEvaluating Selective Prediction on independent Test Set (10,000 samples)...")
    test_data = extract_features_and_errors(model_b, test_loader)
    test_correct = (test_data["errors"].numpy() == 0).astype(int)
    base_acc = float(np.mean(test_correct))

    # Test confidence scores
    conf_test_a = test_data["conf"].numpy()
    conf_test_b = probe_b.predict_proba(test_data["stats"].numpy())[:, 0]
    with torch.no_grad():
        conf_test_c = 1.0 - obs_head.predict_proba(test_data["h"]).squeeze(1).numpy()

    X_blend_test = np.stack([conf_test_a, conf_test_b, conf_test_c], axis=1)
    conf_test_blend = blender.predict_proba(X_blend_test)[:, 1]

    # 5. Compute Risk-Coverage Curves
    covs, accs_a = compute_risk_coverage(test_correct, conf_test_a, min_coverage=0.10, num_points=100)
    _, accs_b = compute_risk_coverage(test_correct, conf_test_b, min_coverage=0.10, num_points=100)
    _, accs_c = compute_risk_coverage(test_correct, conf_test_c, min_coverage=0.10, num_points=100)
    _, accs_blend = compute_risk_coverage(test_correct, conf_test_blend, min_coverage=0.10, num_points=100)

    # 6. Find Coverage at Target Accuracy (99.0%)
    cov_at_99_a = find_coverage_at_accuracy(covs, accs_a, target_accuracy=target_accuracy)
    cov_at_99_b = find_coverage_at_accuracy(covs, accs_b, target_accuracy=target_accuracy)
    cov_at_99_c = find_coverage_at_accuracy(covs, accs_c, target_accuracy=target_accuracy)
    cov_at_99_blend = find_coverage_at_accuracy(covs, accs_blend, target_accuracy=target_accuracy)

    cov_gain = (cov_at_99_blend - cov_at_99_a) * 100

    print("\n" + "=" * 85)
    print("PHASE 3 EXPERIMENTAL RESULTS: SELECTIVE PREDICTION BENCHMARK")
    print("=" * 85)
    print(f"Base Model B Test Accuracy (100% Coverage): {base_acc*100:.2f}%\n")
    print(f"{'Method / Signal':<35} | {'Coverage at >= 99.0% Accuracy':<30} | {'Samples Retained (out of 10k)':<25}")
    print("-" * 85)
    print(f"{'(a) Softmax Confidence Only':<35} | {cov_at_99_a*100:>26.2f}% | {int(round(cov_at_99_a*10000)):>22,}")
    print(f"{'(b) 4-Stat Introspection Only':<35} | {cov_at_99_b*100:>26.2f}% | {int(round(cov_at_99_b*10000)):>22,}")
    print(f"{'(c) Observer Head Only':<35} | {cov_at_99_c*100:>26.2f}% | {int(round(cov_at_99_c*10000)):>22,}")
    print(f"{'[*] Meta-Interoceptive Blend':<35} | {cov_at_99_blend*100:>26.2f}% | {int(round(cov_at_99_blend*10000)):>22,}")
    print("-" * 85)
    print(f"Coverage Gain (Blend vs Confidence): {cov_gain:+.2f}% (+{int(round(cov_gain*100))} additional safe predictions)")

    # Sample coverage checkpoints
    print("\nAccuracy by Selected Coverage Fraction:")
    print(f"{'Coverage Fraction':<20} | {'(a) Confidence':<16} | {'(c) Observer Head':<18} | {'[*] Learned Blend':<16}")
    print("-" * 75)
    for target_cov in [0.20, 0.50, 0.70, 0.85, 0.90, 0.95, 1.00]:
        idx = np.argmin(np.abs(covs - target_cov))
        print(
            f"{covs[idx]*100:>15.1f}% | {accs_a[idx]*100:>13.2f}% | "
            f"{accs_c[idx]*100:>15.2f}% | {accs_blend[idx]*100:>13.2f}%"
        )

    # 7. Visualization: figures/phase3_risk_coverage.png (dpi=150)
    fig_dir = Path(figure_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "phase3_risk_coverage.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # Plot 1: Risk-Coverage Curve
    axes[0].plot(covs * 100, accs_a * 100, color="#1f77b4", lw=2, label=f"(a) Confidence (Cov@99%: {cov_at_99_a*100:.1f}%)")
    axes[0].plot(covs * 100, accs_c * 100, color="#d62728", lw=2, linestyle="--", label=f"(c) Observer Head (Cov@99%: {cov_at_99_c*100:.1f}%)")
    axes[0].plot(covs * 100, accs_blend * 100, color="#2ca02c", lw=2.5, label=f"[*] Blend (Cov@99%: {cov_at_99_blend*100:.1f}%)")
    axes[0].axhline(y=target_accuracy * 100, color="grey", linestyle=":", lw=1.5, label="99% Target Accuracy")

    axes[0].set_title("Risk-Coverage Tradeoff Curves", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Coverage (% of test data retained)", fontsize=10)
    axes[0].set_ylabel("Selective Accuracy (%)", fontsize=10)
    axes[0].set_ylim([95.0, 100.2])
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower left", fontsize=9)

    # Plot 2: Bar chart of Coverage at 99%
    methods = ["(a) Confidence", "(b) 4-Stat", "(c) Observer", "[*] Blend"]
    cov_values = [cov_at_99_a * 100, cov_at_99_b * 100, cov_at_99_c * 100, cov_at_99_blend * 100]
    colors = ["#1f77b4", "#7f7f7f", "#d62728", "#2ca02c"]

    bars = axes[1].bar(methods, cov_values, color=colors, width=0.55)
    axes[1].set_title("Coverage Achievable at >= 99.0% Accuracy", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Coverage (%)", fontsize=10)
    axes[1].set_ylim([0, 105])
    axes[1].grid(True, linestyle=":", alpha=0.5, axis="y")

    for bar in bars:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\nSaved risk-coverage figure: {fig_path}")

    # 8. Save JSON results
    res_dir = Path(output_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    res_path = res_dir / "phase3.json"

    results_payload = {
        "target_accuracy": target_accuracy,
        "base_accuracy": base_acc,
        "coverage_at_99_pct": {
            "confidence_only": float(cov_at_99_a),
            "self_stats_only": float(cov_at_99_b),
            "observer_only": float(cov_at_99_c),
            "blend": float(cov_at_99_blend),
            "absolute_gain": float(cov_gain / 100.0),
        },
        "blender_coefficients": {
            "w_confidence": float(blend_weights[0]),
            "w_self_stats": float(blend_weights[1]),
            "w_observer": float(blend_weights[2]),
        },
        "headline": (
            f"The meta-interoceptive blend unlocks {cov_at_99_blend*100:.1f}% coverage at >= 99.0% accuracy, "
            f"compared to {cov_at_99_a*100:.1f}% for confidence-only "
            f"(a gain of {cov_gain:+.2f}%, allowing {int(round(cov_gain*100))} additional samples to be safely automated)."
        ),
    }
    saved_data = save_results(results_payload, res_path, seed=seed)
    print(f"Saved results: {res_path}")

    return saved_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-acc", type=float, default=0.99)
    parser.add_argument("--epochs-head", type=int, default=20)
    args = parser.parse_args()
    run_phase3(seed=args.seed, target_accuracy=args.target_acc, epochs_head=args.epochs_head)
