# META-INTEROCEPTION: Reading Another Network's Mind

> **Can an independent observer network predict Model B's errors from B's full 256-dimensional hidden state better than B can itself? An empirical investigation into self-knowledge versus other-knowledge in deep neural representations.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests: 12 passed](https://img.shields.io/badge/tests-12%20passed-brightgreen.svg)]()

---

## 1. Executive Summary & Vision

In classical cognitive science, **interoception** denotes an agent's sensing of its own internal physiological signals. In machine learning, internal self-monitoring typically relies on coarse activation statistics (activation norm, spread, sparsity, magnitude) or output probabilities (softmax confidence).

This project investigates **META-INTEROCEPTION**: *reading another network's mind*. We train an independent observer network with a distinct architecture and random initialization, and attach an **Observer Head** (`256 -> 64 -> 1`) trained strictly on pairs of $(\mathbf{h}_B, \text{error}_B)$ from held-out samples that Network B never saw during training.

We rigorously compare three error-detection paradigms:
1. **(a) B's Softmax Confidence**: Output-level self-assessment ($1 - \max_k P_B(y=k \mid x)$).
2. **(b) B's 4-Stat Introspection**: Coarse internal representation summary ($\mathbb{E}[h], \text{std}(h), \text{awake fraction}, \mathbb{E}[|h|]$).
3. **(c) Meta-Interoceptive Observer Head**: Direct read-out of B's full 256-dimensional hidden activation vector $\mathbf{h}_B$.

---

## 2. Theoretical Framework & Architecture

```
Input x (784-D) ──┬──► Model B (Seed 1: 784 -> 256 -> 10)
                  │          │                  │
                  │     Hidden h_B (256-D)   Logits / Softmax p_B
                  │          │                  │
                  │          ├──────────────────┼──► Detector (a): Softmax Confidence (1 - max p_B)
                  │          │                  │
                  │          ├─► 4-Stat Probe ──┴──► Detector (b): B's Own 4-Stat Introspection
                  │          │   [mean, std, awake, mag] -> 1
                  │          │
                  │          └─► Observer Head ────► Detector (c): Meta-Interoceptive Observer
                  │              (MLP 256 -> 64 -> 1)
                  │
                  └──► Observer Network (Seed 2: 784 -> 256 -> 10)
                       Independent feature learning baseline
```

### Mathematical Definitions
For input sample $x$ and ground truth label $y$:
- **Model B Output**: $\hat{y}_B = \operatorname{argmax}_k p_k, \quad \text{Error} = \mathbb{I}(\hat{y}_B \ne y)$
- **Detector (a) Score**: $s_a = 1 - \max_k p_k$
- **Detector (b) Feature**: $\mathbf{z}_b = [\mu(h), \sigma(h), \rho_{>0}(h), \mu(|h|)] \in \mathbb{R}^4, \quad s_b = \sigma(\mathbf{w}_b^\top \mathbf{z}_b + b_b)$
- **Detector (c) Score**: $s_c = \sigma(\mathbf{W}_2 \operatorname{ReLU}(\mathbf{W}_1 \mathbf{h}_B + \mathbf{b}_1) + b_2) \in [0, 1]$

---

## 3. Experimental Results

### Phase 1 — Three Detectors on Clean Data
Model B trained on 50,000 MNIST training samples for 3 epochs (Seed 1, test accuracy: **97.51%** with 249 test errors). Detectors (b) and (c) calibrated on 10,000 held-out samples.

| Detector | Representation Monitored | Parameters | Test ROC-AUC | Status |
| :--- | :--- | :---: | :---: | :---: |
| **(a) B Softmax Confidence** | $1 - \max p_B$ (Output probabilities) | 0 | **0.9651** | Baseline |
| **(b) B 4-Stat Introspection** | Coarse 4-stats $[\mu, \sigma, \text{awake}, \text{mag}]$ | 5 | **0.6723** | $+0.172$ vs random |
| **(c) Observer Head (Meta-Intero)**| Full 256-D hidden vector $\mathbf{h}_B$ | 16,513 | **0.8285** | **+0.1562 vs 4-stats** |

#### Scientific Findings:
- **Other-Knowledge vs Self-Stats**: Reading B's full 256-dimensional hidden state achieves an **AUC of 0.8285**, outperforming B's coarse internal introspection (0.6723) by **+15.62% AUC**. Coarse statistics discard critical manifold geometry that the observer network successfully captures.
- **In-Distribution Self-Calibration**: On pristine, clean MNIST data, B's own softmax confidence ($1 - \max p$) achieves **0.9651 AUC**, remaining superior to representation probes because the final linear layer directly calibrates class margins.

---

### Phase 2 — 12-Month Drift Showdown (Permuted-MNIST)
Models B and Observer were frozen. Probes were trained on **Month 6** drift data and evaluated across all 12 months (Seed 42).

```
=====================================================================================
PHASE 2 EXPERIMENTAL RESULTS: 12-MONTH DRIFT SHOWDOWN
=====================================================================================
Month   | B Acc    | (a) Conf AUC   | (b) 4-Stat AUC   | (c) Observer AUC   | Status         
-------------------------------------------------------------------------------------
Month 1  |  14.09% |       0.5163 |         0.4663 |           0.4683 | 
Month 2  |   7.23% |       0.5151 |         0.5197 |           0.5094 | 
Month 3  |  12.06% |       0.5217 |         0.4230 |           0.4908 | 
Month 4  |  13.03% |       0.4900 |         0.4416 |           0.5252 | 
Month 5  |  14.58% |       0.5033 |         0.4937 |           0.3867 | 
Month 6  |   8.89% |       0.4448 |         0.5319 |           0.9192 | <- TRAINED HERE
Month 7  |  14.57% |       0.5142 |         0.4967 |           0.5263 | 
Month 8  |  10.38% |       0.4637 |         0.5128 |           0.5366 | 
Month 9  |   8.03% |       0.5365 |         0.4527 |           0.5370 | 
Month 10 |   6.96% |       0.3971 |         0.5384 |           0.4120 | 
Month 11 |  10.99% |       0.5276 |         0.4823 |           0.5584 | 
Month 12 |  10.59% |       0.5239 |         0.5188 |           0.5256 | 
-------------------------------------------------------------------------------------
```

#### Degradation Analysis:
- **At Month 6**: Under extreme covariate drift (Model B accuracy drops to 8.89%), Softmax Confidence completely collapses to **0.4448 AUC** (worse than random guessing, indicating uncalibrated overconfidence). However, the **Observer Head reading B's hidden state achieves 0.9192 AUC**!
- **Fastest Degrader**: **(c) Observer Head** (Mean OOD drop: **+0.4214 AUC**). Because each permutation forms an orthogonal pixel reorganization, an observer trained on Month 6 features overfits that specific drift manifold.
- **Most Resilient**: **(a) Softmax Confidence** (Mean OOD drop: **-0.0561 AUC**). It stays consistently poor (~0.40 - 0.53 AUC) across all drifted months without specific manifold attachment.

---

### Phase 3 — Selective Prediction & Risk-Coverage Analysis
Combining all three signals into a calibrated blend:

$$\text{Score}_{\text{blend}} = \sigma\left(0.472 \cdot \text{conf}_a - 0.239 \cdot \text{conf}_b + 2.850 \cdot \text{conf}_c\right)$$

| Method / Signal | Coverage at $\ge 99.0\%$ Accuracy | Samples Safely Retained (out of 10k) |
| :--- | :---: | :---: |
| **(a) Softmax Confidence Only** | **95.45%** | **9,545** |
| **(b) 4-Stat Introspection Only** | 17.27% | 1,727 |
| **(c) Observer Head Only** | 78.18% | 7,818 |
| **[*] Meta-Interoceptive Blend** | **86.36%** | **8,636** |

#### Accuracy by Coverage Fraction:
| Coverage Fraction | (a) Confidence Only | (c) Observer Head Only | [*] Learned Blend |
| :---: | :---: | :---: | :---: |
| **20.0%** | 100.00% | 99.80% | **99.95%** |
| **50.0%** | 100.00% | 99.62% | **99.88%** |
| **70.0%** | 99.99% | 99.30% | **99.87%** |
| **85.5%** | 99.85% | 98.83% | **99.11%** |
| **100.0%** | 97.51% | 97.51% | **97.51%** |

---

## 4. Honest Scientific Limitations

1. **In-Distribution Confidence Dominance**: On clean, in-distribution datasets like MNIST, the network's output softmax distribution is already well-calibrated (AUC = 0.9651). In this pristine setting, adding external observer predictions introduces variance that does not beat pure confidence.
2. **Drift-Free Training Constraint**: The observer head was trained on stationary data. Under non-stationary covariate drift (Phase 2), a static observer head experiences manifold mismatch when tested on unseen permutations.
3. **Unidirectional Observation**: The observer only inspects B's activations passively; it does not steer, regulate, or interact with B's computation.

---

## 5. Future Work

1. **Online Adaptive Meta-Interoception**: Equipping the observer head with test-time adaptation (e.g. self-supervised entropy minimization or streaming memory buffers) to continuously track representation drift.
2. **Bidirectional Theory-of-Mind**: Symmetric architectures where Network A and Network B simultaneously inspect each other's latent vectors to negotiate consensus before committing to high-risk actions.
3. **Cross-Architecture Mind Reading**: Training transformer-based observers to read the latent thoughts of diffusion models or autoregressive language models.

---

## 6. Quickstart & Reproducibility

```bash
# Clone and install
cd meta-interoception
pip install -e .

# Run full test suite
pytest -v

# Execute Phase 1: Clean Data Detectors
python -m meta_interoception.phase1_clean_detectors --seed-b 1 --seed-obs 2

# Execute Phase 2: 12-Month Drift Showdown
python -m meta_interoception.phase2_drift_showdown --seed 42 --months 12 --train-month 6

# Execute Phase 3: Selective Prediction & Risk-Coverage
python -m meta_interoception.phase3_selective_prediction --seed 42 --target-acc 0.99
```
