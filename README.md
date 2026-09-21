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

### Mathematical Formulation

Let input sample $\mathbf{x} \in \mathbb{R}^D$ ($D=784$) with ground-truth categorical label $y \in \{0, 1, \dots, K-1\}$ ($K=10$).

#### Model B Forward Mapping
Model B is parameterized by $\theta_B = \{\mathbf{W}_B^{(1)} \in \mathbb{R}^{H \times D}, \mathbf{b}_B^{(1)} \in \mathbb{R}^H, \mathbf{W}_B^{(2)} \in \mathbb{R}^{K \times H}, \mathbf{b}_B^{(2)} \in \mathbb{R}^K\}$ with $H=256$:
1. **Hidden Representation**:
   $$\mathbf{h}_B(\mathbf{x}) = \operatorname{ReLU}\left(\mathbf{W}_B^{(1)} \mathbf{x} + \mathbf{b}_B^{(1)}\right) \in \mathbb{R}_{\ge 0}^H$$
2. **Logit Vector & Softmax Distribution**:
   $$\mathbf{z}_B(\mathbf{x}) = \mathbf{W}_B^{(2)} \mathbf{h}_B(\mathbf{x}) + \mathbf{b}_B^{(2)} \in \mathbb{R}^K$$
   $$\mathbf{p}_B(\mathbf{x}) = \operatorname{softmax}\left(\mathbf{z}_B(\mathbf{x})\right) \in \Delta^{K-1}, \quad p_{B,k}(\mathbf{x}) = \frac{\exp(z_{B,k})}{\sum_{j=0}^{K-1} \exp(z_{B,j})}$$
3. **Prediction & Binary Classification Error**:
   $$\hat{y}_B(\mathbf{x}) = \operatorname{argmax}_{k \in \{0,\dots,K-1\}} p_{B,k}(\mathbf{x}), \quad e_B(\mathbf{x}, y) = \mathbb{I}\left(\hat{y}_B(\mathbf{x}) \ne y\right) \in \{0, 1\}$$

---

#### The Three Error Detectors

##### Detector (a) — Softmax Confidence Error Score
Unsupervised baseline using Model B's output probability dispersion:
- Confidence: $\kappa_a(\mathbf{x}) = \max_{k} p_{B,k}(\mathbf{x}) = \|\mathbf{p}_B(\mathbf{x})\|_\infty \in [1/K, 1]$
- Predicted error score:
  $$s_a(\mathbf{x}) = 1 - \kappa_a(\mathbf{x}) = 1 - \max_{k \in \{0,\dots,K-1\}} p_{B,k}(\mathbf{x}) \in [0, 1 - 1/K]$$

##### Detector (b) — 4-Stat Internal Introspection Probe
Extracts 4 summary statistics of internal hidden activations $\mathbf{h}_B = [h_{B,1}, \dots, h_{B,H}]^\top \in \mathbb{R}^H$:
$$\mathbf{z}_b(\mathbf{x}) = \begin{bmatrix}
\mu(\mathbf{h}_B) \\
\sigma(\mathbf{h}_B) \\
\rho_{>0}(\mathbf{h}_B) \\
\nu(\mathbf{h}_B)
\end{bmatrix} \in \mathbb{R}^4, \quad \text{where} \quad \begin{cases}
\mu(\mathbf{h}_B) = \frac{1}{H}\sum_{d=1}^H h_{B,d} & \text{(mean activation)} \\
\sigma(\mathbf{h}_B) = \sqrt{\frac{1}{H}\sum_{d=1}^H (h_{B,d} - \mu(\mathbf{h}_B))^2} & \text{(activation spread / std)} \\
\rho_{>0}(\mathbf{h}_B) = \frac{1}{H}\sum_{d=1}^H \mathbb{I}(h_{B,d} > 0) & \text{(fraction of active neurons)} \\
\nu(\mathbf{h}_B) = \frac{1}{H}\sum_{d=1}^H |h_{B,d}| & \text{(mean absolute magnitude)}
\end{cases}$$
*(Note: Since $\mathbf{h}_B = \operatorname{ReLU}(\cdot) \ge 0$, $\nu(\mathbf{h}_B) \equiv \mu(\mathbf{h}_B)$ post-activation, highlighting a fundamental information bottleneck of coarse statistics).*

Given standardization $(\boldsymbol{\mu}_{\text{cal}}, \boldsymbol{\sigma}_{\text{cal}})$ on held-out calibration data:
$$\tilde{\mathbf{z}}_b(\mathbf{x}) = \operatorname{diag}(\boldsymbol{\sigma}_{\text{cal}})^{-1} \left(\mathbf{z}_b(\mathbf{x}) - \boldsymbol{\mu}_{\text{cal}}\right)$$
$$s_b(\mathbf{x}) = \sigma\left(\mathbf{w}_b^\top \tilde{\mathbf{z}}_b(\mathbf{x}) + b_b\right) \in (0, 1)$$

##### Detector (c) — Meta-Interoceptive Observer Head
An independent MLP head $g_\phi: \mathbb{R}^H \to (0, 1)$ parameterized by $\phi = \{\mathbf{W}_{\text{head}}^{(1)} \in \mathbb{R}^{64 \times 256}, \mathbf{b}_{\text{head}}^{(1)} \in \mathbb{R}^{64}, \mathbf{w}_{\text{head}}^{(2)} \in \mathbb{R}^{1 \times 64}, b_{\text{head}}^{(2)} \in \mathbb{R}\}$:
$$s_c(\mathbf{x}) = \sigma\left(\mathbf{w}_{\text{head}}^{(2)} \operatorname{ReLU}\left(\mathbf{W}_{\text{head}}^{(1)} \mathbf{h}_B(\mathbf{x}) + \mathbf{b}_{\text{head}}^{(1)}\right) + b_{\text{head}}^{(2)}\right) \in (0, 1)$$
Trained by minimizing the binary cross-entropy loss over a held-out calibration set $\mathcal{D}_{\text{cal}} = \{(\mathbf{x}_i, y_i)\}_{i=1}^M$ unseen by Model B during its training:
$$\mathcal{L}(\phi) = -\frac{1}{M} \sum_{i=1}^M \left[ e_{B,i} \log s_c(\mathbf{x}_i) + (1 - e_{B,i}) \log \left(1 - s_c(\mathbf{x}_i)\right) \right]$$

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

#### Mathematical Formulation of Selective Prediction
For a confidence scoring function $\kappa: \mathbb{R}^D \to \mathbb{R}$, a selective classifier predicts when confidence meets or exceeds a decision threshold $\tau$, abstaining otherwise.

Given a test dataset $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$, empirical coverage and selective accuracy are defined as:
$$\operatorname{Coverage}(\tau) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left(\kappa(\mathbf{x}_i) \ge \tau\right)$$
$$\operatorname{SelectiveAccuracy}(\tau) = \frac{\sum_{i=1}^N \mathbb{I}\left(\hat{y}_{B}(\mathbf{x}_i) = y_i\right) \cdot \mathbb{I}\left(\kappa(\mathbf{x}_i) \ge \tau\right)}{\sum_{i=1}^N \mathbb{I}\left(\kappa(\mathbf{x}_i) \ge \tau\right)}$$

For target coverage $C \in (0, 1]$, threshold $\tau(C) = \operatorname{Quantile}_{1-C}\left(\{\kappa(\mathbf{x}_i)\}_{i=1}^N\right)$.

#### Learned Meta-Interoceptive Blend
The three confidence estimators $\mathbf{c}(\mathbf{x}) = [\kappa_a(\mathbf{x}), \; 1 - s_b(\mathbf{x}), \; 1 - s_c(\mathbf{x})]^\top \in [0, 1]^3$ are standardized and mapped via learned logistic weights:
$$\kappa_{\text{blend}}(\mathbf{x}) = \sigma\left(\mathbf{w}_{\text{blend}}^\top \tilde{\mathbf{c}}(\mathbf{x}) + b_{\text{blend}}\right)$$
Empirical fit coefficients on held-out data:
$$\mathbf{w}_{\text{blend}} = [0.472, \; -0.239, \; 2.850]^\top, \quad b_{\text{blend}} = 4.312$$

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

---

## 🌐 The Neural Self-Awareness Continuum

| # | Project | Biological Analogy | Core Capability | Live Link |
|---|---|---|---|---|
| **1** | **[Interoception](https://haidar167.github.io/interoception/)** | Internal visceral sensing | Senses internal confusion via hidden activation stats | [GitHub](https://github.com/haidar167/interoception) |
| **2** | **[Proprioception](https://haidar167.github.io/proprioception/)** | Body substrate awareness | Senses weight damage & localizes corrupted layers | [GitHub](https://github.com/haidar167/proprioception) |
| **3** | **[Meta-Interoception](https://haidar167.github.io/meta-interoception/)** | Metacognitive monitoring | Monitors the calibration of its own self-monitors | [GitHub](https://github.com/haidar167/meta-interoception) |
| **4** | **[Nociception](https://haidar167.github.io/nociception/)** | Pain-driven help seeking | Spends limited human supervision budget on likely errors | [GitHub](https://github.com/haidar167/nociception) |
| **5** | **[SOMNIA](https://haidar167.github.io/somnia/)** | Targeted sleep consolidation | Dreams targeted examples to patch its own weak spots | [GitHub](https://github.com/haidar167/somnia) |

---
*Part of the Neural Self-Awareness research continuum by [haidar167](https://github.com/haidar167).*
