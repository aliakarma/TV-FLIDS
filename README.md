# TV-FLIDS: Trust-Aware & Verifiable Federated Intrusion Detection System

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange.svg)](https://pytorch.org)
[![Flower](https://img.shields.io/badge/Flower-1.6.0-green.svg)](https://flower.dev)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

> **Final Year Project | Research Paper Ready | IEEE IoT Journal Target**

A production-ready implementation of a Byzantine-resilient Federated Learning system for IoT Intrusion Detection. TV-FLIDS defends against malicious clients through a unified three-criteria verification gate combined with dynamic memory-aware trust scoring.

---

## Overview

Standard Federated Learning is vulnerable to adversarial clients that poison the global model. TV-FLIDS addresses this by introducing:

1. **Verification Gate** — Pre-aggregation filter checking loss consistency, gradient direction, and statistical outliers
2. **Trust Scoring** — Dynamic per-client scores with exponential memory decay (T_i = α·S_i + β·A_i − γ·O_i)
3. **Adaptive Weights** — Meta-gradient learning of α, β, γ from server validation loss
4. **Formal Guarantees** — Proposition 1 bounding Byzantine influence under the trust floor

---

## Architecture

```
NSL-KDD / UNSW-NB15
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Flower FL Simulation                       │
│                                                               │
│  Client 1..N                                                  │
│  ┌────────────┐                                               │
│  │ Local MLP  │──► Δw_i + val_loss_i ──────────────────────► │
│  └────────────┘         (per round)                           │
│                                                               │
│  TVFLIDSStrategy (Server)                                     │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  1. VerificationModule                                  │  │
│  │     ├─ Check 1: loss_after > loss_before?              │  │
│  │     ├─ Check 2: cosine_sim(Δw_i, mean_Δw) > threshold? │  │
│  │     └─ Check 3: z_score(||Δw_i||) < threshold?         │  │
│  │  2. TrustScorer (Adaptive)                              │  │
│  │     T_i(t) = 0.9·T_i(t-1) + 0.1·[α·S_i+β·A_i-γ·O_i]  │  │
│  │     Meta-gradient update on α, β, γ                    │  │
│  │  3. Weighted Aggregation                                │  │
│  │     w^{t+1} = Σ(T_i/ΣT) · w_i                         │  │
│  └────────────────────────────────────────────────────────┘  │
│                        │                                      │
│                   Global Model w^{t+1}                        │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
  NSL-KDD Test Set Evaluation
  Metrics: Accuracy, F1-Macro, Attack Success Rate
```

---

## Features

- **5 FL strategies**: FedAvg, Krum, Trimmed Mean, FLTrust, FoolsGold, TV-FLIDS
- **4 attack types**: Label Flip, Gradient Scaling, Noise Injection, Backdoor
- **IID & Non-IID** data partitioning (Dirichlet α=0.5/0.1)
- **Publication-grade statistics**: 5-seed mean±std, Wilcoxon, McNemar tests
- **6 paper figures**: Convergence, trust evolution, robustness curve, ablation, weight trajectory, confusion matrices
- **Overhead analysis**: Per-round timing and communication cost measurement
- **Full ablation suite**: A1–A5 component contribution analysis

---

## Installation

### Prerequisites
- Conda (recommended) or Python 3.10
- NVIDIA GPU optional but recommended for speed

### Setup (VS Code Terminal)

```bash
# 1. Clone the repository
git clone https://github.com/aliakarma/tv-flids.git
cd tv-flids

# 2. Create conda environment
conda create -n tvflids python=3.10
conda activate tvflids

# 3. Install PyTorch (GPU version — adjust cuda version if needed)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# For CPU-only:
# pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# 4. Install all dependencies
pip install -r requirements.txt

# 5. Verify installation
python -c "import torch; import flwr; print('PyTorch:', torch.__version__, '| Flower:', flwr.__version__)"
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

---

## Dataset Setup

### NSL-KDD (Primary — required)

```bash
# Automatic download via script
bash scripts/download_nslkdd.sh

# OR via Python
python -c "from data.preprocessing.nslkdd_pipeline import download_nslkdd; download_nslkdd('data/raw/KDDTrain+.txt', 'data/raw/KDDTest+.txt')"
```

Verifies as:
```
data/raw/KDDTrain+.txt  → ~125,973 rows
data/raw/KDDTest+.txt   → ~22,544 rows
```

### UNSW-NB15 (Secondary — optional)
Download from: https://research.unsw.edu.au/projects/unsw-nb15-dataset
Place at: `data/raw/UNSW_NB15_training-set.csv` and `data/raw/UNSW_NB15_testing-set.csv`

---

## Quick Start

### Run a single experiment

```bash
# TV-FLIDS vs 30% label flip attack (primary experiment)
python experiments/run_experiment.py --strategy tvflids --attack label_flip_30

# FedAvg baseline (no defense, shows vulnerability)
python experiments/run_experiment.py --strategy fedavg --attack label_flip_30

# Clean baseline (no attacks, upper bound)
python experiments/run_experiment.py --strategy fedavg --attack no_attack

# FLTrust comparison (SOTA baseline)
python experiments/run_experiment.py --strategy fltrust --attack label_flip_30
```

### Run with options

```bash
# Custom seed and rounds (fast test)
python experiments/run_experiment.py \
    --strategy tvflids \
    --attack label_flip_30 \
    --seed 123 \
    --rounds 20

# IID data partition
python experiments/run_experiment.py \
    --strategy tvflids \
    --attack label_flip_30 \
    --partition iid

# Extreme non-IID (stress test)
python experiments/run_experiment.py \
    --strategy tvflids \
    --attack label_flip_30 \
    --partition noniid \
    --alpha 0.1

# Different attack types
python experiments/run_experiment.py --strategy tvflids --attack gradient_scale_30
python experiments/run_experiment.py --strategy tvflids --attack noise_30
python experiments/run_experiment.py --strategy tvflids --attack backdoor_20
```

---

## Reproducing Paper Results

### Table 1 — Full Strategy Comparison

```bash
python experiments/run_full_comparison.py \
    --strategies fedavg krum trimmed_mean fltrust foolsgold tvflids \
    --attack label_flip_30 \
    --seeds 42 123 456 789 1337 \
    --rounds 100
```

### Figure 3 — Robustness Curve

```bash
python experiments/run_ratio_sweep.py \
    --methods fedavg fltrust tvflids \
    --ratios 0.0 0.1 0.2 0.3 0.4 0.5 \
    --seeds 42 123 456 \
    --rounds 100
```

### Table 2 — Ablation Study

```bash
python experiments/run_ablation.py \
    --attack label_flip_30 \
    --rounds 100 \
    --seeds 42 123 456 789 1337
```

### All Experiments (Full Reproduction)

```bash
# Runs everything — expect 2-4 hours on CPU, ~30 min on GPU
bash scripts/run_all_experiments.sh
```

---

## Available Strategies

| Strategy | Reference | Key Property |
|---|---|---|
| `fedavg` | McMahan et al., 2017 | Standard baseline (no defense) |
| `krum` | Blanchard et al., NeurIPS 2017 | Nearest-neighbor selection |
| `trimmed_mean` | Yin et al., ICML 2018 | Coordinate-wise robust mean |
| `fltrust` | Cao et al., NDSS 2021 | Server-root trust bootstrapping |
| `foolsgold` | Fung et al., 2018 | Sybil resistance via history |
| `tvflids` | **This work** | 3-criteria gate + adaptive trust |
| `tvflids_fixed` | **This work** | TV-FLIDS with fixed α, β, γ |

## Available Attacks

| Config Key | Type | Ratio | Description |
|---|---|---|---|
| `no_attack` | — | 0% | Clean baseline |
| `label_flip_10/20/30` | Data | 10/20/30% | Flip attack→Normal labels |
| `gradient_scale_10/30` | Model | 10/30% | Amplify gradient ×10 |
| `noise_30` | Model | 30% | Gaussian noise (σ=0.5) |
| `backdoor_20` | Data | 20% | Trigger pattern insertion |

---

## Configuration

Edit `config/fl_config.yaml` to change hyperparameters:

```yaml
federated_learning:
  num_clients: 20       # Simulated IoT devices
  num_rounds: 100       # FL communication rounds
  fraction_fit: 0.5     # 50% clients participate per round
  local_epochs: 5       # Local training epochs
  local_lr: 0.001       # Adam learning rate

trust:
  alpha: 0.4            # Similarity weight
  beta: 0.4             # Accuracy weight
  gamma: 0.2            # Anomaly penalty weight
  memory_decay: 0.9     # EMA decay factor
  min_trust: 0.01       # Trust floor

verification:
  loss_threshold: 0.0   # Reject if ΔL < 0
  cosine_threshold: 0.0 # Flag if cos_sim < 0
  zscore_threshold: 2.5 # Flag if |z| > 2.5
```

---

## Expected Results

On NSL-KDD with 30% label flip, Non-IID (α=0.5):

| Strategy | Accuracy | F1-Macro | Attack Success Rate |
|---|---|---|---|
| FedAvg (clean) | ~0.930 | ~0.880 | ~0.000 |
| FedAvg (attacked) | ~0.610 | ~0.520 | ~0.720 |
| Krum | ~0.820 | ~0.750 | ~0.380 |
| FLTrust | ~0.860 | ~0.810 | ~0.280 |
| FoolsGold | ~0.800 | ~0.740 | ~0.420 |
| **TV-FLIDS** | **~0.880** | **~0.850** | **~0.190** |

*Values are indicative targets. Actual results depend on seed and system.*

---

## Project Structure

```
tv-flids/
├── config/                    # Hyperparameter configs (YAML)
│   ├── fl_config.yaml
│   └── dataset_config.yaml
├── data/
│   ├── preprocessing/         # NSL-KDD, UNSW-NB15, MNIST pipelines
│   └── partitioning.py        # IID & Non-IID (Dirichlet) partitioners
├── models/
│   └── mlp.py                 # IDSMLP + IDSBiLSTM architectures
├── fl/
│   ├── client.py              # Flower FL client with attack injection
│   ├── strategy.py            # TVFLIDSStrategy (main novel contribution)
│   └── baselines/             # FedAvg, Krum, TrimMean, FLTrust, FoolsGold
├── trust/
│   ├── trust_scorer.py        # Fixed-weight trust scoring
│   ├── adaptive_trust_scorer.py  # Meta-gradient adaptive α,β,γ
│   └── verification.py        # Three-criteria verification gate
├── attacks/
│   └── adversarial.py         # 4 attack types + configuration registry
├── evaluation/
│   ├── metrics.py             # Accuracy, F1, ASR, FNR tracking
│   ├── statistical_testing.py # Wilcoxon, McNemar, multi-seed reporting
│   ├── visualization.py       # 6 paper-ready figure generators
│   └── overhead.py            # Time/communication cost analysis
├── experiments/
│   ├── run_experiment.py      # Main experiment runner (start here)
│   ├── run_ablation.py        # A1-A5 ablation studies
│   ├── run_ratio_sweep.py     # Adversarial ratio sweep
│   └── run_full_comparison.py # Multi-seed Table 1 reproduction
├── utils/
│   ├── seed.py                # Centralized seed management
│   └── logger.py              # JSON + TensorBoard logging
├── scripts/
│   ├── download_nslkdd.sh     # Dataset download
│   └── run_all_experiments.sh # Full paper reproduction
├── results/                   # Generated outputs (gitignored)
│   ├── logs/                  # Per-experiment JSON logs
│   ├── figures/               # PDF paper figures
│   └── tables/                # CSV/JSON result tables
├── requirements.txt
└── README.md
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{tvflids2024,
  title   = {TV-FLIDS: Trust-Aware and Verifiable Federated Intrusion Detection
             for IoT under Adversarial Clients},
  author  = {Ali Akarma},
  journal = {},
  year    = {}
}
```

### Key References

```bibtex
@inproceedings{blanchard2017nips,
  title={Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent},
  author={Blanchard et al.},
  booktitle={NeurIPS}, year={2017}
}

@inproceedings{cao2021fltrust,
  title={FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping},
  author={Cao et al.},
  booktitle={NDSS}, year={2021}
}

@article{mcmahan2017fedavg,
  title={Communication-Efficient Learning of Deep Networks from Decentralized Data},
  author={McMahan et al.},
  booktitle={AISTATS}, year={2017}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
