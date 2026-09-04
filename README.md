# TV-FLIDS: Trust-Aware & Verifiable Federated Intrusion Detection System

[![CI](https://github.com/aliakarma/tv-flids/actions/workflows/ci.yml/badge.svg)](https://github.com/aliakarma/tv-flids/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange.svg)](https://pytorch.org)
[![Flower](https://img.shields.io/badge/Flower-1.6.0-green.svg)](https://flower.dev)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

> **Final Year Project | Research Paper Ready | IEEE IoT Journal Target**

A production-ready implementation of a Byzantine-resilient Federated Learning system for IoT Intrusion Detection. TV-FLIDS defends against malicious clients through a unified three-criteria verification gate combined with dynamic memory-aware trust scoring.

---

## Hardware Requirements

| Setup | Min RAM | Recommended GPU | Est. Full Run Time |
|-------|---------|----------------|--------------------|
| CPU only | 8 GB | — | ~8 hours |
| GPU (8 GB VRAM) | 16 GB | NVIDIA RTX 3080+ | ~90 min |

Set CPU parallelism: `export TVFLIDS_SIM_CLIENT_CPUS=4`

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
│                    Flower FL Simulation                     │
│                                                             │
│  Client 1..N                                                │
│  ┌────────────┐                                             │
│  │ Local MLP  │──► Δw_i + val_loss_i ──────────────────────►│
│  └────────────┘         (per round)                         │
│                                                             │
│  TVFLIDSStrategy (Server)                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  1. VerificationModule                                 │ │
│  │     ├─ Check 1: loss_after > loss_before?              │ │
│  │     ├─ Check 2: cosine_sim(Δw_i, mean_Δw) > threshold? │ │
│  │     └─ Check 3: z_score(||Δw_i||) < threshold?         │ │
│  │  2. TrustScorer (Adaptive)                             │ │
│  │     T_i(t) = 0.9·T_i(t-1) + 0.1·[α·S_i+β·A_i-γ·O_i]    │ │
│  │     Meta-gradient update on α, β, γ                    │ │
│  │  3. Weighted Aggregation                               │ │
│  │     w^{t+1} = Σ(T_i/ΣT) · w_i                          │ │
│  └────────────────────────────────────────────────────────┘ │
│                        │                                    │
│                   Global Model w^{t+1}                      │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
  NSL-KDD Test Set Evaluation
  Metrics: Accuracy, F1-Macro, Attack Success Rate
```

### Model Selection

The default model is `IDSMLP` (4-layer MLP with BatchNorm + Dropout).  
An experimental `IDSBiLSTM` is available for sequential traffic analysis:

```bash
python experiments/run_experiment.py --strategy tvflids --attack label_flip_30 --model bilstm
```

The paper results use `--model mlp` (default).

---

## Features

- **10 FL strategies**: FedAvg, Krum (Multi-Krum, m=2), Trimmed Mean, FLTrust, FoolsGold, FLAME, RFA, Bucketing, DeepSight, TV-FLIDS
- **NSL-KDD (primary), UNSW-NB15, and CIC-IoT-2023** dataset pipelines (`--dataset {nslkdd,unswnb15,ciciot2023}`)
- **Two data-preparation protocols**: `--protocol main` (paper's main-table protocol) and `--protocol leakage_free` (D_val drawn pre-SMOTE, SMOTE applied per-client post-partition)
- **Attack types**: Label Flip, Gradient Scaling, Noise Injection, Backdoor, Min-Max, and the adaptive gate-evasion attacks ACK1 (`ack1_evasion_30`) and ACK2 (`ack2_coalition_30`)
- **IID & Non-IID** data partitioning (Dirichlet α=0.5/0.1)
- **Publication-grade statistics**: 5-seed (and extended 10-seed, `experiments/run_extended_significance.py`) mean±std, Wilcoxon, McNemar tests
- **6 paper figures**: Convergence, trust evolution, robustness curve, ablation, weight trajectory, confusion matrices
- **Overhead analysis**: Per-round timing and communication cost measurement
- **Full ablation suite**: A1–A6 component contribution analysis (A6 isolates the meta-gradient's marginal contribution)
- **Dedicated sweep/matrix runners**: non-IID concentration sweep, multi-attack × baseline matrix, baseline and TV-FLIDS hyperparameter sensitivity sweeps (`experiments/run_noniid_sweep.py`, `run_multi_attack_matrix.py`, `run_hyperparameter_sweep.py`)

---

## Installation

### Option A — Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate tvflids
make data
```

### Option B — pip

```bash
pip install -r requirements.txt
bash scripts/download_nslkdd.sh
```

### Verify Installation

```bash
make verify-env      # reports the ACTUAL stack against the paper's stated one
make smoke           # 3-round sanity run
```

`scripts/verify_environment.py` compares the live interpreter and package
versions against the environment the paper states in Section VI-D
(Python 3.10.12, PyTorch 2.1.0, Flower 1.6.0, scikit-learn 1.3.2,
imbalanced-learn 0.11.0, NumPy 1.26.2, SciPy 1.11.4) and prints every
mismatch. It reports; it never claims equivalence. Add `--strict` to make a
mismatch a non-zero exit code, or `--json` for machine-readable output.

**Reproduce the paper's environment exactly** with `environment.yml` (which
pins Python 3.10.12) or `requirements.txt` (pins every package, but must be
installed into a 3.10.x interpreter — `.python-version` records 3.10.12).

### Known environment limitation (Windows)

The Flower simulation engine runs every virtual client inside a **Ray worker
process**, and importing PyTorch inside those workers fails on Windows with:

```
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine
failed. Error loading ...\torch\lib\c10.dll or one of its dependencies.
```

This is an environment/platform interaction, not a defect in this code. What
it means in practice:

| Works on Windows | Needs Linux or WSL2 |
|---|---|
| All unit tests not routed through Ray | `make smoke`, `make full-comparison`, and every other full experiment |
| In-process strategy tests (`tests/test_overhead_tracking.py`, `test_warmup_schedule.py`, `test_ste_gradient.py`, `test_val_size_regression.py`) | Anything invoking `flwr.simulation.start_simulation` |
| `theory/proposition1_verification.py` | |
| The preprocessing pipelines and all dataset code | |

Running `pytest tests/` on Windows therefore leaves a fixed set of
simulation-dependent failures (all with the `c10.dll` signature above). Do not
"fix" those by editing the tests; run them on the paper's stack instead.

## Performance Notes

- **Simulation speed**: the manuscript reports ~15% per-round overhead over
  FedAvg (verification gate + trust scoring; evaluation is shared). That figure
  is hardware-specific and predates the timing instrumentation now wired into
  `TVFLIDSStrategy.aggregate_fit`; it has not been re-measured. Each run
  summary carries the live per-stage breakdown under `compute_overhead_ms`.
- **CPU parallelism**: Set `TVFLIDS_SIM_CLIENT_CPUS` to at most `nproc / 2`  
  to avoid Ray resource starvation:
  ```bash
  export TVFLIDS_SIM_CLIENT_CPUS=4
  python experiments/run_experiment.py --strategy tvflids --attack label_flip_30
  ```
- **GPU usage**: Set `TVFLIDS_SIM_CLIENT_GPUS=0.1` per virtual client  
  if your GPU has ≥ 8 GB VRAM.

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

## Reproducibility Guarantee

All stochastic operations are seeded via `utils/seed.py::set_all_seeds(seed)`:

| Operation | Seeded via |
|-----------|-----------|
| NumPy global state | `np.random.seed(seed)` |
| NumPy RNGs | `np.random.default_rng(seed)` |
| PyTorch | `torch.manual_seed(seed)` + `torch.backends.cudnn.deterministic=True` |
| CUDA | `torch.cuda.manual_seed_all(seed)` |
| Python hash | `os.environ["PYTHONHASHSEED"] = str(seed)` |
| Attacks | `seed + client_id` per client |
| SMOTE | `random_state=seed` |
| Data partitioning | `np.random.default_rng(seed)` |

To reproduce Table 1, use seeds `[42, 123, 456, 789, 1337]`.

## Data Integrity Verification

After downloading NSL-KDD, verify file integrity:

```bash
python scripts/verify_data.py
```

Expected SHA-256 hashes are recorded in `scripts/verify_data.py`.
If the check fails, re-run `bash scripts/download_nslkdd.sh`.

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

## Testing

Integration tests will auto-download NSL-KDD if needed.

```bash
python tests/test_all.py
python tests/test_integration.py
```

---

## Result Provenance

A forensic audit (2026-09-03) found that several files under `results/tables/` and
`results/logs/comparison/` had been committed as fabricated placeholder output (from a
since-removed `mock_phase7_results.py` generator) rather than genuine experiment results —
including a literal `"_config_hash": "mockhash"` field and zero-variance metrics across
seeds, both impossible for a real stochastic FL run. These have been moved to
`results/_QUARANTINED_MOCK/` (see the README there for full per-file provenance) and must
never be treated as evidence. Genuine results are produced only by actually running the
experiment scripts below; `utils/logger.py::ExperimentLogger` computes a real sha256-based
config hash for every genuine run, distinguishing it from the old placeholder string.

Run `make check-results` before citing anything from `results/`. It checks both
**completeness** (are the expected files present, with all strategy x seed
cells?) and **authenticity** (does any live file carry a quarantined
fabrication fingerprint, is any "figure" a zero-byte placeholder, does the
generated Proposition 1 output carry real provenance?). It exits non-zero
today, correctly: the genuine tables have not yet been produced.

### Second pass (final hardening, 2026-09-03)

A follow-up pass reconciled the implementation against the manuscript's
methodology. Four findings change how the existing numbers must be read:

1. **Gate warmup annealing was never active.** `use_adaptive_thresholds` was
   hardcoded `False` at the only production entry point, so the tau_L and
   tau_z schedules of paper Section IV-A never ran. The tau_z formula was also
   wrong — multiplicative instead of additive, giving tau_z(0) = 7.5 rather
   than 3.0, and discontinuous at `t = T_warm` (jumping back to ~4.88 at
   `t = T_warm + 1`, i.e. *more* permissive after warmup than during it). Both
   are fixed; annealing is now on by default via
   `verification.adaptive_thresholds`. Consequence: the `warmup_rounds` axis of
   the Supplementary Table S2 sweep previously had no effect.
2. **The straight-through estimator was not implemented.** The meta-gradient
   used `torch.clamp`, which zeroes the gradient for saturated clients. It now
   uses `utils/ste.py::clip_ste` (forward identical, backward = identity), as
   the paper specifies.
3. **Table XII had no instrumented source.** `OverheadTracker` was constructed
   but never used. It is now wired into every stage of
   `TVFLIDSStrategy.aggregate_fit` and serialized under `compute_overhead_ms`.
   The table's *values* still need re-measuring on the paper's hardware.
4. **The client-sampling description was wrong.** Flower samples the round
   cohort *without* replacement, so the paper's "with replacement" convention
   and its "<3% of rounds" collision rate were both incorrect. Corrected in the
   manuscript.

Items 1 and 2 change aggregation dynamics, so **the full campaign must be re-run
against the corrected implementation**; no existing number carries over.

### Third pass (final closure, 2026-09-04)

A closure pass before the campaign resolved the remaining methodology and
manuscript defects. Seven changes affect how the repository behaves or how the
manuscript must be read:

1. **Validation forward-pass count corrected.** The manuscript stated
   `|A| + 1` (accepted clients) and called it an upper bound. Check 1 evaluates
   every *submitted* client before acceptance is known, so the exact count is
   `|P| + 1` over the round's cohort `P`, and `|A| + 1` is a **lower** bound
   that under-counts by exactly the number of rejections.
   `tests/test_forward_pass_count.py` asserts this against the real
   `aggregate_fit` path.
2. **Notation collision resolved.** Section III's `H`, `B`, `f`, `N_H` are
   population-level; Section V had silently reused them for accepted-set
   quantities, which put `N_H = 14` and `N_H = accepted honest count` in the
   same paper under one symbol. Section V now uses `H_A`, `B_A`, `f_A`,
   `N_{H,A}`, `tau-bar_{H,A}`, `tau^max_{B,A}`.
3. **Proposition 1 gained its missing hypothesis.** The accepted-set
   formulation makes `H_A = {}` reachable, where `w*` and `tau-bar_{H,A}` are
   undefined and the denominator is zero. The proposition now assumes
   `N_{H,A} >= 1`. The verification script reports that case as outside the
   proposition's domain instead of scoring it, and a real bug was fixed on the
   way: the documented `B_A = {}` equality convention raised a numpy
   `ValueError` rather than returning equality.
   `tests/test_proposition1_domain.py` covers all three boundary cases.
4. **A malformed cross-reference was found and fixed.** The backslash of a
   `ef` had been replaced by a raw carriage return, so the PDF typeset
   `Section efsec:adaptive` with no LaTeX warning of any kind.
   `scripts/check_manuscript.py` (`make check-manuscript`) now detects this
   class of defect at three levels: source bytes, macro tokens, and compiled
   PDF text.
5. **`--strategy tvflids_fixed` now *is* ablation A6.** `config/fl_config.yaml`
   shipped `alpha=0.4, beta=0.4, gamma=0.2`, and only `run_ablation.py`'s
   override pushed the arm to the intended `1/3` each, so a direct CLI
   invocation silently produced a different arm under the same name. The base
   config now matches paper Table IV. The adaptive strategy is unaffected: it
   never read those keys. Pinned by `tests/test_config_defaults.py`.
6. **A1's semantics are now stated.** `tau_z` is shared between Check 3 and the
   anomaly signal `O_i`, so A1's `tau_z = +inf` disables **both** the anomaly
   gate and the anomaly contribution to trust. The implementation is faithful
   to the stated definition; the manuscript now says so, and treats the A1 gap
   as an upper bound on the gate's own contribution.
   `tests/test_a1_ablation_semantics.py` pins the behaviour.
7. **Figures gained a reproducible generation path.** See
   [Manuscript figures](#manuscript-figures). `utils/logger.py` now persists the
   per-client trust history and the strategy round logs, without which
   Figures 3 and 4 had no disk artifact to regenerate from.

The manuscript's Abstract, Conclusion, Section VII banner, Table VII, and every
results-section table caption were rewritten so that no value reads as a
measurement. Table VII's A1-A5 rows are labelled with their traced origin (the
mock generator's hardcoded constants) and A6's with the fact that no A6
implementation existed when its row was written.

## Results pending regeneration

Nothing in `results/tables/` other than `proposition1_real.json` has been
produced by a genuine run. Every table below is technically reproducible — the
code exists, is reachable from a documented command, and has been smoke-tested
— but **has not been executed at full scale**.

| Paper artifact | Command | Status |
|---|---|---|
| Table V (main comparison) | `make full-comparison` | Pending full run |
| Table VI (leakage-free) | `make leakage-free` | Pending full run |
| Table VII (ablation A1-A6) | `make ablation` | Pending full run; prior values traced to the quarantined mock generator |
| Table VIII (non-IID sweep) | `make noniid-sweep` | Pending full run |
| Tables IX / X (attack matrix) | `make multi-attack` | Pending full run |
| Table XI (ACK1, ACK2) | `make multi-attack` / `run_experiment.py --attack ack1_*` | Pending full run |
| Table XII (overhead) | any `run_experiment.py` run on the paper's hardware | Instrumentation present; values pending re-measurement |
| Figures 2-5 (manuscript) | `make full-comparison`, `make figures`, then `make manuscript-figures` | Pending; the manuscript renders provisional coordinates until `Paper/figures/fig_*.tex` exist |
| Figures 1-6 (`results/figures/`) | `make full-comparison`, `make figures` | Pending; the six 0-byte placeholder PDFs that previously sat at these paths were deleted during the closure pass |
| Supp. Table S1 (baseline HP) | `make hp-sweep-baseline` | Pending full run |
| Supp. Table S2 (TV-FLIDS HP) | `make hp-sweep-tvflids` | Pending full run |
| Supp. Bucketing / DeepSight | `make full-comparison --strategies ... bucketing deepsight` | Pending full run |
| Supp. CIC-IoT-2023 | `make ciciot2023` | Pending full run **and** the real dataset |
| Section VIII-B (10-seed) | `make extended-significance` | Pending full run |
| **Proposition 1 (Section XIII)** | `make theory` | **Done** — deterministic, seed 42, reproduces 0.588 +/- 0.198 with real provenance |

### Manuscript figures

Manuscript Figures 2-5 (and Supplementary Figure S1) are rendered as pgfplots
coordinate lists rather than imported images. Since the closure pass they are no
longer hand-maintained: each axis reads its plot bodies from a generated file
under `Paper/figures/` when that file exists, and otherwise falls back to
provisional coordinates that the caption marks as pending.

```bash
make manuscript-figures         # results/ -> Paper/figures/fig_*.tex
make check-manuscript-figures   # report which figures are backed by real output
make paper                      # regenerate figure data, then build both PDFs
make check-manuscript           # integrity-check the .tex sources and built PDFs
```

`scripts/generate_manuscript_figures.py` never fabricates data: a figure whose
backing artifact is missing is skipped with a message and no file is written.
See `Paper/figures/README.md` for the artifact-to-figure mapping.

`scripts/check_manuscript.py` scans for the class of manuscript defect that
compiles cleanly and is therefore invisible to `latexmk`'s exit status: stray
control bytes in the source, TeX macros whose leading backslash was replaced by
a control character, undefined references, and forbidden text in the compiled
PDF text. It found a `
ef` whose backslash had been replaced by a raw carriage
return, which had been silently typesetting as `Section efsec:adaptive`.

## Reproducing Paper Results

### Table 1 — Full Strategy Comparison

```bash
python experiments/run_full_comparison.py \
    --strategies fedavg krum trimmed_mean fltrust foolsgold flame rfa tvflids \
    --attack label_flip_30 \
    --seeds 42 123 456 789 1337 \
    --rounds 100
```

`bucketing` and `deepsight` are implemented and selectable but intentionally excluded
from this default list pending a full experimental run (see the remediation report) —
pass `--strategies ... bucketing deepsight` to include them.

### Figure 3 — Robustness Curve

```bash
python experiments/run_ratio_sweep.py \
    --methods fedavg krum fltrust tvflids \
    --ratios 0.0 0.1 0.2 0.3 0.4 0.5 0.6 \
    --seeds 42 123 456 \
    --rounds 100
```

### Table VIII — Non-IID Concentration Sweep

```bash
python experiments/run_noniid_sweep.py --strategies krum tvflids \
    --alphas 0.1 0.5 1.0 --seeds 42 123 456 789 1337 --rounds 100
```

### Table X — Multi-Attack × Baseline Matrix

```bash
python experiments/run_multi_attack_matrix.py \
    --strategies fedavg krum trimmed_mean fltrust foolsgold flame rfa tvflids \
    --attacks gradient_scale_30 noise_30 backdoor_20 \
    --seeds 42 123 456 789 1337 --rounds 100
```

### Table XI — Adaptive Attacks (ACK1, ACK2)

```bash
python experiments/run_experiment.py --strategy tvflids --attack ack1_evasion_30
python experiments/run_experiment.py --strategy tvflids --attack ack2_coalition_30
```

### Section VIII-B — Extended Ten-Seed Significance Test

```bash
python experiments/run_extended_significance.py \
    --strategies tvflids fltrust \
    --seeds 42 123 456 789 1337 2024 31415 8080 555 999 --rounds 100
```

### Supplementary Tables S1/S2 — Hyperparameter Sensitivity Sweeps

```bash
python experiments/run_hyperparameter_sweep.py --target baseline
python experiments/run_hyperparameter_sweep.py --target tvflids
```

### Table 2 — Ablation Study (A1–A6)

```bash
python experiments/run_ablation.py \
    --attack label_flip_30 \
    --rounds 100 \
    --seeds 42 123 456 789 1337
```

A6 ("Fixed Equal Weights": verification gate on, meta-gradient adaptation off,
α=β=γ=1/3 for the whole run) isolates the meta-gradient's marginal contribution
over a static trust-signal blend.

### Server validation set (D_val) and data protocol

```bash
# Main protocol (paper's main-table results): global SMOTE, then D_val (2,000
# stratified samples) drawn from the SMOTE-balanced pool. This is now the
# actual default -- previously the pipeline silently used an internal 5%
# fraction (~6,299 samples on NSL-KDD) that never matched the paper.
python experiments/run_experiment.py --strategy tvflids --attack label_flip_30 \
    --val-size 2000 --protocol main

# Leakage-free protocol (paper Table VI): D_val drawn BEFORE SMOTE; SMOTE
# applied per-client, after Dirichlet partitioning, so no synthetic sample
# derived from a validation example can leak into client training data.
python experiments/run_experiment.py --strategy tvflids --attack label_flip_30 \
    --val-size 2000 --protocol leakage_free
```

### All Experiments (Full Reproduction)

```bash
# Runs everything — expect 2-4 hours on CPU, ~30 min on GPU
bash scripts/run_all_experiments.sh
```

## Reproducing Paper Tables

After running all experiments:

```bash
# Generate LaTeX Table 1
python scripts/generate_tables.py \
    --input results/tables/full_comparison_results.json \
    --output results/tables/table1.tex

# Check result completeness before submission
python scripts/check_results.py
```

## Theoretical Verification

Proposition 1 and Lemma 1 (trust convergence) can be verified numerically:

```bash
# Synthetic verification (no data required)
python theory/proposition1_verification.py

# Full theory validation suite
python scripts/run_theory_validation.py
```

Results are saved to `results/tables/proposition1_real.json`.

---

## Available Strategies

| Strategy | Reference | Key Property |
|---|---|---|
| `fedavg` | McMahan et al., 2017 | Standard baseline (no defense) |
| `krum` | Blanchard et al., NeurIPS 2017 | Nearest-neighbor selection |
| `trimmed_mean` | Yin et al., ICML 2018 | Coordinate-wise robust mean |
| `fltrust` | Cao et al., NDSS 2021 | Server-root trust bootstrapping |
| `foolsgold` | Fung et al., 2018 | Sybil resistance via history |
| `flame` | Nguyen et al., USENIX Security 2022 | HDBSCAN + adaptive noise |
| `rfa` | Pillutla et al., IEEE TSP 2022 | Geometric median (Weiszfeld) |
| `bucketing` | Karimireddy et al., ICLR 2022 | Bucket-then-trim (s=2, trimmed mean over bucket averages) |
| `deepsight` | Rieger et al., NDSS 2022 | Bias-gradient clustering + weight-clip |
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
| `min_max_30` | Model | 30% | Norm-ball-constrained deviation maximization |
| `ack1_evasion_30` | Data + training-time | 30% | Check-1 evasion: auxiliary loss on a proxy-D_val slice, alongside label-flip poisoning |
| `ack2_coalition_30` | Model, coalition | 30% | Check-2 evasion: colluding clients shift the round's pseudo-gradient toward their shared poison direction |

---

## Configuration

Edit `config/fl_config.yaml` to change hyperparameters:

Dataset paths live in `config/dataset_config.yaml`.

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

> **These are the manuscript's reported values, NOT a measurement produced by
> this repository in its current state.** They are reproduced here so the
> pipeline has a reference to be checked against. The file they cite,
> `results/tables/full_comparison_results.json`, was quarantined during the
> forensic audit (see [Result Provenance](#result-provenance)) and has not been
> regenerated: `make check-results` will report it missing. Two subsequent
> implementation corrections — enabling the gate warmup annealing and
> implementing the straight-through estimator — also change the aggregation
> dynamics, so a genuine `make full-comparison` run is expected to differ from
> the table below. Treat these as the target to reproduce, not as evidence.

| Strategy | Accuracy | F1-Macro | ASR |
|---|---|---|---|
| FedAvg (no defense) | 0.6026 ± 0.0114 | 0.5726 ± 0.0089 | 0.4012 ± 0.0253 |
| Krum | 0.8170 ± 0.0089 | 0.7891 ± 0.0108 | 0.1650 ± 0.0243 |
| Trimmed Mean | 0.8038 ± 0.0097 | 0.7755 ± 0.0104 | 0.2157 ± 0.0404 |
| FLTrust | 0.8642 ± 0.0121 | 0.8349 ± 0.0107 | 0.1357 ± 0.0388 |
| FoolsGold | 0.8081 ± 0.0109 | 0.7794 ± 0.0119 | 0.1908 ± 0.0352 |
| FLAME | 0.8256 ± 0.0082 | 0.8016 ± 0.0095 | 0.1878 ± 0.0308 |
| RFA | 0.8082 ± 0.0117 | 0.7811 ± 0.0130 | 0.2016 ± 0.0226 |
| **TV-FLIDS** | **0.8801 ± 0.0095** | **0.8502 ± 0.0141** | **0.1908 ± 0.0035** |

*Values are mean ± std over 5 seeds (42, 123, 456, 789, 1337), as reported in
the manuscript. Regenerate with `make full-comparison`, which writes
`results/tables/full_comparison_results.json`; verify with `make check-results`.*

---

## Project Structure

```
tv-flids/
├── config/                    # Hyperparameter configs (YAML)
│   ├── fl_config.yaml
│   └── dataset_config.yaml
├── data/
│   ├── preprocessing/         # NSL-KDD, UNSW-NB15 pipelines
│   └── partitioning.py        # IID & Non-IID (Dirichlet) partitioners
├── extras/
│   └── mnist_fl_pipeline.py          # MNIST FL benchmark (not used in paper)
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
├── theory/
│   ├── proposition1_verification.py  # Prop. 1 numerical verification
│   └── convergence_analysis.py       # Convergence rate (τ) fitting
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

## Known Limitations

- **Byzantine threshold**: TV-FLIDS degrades when adversarial fraction exceeds ~50%.  
  At f/N > 0.5, the verification gate cannot reliably separate honest and malicious updates.
- **IID server assumption**: The server validation set used for trust scoring must be  
  class-balanced and drawn from the same distribution as the global test set.  
  Distribution shift between server val and test data is not handled.
- **Single model architecture**: All clients use the same MLP architecture.  
  Heterogeneous model support (e.g., different depths) is out of scope.
- **NSL-KDD age**: NSL-KDD is a 2009-era dataset. A full CIC-IoT-2023 data-loading,
  preprocessing, taxonomy-mapping, and model-compatibility pipeline is implemented
  (`data/preprocessing/ciciot2023_pipeline.py`, `--dataset ciciot2023`) and smoke-tested
  against synthetic data matching the dataset's schema, but the full cross-dataset
  experimental campaign (Supplementary Table S4/Figure S1) has not yet been executed
  against the real dataset — that run is out of scope for the current remediation pass
  and remains pending.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{tvflids2025,
  title   = {TV-FLIDS: Trust-Aware and Verifiable Federated Learning for
             Intrusion Detection under Adaptive Byzantine Clients},
  author  = {Ali Akarma},
  booktitle = {},
  year    = {2025},
  url     = {https://github.com/aliakarma/tv-flids}
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
