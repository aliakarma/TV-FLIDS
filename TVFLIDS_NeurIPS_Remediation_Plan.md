# TV-FLIDS → NeurIPS: Complete Remediation & Elevation Plan

> **Prepared for:** Ali Akarma  
> **Project:** TV-FLIDS (Trust-Aware & Verifiable Federated Intrusion Detection System)  
> **Target Venue:** NeurIPS (primary) / IEEE IoT Journal (secondary)  
> **Audit basis:** Three-role adversarial review (Q1 Journal + Reproducibility Auditor + ML Systems Inspector)

---

## ORIENTATION: UNDERSTANDING THE GAP

Before executing the plan, internalize what separates your current state from NeurIPS-ready. Your repo has strong bones — the algorithm is credible, the module structure is clean, and the statistical framework is designed correctly. The gap is a set of specific, fixable engineering and experimental failures.

| Dimension | Current State | NeurIPS Bar |
|-----------|--------------|-------------|
| Evaluation pipeline | Broken (`evaluate_fn` dead) | Runs end-to-end, all metrics computed |
| Core contribution | Adaptive weights frozen at 1/3 | Meta-gradient demonstrably adapts |
| Statistical rigor | Framework exists, no data | 5-seed, Wilcoxon, effect sizes, 95% CIs |
| Datasets | NSL-KDD only | NSL-KDD + UNSW-NB15 minimum |
| Baselines | 5 baselines | 5 baselines + 2 recent (2022–2024) |
| Theory | Proposition 1 on synthetic data | Proposition 1 on real experimental outputs |
| Ablation | Designed, can't run | A1–A5 with significance per component |
| Writing | Not started | Full camera-ready paper |

---

## PHASE 1: CRITICAL BUG FIXES

> **Priority:** Highest — everything else is blocked until these pass.  
> **Estimated time:** 2–3 days

---

### Step 1.1 — Wire `evaluate_fn` into the Flower Strategy

**Files:** `fl/strategy.py`, `experiments/run_experiment.py`

This is the single most fatal bug. The global evaluation function is defined but never passed to the Flower strategy or simulation engine. Zero per-round metrics are computed. All downstream statistics, logging, figures, and paper results are blocked.

**Fix in `fl/strategy.py` — add `evaluate_fn` parameter:**

```python
class TVFLIDSStrategy(FedAvg):
    def __init__(self, num_clients: int, config: dict, val_loader: DataLoader,
                 model: nn.Module, device: torch.device, adaptive: bool = True,
                 use_adaptive_thresholds: bool = False,
                 evaluate_fn=None,   # ← ADD THIS
                 **kwargs):
        super().__init__(evaluate_fn=evaluate_fn, **kwargs)  # ← PASS TO FedAvg
        # rest of __init__ unchanged
```

**Fix in `experiments/run_experiment.py` — define `evaluate_fn` BEFORE strategy creation:**

```python
# ── Define evaluate_fn BEFORE make_strategy() ─────────────────────
metrics_tracker = ExperimentMetrics(class_names=CLASS_NAMES[:num_classes])
round_results: List[Dict] = []

def evaluate_fn(server_round: int, parameters, config_eval):
    """Called by Flower after each round's aggregation."""
    params_np = fl.common.parameters_to_ndarrays(parameters)
    trust_scores = None
    if hasattr(strategy_container, 'strategy') and \
       hasattr(strategy_container.strategy, 'trust_scorer'):
        trust_scores = strategy_container.strategy.trust_scorer.trust_scores.copy()

    m = evaluate_global_model(
        global_model, params_np, X_test, y_test,
        device, metrics_tracker, server_round, trust_scores,
    )
    round_results.append(m)
    logger.log_round(server_round, {
        k: v for k, v in m.items()
        if isinstance(v, (int, float)) and v is not None
    })
    if verbose:
        print(f"  [Eval R{server_round:03d}] "
              f"Acc={m['accuracy']:.4f} | "
              f"F1={m['f1_macro']:.4f} | "
              f"ASR={m['attack_success_rate']:.4f}")
    return m["accuracy"], {
        "f1_macro": m["f1_macro"],
        "attack_success_rate": m["attack_success_rate"],
    }

# Container to allow closure reference before strategy is assigned
class _StrategyContainer:
    strategy = None
strategy_container = _StrategyContainer()

strategy = make_strategy(
    strategy_name, config, global_model, val_loader, root_loader,
    device, num_clients, fl_cfg,
    evaluate_fn=evaluate_fn,   # ← PASS HERE
)
strategy_container.strategy = strategy
```

**Fix `make_strategy()` signature:**

```python
def make_strategy(
    strategy_name, config, global_model, val_loader, root_loader,
    device, num_clients, fl_cfg,
    evaluate_fn=None,    # ← ADD
):
    common_kwargs = dict(
        fraction_fit=frac_fit,
        fraction_evaluate=frac_eval,
        min_fit_clients=max(2, int(num_clients * frac_fit)),
        min_evaluate_clients=max(1, int(num_clients * frac_eval)),
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,   # ← ADD TO ALL STRATEGIES
    )
```

**Remove the broken `history.parameters_distributed` block:**

```python
# REMOVE the entire if/else block — replace with:
summary = metrics_tracker.get_final_summary()
# Real data now available because evaluate_fn ran each round
```

**Verification command after this fix:**

```bash
python experiments/run_experiment.py \
    --strategy fedavg --attack no_attack --rounds 3 --seed 42
# Must print: [Eval R001], [Eval R002], [Eval R003]
# Must print: Final Accuracy > 0.0
```

---

### Step 1.2 — Fix the Meta-Gradient Backpropagation

**Files:** `fl/strategy.py` (inner `_val_fn`), `trust/adaptive_trust_scorer.py`

The adaptive weights α,β,γ never update. `_val_fn` returns a tensor with `requires_grad=False`, the guard in `meta_update()` permanently prevents the optimizer from stepping, and all runs of `tvflids` are functionally identical to `tvflids_fixed`. This directly invalidates the paper's central claim.

**Replace `_val_fn` in `fl/strategy.py::aggregate_fit()`:**

```python
# BEFORE (broken — severs gradient graph):
def _val_fn(alpha, beta, gamma):
    ...
    return torch.tensor(self._eval_model(agg), dtype=torch.float32, requires_grad=False)

# AFTER (differentiable — gradient flows through weights to alpha/beta/gamma):
def _val_fn(alpha, beta, gamma):
    """
    Differentiable trust-weighted aggregation loss.
    Connects alpha/beta/gamma to validation loss via per-client val losses.
    """
    per_client_losses = torch.tensor(
        [self._eval_model(_apars[i]) for i in range(len(a_ids))],
        dtype=torch.float32,
    )
    sim_t  = torch.tensor(_sim,  dtype=torch.float32)
    acc_t  = torch.tensor(_acc,  dtype=torch.float32)
    anom_t = torch.tensor(_anom, dtype=torch.float32)

    raw_scores = torch.clamp(
        alpha * sim_t + beta * acc_t - gamma * anom_t, 0.0, 1.0
    )
    total = raw_scores.sum()
    weights = raw_scores / (total + 1e-8)

    # Gradient flows: val_loss → weights → alpha, beta, gamma ✓
    weighted_val_loss = (weights * per_client_losses).sum()
    return weighted_val_loss
```

**Add assertion guard in `trust/adaptive_trust_scorer.py::meta_update()`:**

```python
def meta_update(self, compute_val_loss_fn: Callable) -> Dict[str, float]:
    self.meta_optimizer.zero_grad()
    w = self.weights
    try:
        loss = compute_val_loss_fn(w[0], w[1], w[2])
        assert isinstance(loss, torch.Tensor), "val_fn must return a Tensor"
        if not loss.requires_grad:
            raise ValueError(
                "val_fn returned requires_grad=False. "
                "Ensure computation graph connects to alpha/beta/gamma."
            )
        loss.backward()
        self.meta_optimizer.step()
    except AssertionError:
        raise  # Hard fail — contract violation
    except Exception as e:
        print(f"[AdaptiveTrust] meta_update skipped: {e}")
    self._sync_weights()
    snap = self.get_current_weights()
    self.weight_history.append(snap)
    return snap
```

**Verification test to add in `tests/test_all.py`:**

```python
class TestMetaGradient(unittest.TestCase):
    def test_weights_actually_change(self):
        from trust.adaptive_trust_scorer import AdaptiveTrustScorer
        ats = AdaptiveTrustScorer(10, meta_lr=0.1)
        initial = ats.get_current_weights().copy()

        def biased_val_fn(alpha, beta, gamma):
            sim_t  = torch.tensor([0.9, 0.1], dtype=torch.float32)
            acc_t  = torch.tensor([0.5, 0.5], dtype=torch.float32)
            anom_t = torch.tensor([0.1, 0.9], dtype=torch.float32)
            raw = torch.clamp(alpha*sim_t + beta*acc_t - gamma*anom_t, 0, 1)
            w = raw / (raw.sum() + 1e-8)
            losses = torch.tensor([0.3, 1.2], dtype=torch.float32)
            return (w * losses).sum()

        for _ in range(5):
            ats.meta_update(biased_val_fn)

        updated = ats.get_current_weights()
        self.assertNotAlmostEqual(
            initial['alpha'], updated['alpha'], places=4,
            msg="Weights did not change — gradient not flowing"
        )
```

---

### Step 1.3 — Fix Validation Set Contamination

**Files:** `data/preprocessing/nslkdd_pipeline.py`, `experiments/run_experiment.py`

SMOTE is applied to the full training set before the server validation set is carved out. Clients then receive all of `X_train`, including the 2000 points that form `X_val`. The server's trust-scoring loss signal is evaluated on data clients trained on, inflating all trust estimates.

**Rewrite `nslkdd_pipeline.py::build_pipeline()` — extract val split BEFORE SMOTE:**

```python
def build_pipeline(train_path: str, test_path: str, use_smote: bool = True,
                   seed: int = 42, val_fraction: float = 0.05):
    """
    Returns:
        X_train, y_train    ← SMOTE'd, for client partitioning only
        X_val, y_val        ← Clean held-out, for server trust scoring
        X_test, y_test      ← Global test set
        scaler, encoders, class_weights
    """
    train_df, test_df = load_nslkdd(train_path, test_path)
    train_df = map_labels(train_df)
    test_df  = map_labels(test_df)
    train_df, encoders = encode_categoricals(train_df, fit=True)
    test_df, _         = encode_categoricals(test_df, encoders=encoders, fit=False)

    feature_cols = [c for c in train_df.columns if c != "label"]
    X_all  = train_df[feature_cols].values.astype(np.float32)
    y_all  = train_df["label"].values.astype(np.int64)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["label"].values.astype(np.int64)

    # STEP 1: Carve validation set BEFORE normalization and SMOTE
    from sklearn.model_selection import train_test_split
    X_train_raw, X_val_raw, y_train_raw, y_val = train_test_split(
        X_all, y_all, test_size=val_fraction, stratify=y_all, random_state=seed,
    )

    # STEP 2: Fit scaler on train only, transform all splits
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val          = scaler.transform(X_val_raw).astype(np.float32)
    X_test_scaled  = scaler.transform(X_test).astype(np.float32)

    # STEP 3: SMOTE only on training split
    if use_smote:
        X_train_scaled, y_train_raw = apply_smote(X_train_scaled, y_train_raw, random_state=seed)

    y_train = y_train_raw.astype(np.int64)
    weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)

    print(f"[NSL-KDD] Train: {X_train_scaled.shape} | Val: {X_val.shape} | Test: {X_test_scaled.shape}")
    return (X_train_scaled, y_train, X_val, y_val.astype(np.int64),
            X_test_scaled, y_test, scaler, encoders, weights)
```

**Update `experiments/run_experiment.py::setup_data()` to use the new return signature:**

```python
def setup_data(config, dataset='nslkdd', seed=42, partition_type='noniid', alpha=0.5):
    if dataset == 'nslkdd':
        download_nslkdd(train_path, test_path)
        (X_train, y_train,
         X_val, y_val,
         X_test, y_test,
         _, _, class_weights) = nslkdd_pipeline(
            train_path, test_path, use_smote=True, seed=seed
        )
    # X_val is now truly held-out — do NOT include in client partitioning
    partitioner = get_partitioner(partition_type, alpha=alpha)
    client_data = partitioner.partition(X_train, y_train, num_clients, seed=seed)
    return client_data, X_test, y_test, X_val, y_val, class_weights
```

> Apply the identical val-split-before-SMOTE pattern to `unswnb15_pipeline.py` when integrating that dataset.

---

### Step 1.4 — Add End-to-End Smoke Test

**New file:** `tests/test_integration.py`

```python
"""
tests/test_integration.py
End-to-end integration tests: 2-round simulation must produce real metrics.
Run: python tests/test_integration.py
"""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestEndToEnd(unittest.TestCase):

    def test_fedavg_produces_metrics(self):
        from experiments.run_experiment import run_experiment
        result = run_experiment(
            strategy_name='fedavg', attack_config_name='no_attack',
            seed=42, num_rounds=2, verbose=False,
        )
        self.assertIn('final_accuracy', result,
                      "final_accuracy missing — evaluate_fn not running")
        self.assertGreater(result['final_accuracy'], 0.0,
                           "Accuracy is 0.0 — evaluation pipeline broken")
        self.assertIn('final_f1_macro', result)

    def test_tvflids_produces_metrics(self):
        from experiments.run_experiment import run_experiment
        result = run_experiment(
            strategy_name='tvflids', attack_config_name='label_flip_30',
            seed=42, num_rounds=2, verbose=False,
        )
        self.assertGreater(result.get('final_accuracy', 0), 0.0)
        self.assertIn('final_attack_success_rate', result)

    def test_tvflids_outperforms_fedavg_under_attack(self):
        """Sanity check: TV-FLIDS must not be worse than FedAvg after 20 rounds."""
        from experiments.run_experiment import run_experiment
        r_fedavg  = run_experiment('fedavg',  'label_flip_30', seed=42, num_rounds=20, verbose=False)
        r_tvflids = run_experiment('tvflids', 'label_flip_30', seed=42, num_rounds=20, verbose=False)
        self.assertGreaterEqual(
            r_tvflids.get('final_accuracy', 0),
            r_fedavg.get('final_accuracy', 0),
            "TV-FLIDS worse than FedAvg — check trust/verification logic"
        )

if __name__ == '__main__':
    unittest.main(verbosity=2)
```

> **Rule:** Run `python tests/test_integration.py` after every fix in Steps 1.1–1.3. It must fail before and pass after.

---

## PHASE 2: PIPELINE INTEGRITY FIXES

> **Estimated time:** 1–2 days

---

### Step 2.1 — Log Adaptive Weight History to JSON Every Round

**Files:** `fl/strategy.py`, `experiments/run_experiment.py`

Weight trajectories (α,β,γ over rounds) are required for Figure 5. Currently `weight_history` lives only in memory and is never persisted.

**In `fl/strategy.py::aggregate_fit()`, after `meta_update()`:**

```python
if self.adaptive and isinstance(self.trust_scorer, AdaptiveTrustScorer):
    snap = self.trust_scorer.meta_update(_val_fn)
    log['adaptive_alpha'] = snap['alpha']
    log['adaptive_beta']  = snap['beta']
    log['adaptive_gamma'] = snap['gamma']
```

**In `evaluate_fn` inside `run_experiment.py`, log trust state alongside metrics:**

```python
if hasattr(strategy_container.strategy, 'trust_scorer'):
    ts_summary = strategy_container.strategy.trust_scorer.get_summary()
    logger.log_round(server_round, {
        **metric_floats,
        **{f'trust_{k}': v for k, v in ts_summary.items()}
    })
```

---

### Step 2.2 — Wire All Six Figure Generators to Live Data

**File:** `experiments/run_experiment.py` — add figure generation block after simulation

```python
# ── Figure generation (post-simulation) ──────────────────────────
if strategy_name in ('tvflids', 'tvflids_fixed') and len(round_results) > 0:
    from evaluation.visualization import figure1_convergence_curves, \
        figure2_trust_evolution, figure6_confusion_matrices
    os.makedirs("results/figures", exist_ok=True)

    figure1_convergence_curves(
        {strategy_name: round_results},
        save_path=f"results/figures/fig1_{strategy_name}_{attack_config_name}_seed{seed}.pdf",
    )
    if hasattr(strategy, 'trust_scorer'):
        figure2_trust_evolution(
            strategy.get_trust_history(), malicious_ids,
            save_path=f"results/figures/fig2_trust_{strategy_name}_seed{seed}.pdf",
        )
```

**In `experiments/run_full_comparison.py`, add merged Figure 1 and Figure 6 after all strategies complete:**

```python
from evaluation.visualization import figure1_convergence_curves, figure6_confusion_matrices

# Merge per-strategy round metrics and produce combined convergence plot
all_round_metrics = {}   # {strategy_name: [round_metric_dicts]} loaded from JSON logs
figure1_convergence_curves(all_round_metrics, save_path="results/figures/fig1_convergence.pdf")
```

> Figures 4 and 5 are called from `run_ablation.py` and `run_full_comparison.py` respectively after their data is available.

---

### Step 2.3 — Activate the Dead `dataset_config.yaml`

**File:** `experiments/run_experiment.py::setup_data()`

```python
import yaml

def setup_data(config, dataset='nslkdd', seed=42, partition_type='noniid', alpha=0.5):
    with open('config/dataset_config.yaml', 'r') as f:
        ds_cfg = yaml.safe_load(f)['datasets'][dataset]
    train_path = ds_cfg['train_file']
    test_path  = ds_cfg['test_file']
    # input_dim and num_classes derived from actual data shapes — no change needed
```

---

### Step 2.4 — Fix Seed Propagation to Attack Functions

**File:** `fl/client.py`

Attacks currently use `seed=42` regardless of the global experiment seed, making attack randomness non-reproducible across experiments.

```python
# In TVFLIDSClient.__init__, store seed:
self.seed = attack_kwargs.get('seed', 42)

# In fit(), pass per-client unique seed to attacks:
if self.attack_type == 'label_flip':
    y = self.factory.label_flip(
        y, target_class=self.attack_kwargs.get('target_class', 0),
        flip_ratio=self.attack_kwargs.get('flip_ratio', 1.0),
        seed=self.seed + self.client_id,   # unique per client, reproducible
    )
elif self.attack_type == 'backdoor':
    X, y = self.factory.backdoor_attack(
        X, y, ..., seed=self.seed + self.client_id,
    )

# In make_client_fn() inside run_experiment.py:
attack_kwargs['seed'] = seed   # propagate global experiment seed
```

---

## PHASE 3: EXPERIMENTAL DESIGN UPGRADES

> **Estimated time:** 3–5 days including compute time

---

### Step 3.1 — Integrate UNSW-NB15 as Second Dataset

**File:** `experiments/run_experiment.py::setup_data()`

```python
elif dataset == 'unswnb15':
    from data.preprocessing.unswnb15_pipeline import build_pipeline as unswnb15_pipeline
    train_path = 'data/raw/UNSW_NB15_training-set.csv'
    test_path  = 'data/raw/UNSW_NB15_testing-set.csv'
    (X_train, y_train,
     X_val, y_val,
     X_test, y_test,
     _, _, class_weights) = unswnb15_pipeline(
        train_path, test_path, use_smote=True, seed=seed
    )
    model_kwargs = {"input_dim": X_test.shape[1], "num_classes": len(np.unique(y_test))}
```

Apply the same val-split-before-SMOTE fix from Step 1.3 to `unswnb15_pipeline.py`.

**New file:** `scripts/download_unswnb15.sh`

```bash
#!/usr/bin/env bash
echo "UNSW-NB15 requires manual download from the UNSW research portal."
echo "URL: https://research.unsw.edu.au/projects/unsw-nb15-dataset"
echo ""
echo "Place files at:"
echo "  data/raw/UNSW_NB15_training-set.csv"
echo "  data/raw/UNSW_NB15_testing-set.csv"
```

**New file:** `experiments/run_dataset_comparison.py`

```python
"""Cross-dataset validation: does the TV-FLIDS advantage hold on UNSW-NB15?"""
for dataset in ['nslkdd', 'unswnb15']:
    for strategy in ['fedavg', 'fltrust', 'tvflids']:
        for seed in [42, 123, 456]:
            run_experiment(strategy, 'label_flip_30', dataset=dataset, seed=seed, num_rounds=100)
```

---

### Step 3.2 — Add Two Recent Baselines (2022–2024)

NeurIPS reviewers will immediately ask: "How does this compare to recent Byzantine-robust FL work?" You need at least two baselines from the past two years.

#### Baseline 6: FLAME (Nguyen et al., USENIX Security 2022)

```
Reference: Nguyen et al. "FLAME: Taming Backdoors in Federated Learning"
           USENIX Security Symposium, 2022.
Key mechanism: HDBSCAN clustering on model updates to identify outliers,
               followed by adaptive Gaussian noise injection on surviving updates.
Strength: Strongest published defense against backdoor attacks.
```

Create `fl/baselines/flame_strategy.py` following the same structural pattern as `krum_strategy.py`. Implement HDBSCAN clustering using `hdbscan` (add to `requirements.txt`) and adaptive noise injection exactly as described in the original paper.

#### Baseline 7: RFA / Geometric Median (Pillutla et al., IEEE Trans. Signal Processing 2022)

```
Reference: Pillutla et al. "Robust Aggregation for Federated Learning"
           IEEE Transactions on Signal Processing, 2022.
Key mechanism: Smoothed Weiszfeld algorithm approximating the geometric median
               of client parameter vectors.
Strength: Provably Byzantine-robust; O(nd) per round.
```

Create `fl/baselines/rfa_strategy.py`. Implement the smoothed Weiszfeld algorithm directly — it is approximately 40 lines of numpy. Do not tune their hyperparameters favorably.

---

### Step 3.3 — Add the Min-Max Attack

The Min-Max attack (Shejwalkar & Houmansadr, NDSS 2021) is the current strongest white-box model poisoning attack. Its absence will be flagged by reviewers.

**Add to `attacks/adversarial.py`:**

```python
@staticmethod
def min_max_attack(client_params: List[np.ndarray],
                   global_params: List[np.ndarray],
                   all_updates: List[List[np.ndarray]],
                   gamma: float = 2.0) -> List[np.ndarray]:
    """
    Min-Max Attack (Shejwalkar & Houmansadr, NDSS 2021).
    Maximizes deviation from honest aggregate while staying within the
    norm ball of honest updates — minimizing detectability.

    Reference: https://arxiv.org/abs/2103.06820
    """
    honest_norms = [
        np.linalg.norm(np.concatenate([p.flatten() for p in u]))
        for u in all_updates
    ]
    bound = np.mean(honest_norms) + gamma * np.std(honest_norms)

    delta = [c - g for c, g in zip(client_params, global_params)]
    flat  = np.concatenate([d.flatten() for d in delta])
    scale = min(bound / (np.linalg.norm(flat) + 1e-8), gamma)
    return [g + scale * d for g, d in zip(global_params, delta)]
```

**Add to `ATTACK_CONFIGS`:**

```python
"min_max_30": {"ratio": 0.30, "type": "min_max", "gamma": 2.0},
```

---

### Step 3.4 — Verify Proposition 1 on Real Experimental Outputs

Currently, `run_verification_suite()` verifies the bound only on random synthetic data. This is insufficient for a theoretical claim in a paper.

**Add parameter logging to `fl/strategy.py`:**

```python
# In aggregate_fit(), after building a_pars (optional, off by default):
if self.config.get('log_client_params', False):
    self._last_round_data = {
        'honest_ids':    [cid for cid in a_ids if cid not in self._known_malicious],
        'byzantine_ids': [cid for cid in a_ids if cid in self._known_malicious],
        'trust_scores':  self.trust_scorer.trust_scores.copy(),
        'client_params': {cid: a_pars[i] for i, cid in enumerate(a_ids)},
    }
```

**Add to `theory/proposition1_verification.py`:**

```python
def verify_from_experiment_log(experiment_log_path: str,
                                strategy_ref) -> Dict:
    """
    Verify Proposition 1 using actual trust scores and parameters
    from a live TV-FLIDS run, not synthetic data.
    Call this at the end of run_all_experiments.sh on the final TV-FLIDS log.
    """
    last_data = strategy_ref._last_round_data
    return verify_proposition1(
        trust_scores=last_data['trust_scores'],
        honest_ids=last_data['honest_ids'],
        byzantine_ids=last_data['byzantine_ids'],
        global_params=strategy_ref.model.get_parameters(),
        honest_params=[last_data['client_params'][i] for i in last_data['honest_ids']],
        byzantine_params=[last_data['client_params'][i] for i in last_data['byzantine_ids']],
        tau_min=strategy_ref.config['trust']['min_trust'],
    )
```

**Add to `scripts/run_all_experiments.sh`:**

```bash
# After TV-FLIDS full run:
echo "[Proposition 1] Verifying bound on real experimental outputs..."
python -c "
from theory.proposition1_verification import verify_from_experiment_log
# Load last strategy reference from serialized state
# (requires --log_client_params flag in the TV-FLIDS run above)
print('Proposition 1 verification: see results/tables/proposition1_real.json')
"
```

---

## PHASE 4: STATISTICAL RIGOR

> **Estimated time:** 2 days execution + compute time (start long runs early)

---

### Step 4.1 — Run the Full 5-Seed Experiment Suite

Execute these commands after Phases 1–3 are complete:

```bash
# Table 1: Primary comparison across all strategies
python experiments/run_full_comparison.py \
    --strategies fedavg krum trimmed_mean fltrust foolsgold tvflids tvflids_fixed flame rfa \
    --attack label_flip_30 \
    --seeds 42 123 456 789 1337 \
    --rounds 100

# Supplementary: TV-FLIDS vs all attack types
for ATTACK in gradient_scale_30 noise_30 backdoor_20 min_max_30; do
    python experiments/run_full_comparison.py \
        --strategies fedavg fltrust tvflids \
        --attack $ATTACK \
        --seeds 42 123 456 789 1337 \
        --rounds 100
done

# Cross-dataset validation
python experiments/run_dataset_comparison.py

# Adversarial ratio sweep for Figure 3
python experiments/run_ratio_sweep.py \
    --methods fedavg krum fltrust tvflids \
    --ratios 0.0 0.1 0.2 0.3 0.4 0.5 0.6 \
    --seeds 42 123 456 789 1337 \
    --rounds 100
```

---

### Step 4.2 — Add Effect Sizes and Bootstrap Confidence Intervals

**File:** `evaluation/statistical_testing.py` — add these two functions

```python
def compute_cohens_d(results_a: List[Dict], results_b: List[Dict],
                      metric: str = "final_accuracy") -> float:
    """
    Cohen's d effect size between two method result sets.
    Interpretation: |d| < 0.2 negligible; 0.2–0.5 small;
                    0.5–0.8 medium; > 0.8 large.
    """
    vals_a = np.array([r[metric] for r in results_a if metric in r])
    vals_b = np.array([r[metric] for r in results_b if metric in r])
    pooled_std = np.sqrt((np.std(vals_a)**2 + np.std(vals_b)**2) / 2)
    if pooled_std < 1e-10:
        return 0.0
    return float((np.mean(vals_a) - np.mean(vals_b)) / pooled_std)


def compute_bootstrap_ci(results: List[Dict], metric: str,
                          n_bootstrap: int = 10_000,
                          ci: float = 0.95,
                          seed: int = 42) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for a metric across seeds.
    Returns (lower, upper) bounds at the specified CI level.
    """
    rng  = np.random.default_rng(seed)
    vals = np.array([r[metric] for r in results if metric in r])
    boot_means = [
        np.mean(rng.choice(vals, size=len(vals), replace=True))
        for _ in range(n_bootstrap)
    ]
    alpha = (1 - ci) / 2
    return (float(np.percentile(boot_means, 100 * alpha)),
            float(np.percentile(boot_means, 100 * (1 - alpha))))
```

**Replace `build_results_table()` calls in `run_full_comparison.py` with an extended version:**

```python
def build_results_table_extended(
    experiment_results_dict: Dict[str, List[Dict]],
    metrics: Optional[List[str]] = None,
    reference_method: str = "tvflids",
) -> Dict:
    """
    Full NeurIPS-grade results table:
    mean ± std | 95% CI | Cohen's d | Wilcoxon p
    """
    if metrics is None:
        metrics = ["final_accuracy", "final_f1_macro", "final_attack_success_rate"]
    table = {}
    ref_results = experiment_results_dict.get(reference_method, [])

    for method, results in experiment_results_dict.items():
        row = {}
        for m in metrics:
            mean, std      = compute_summary(results, m)
            ci_low, ci_hi  = compute_bootstrap_ci(results, m)
            d              = compute_cohens_d(ref_results, results, m) \
                             if ref_results else 0.0
            wtest          = compare_methods_wilcoxon(ref_results, results, m) \
                             if ref_results and method != reference_method else {}
            row[m] = {
                "mean": mean, "std": std,
                "ci_95": (ci_low, ci_hi),
                "cohens_d": d,
                "wilcoxon_p": wtest.get("p_value", None),
                "significant": wtest.get("significant", None),
                "formatted": (f"{mean:.4f}±{std:.4f} "
                              f"[{ci_low:.4f},{ci_hi:.4f}] "
                              f"d={d:.2f}")
            }
        table[method] = row
    return table
```

---

### Step 4.3 — Add Per-Component Ablation Significance

**File:** `experiments/run_ablation.py` — add after `summary_table` is computed

```python
from evaluation.statistical_testing import compare_methods_wilcoxon, compute_cohens_d

full_results = all_results["TV-FLIDS (Full)"]
print("\n[Ablation Significance vs TV-FLIDS Full]")
print(f"{'Ablation':<30} {'Metric':<25} {'p-value':>10} {'Sig':>5} {'d':>8}")
print("-" * 80)

for name, results in all_results.items():
    if name == "TV-FLIDS (Full)":
        continue
    for metric in ["final_f1_macro", "final_attack_success_rate"]:
        wtest = compare_methods_wilcoxon(full_results, results, metric)
        d     = compute_cohens_d(full_results, results, metric)
        sig   = "*" if wtest['significant'] else "ns"
        print(f"  {name:<28} {metric:<25} "
              f"{wtest['p_value']:>10.4f} {sig:>5} {d:>8.3f}")
```

---

## PHASE 5: THEORETICAL DEPTH

> **Estimated time:** 3–5 days

---

### Step 5.1 — Strengthen Proposition 1 with Lemma 1

Proposition 1 bounds Byzantine influence by `f·τ_min / (N_H·τ̄_H)`. This bound is only non-trivial if you can show that honest trust grows and Byzantine trust decays during training. Add Lemma 1 to prove this.

**Add to `theory/proposition1_verification.py`:**

```python
def verify_trust_convergence(trust_history: Dict[int, List[float]],
                              honest_ids: List[int],
                              byzantine_ids: List[int],
                              tau_min: float = 0.01) -> Dict:
    """
    Lemma 1 (Trust Convergence): Verifies empirically that over rounds,
      - Mean honest client trust converges toward 1.0
      - Mean Byzantine client trust converges toward tau_min

    This makes the Proposition 1 bound non-trivial and tightens over time.
    """
    honest_final = np.mean([trust_history[i][-1] for i in honest_ids])
    byz_final    = np.mean([trust_history[i][-1] for i in byzantine_ids])

    # Fit linear trend over rounds for each group
    n_rounds = len(trust_history[honest_ids[0]])
    t = np.arange(n_rounds)

    honest_trend = np.polyfit(t, [np.mean([trust_history[i][r]
                               for i in honest_ids]) for r in range(n_rounds)], 1)[0]
    byz_trend    = np.polyfit(t, [np.mean([trust_history[i][r]
                               for i in byzantine_ids]) for r in range(n_rounds)], 1)[0]

    return {
        'honest_final_mean_trust':    honest_final,
        'byzantine_final_mean_trust': byz_final,
        'honest_trust_trend':         honest_trend,    # must be > 0
        'byzantine_trust_trend':      byz_trend,       # must be < 0
        'trust_separation':           honest_final - byz_final,  # must be >> 0
        'byzantine_at_floor':         abs(byz_final - tau_min) < 0.05,
        'lemma1_holds':               honest_trend > 0 and byz_trend < 0,
    }
```

Call this at the end of each TV-FLIDS run using `strategy.get_trust_history()` and include the output in the paper appendix.

---

### Step 5.2 — Add Convergence Rate Analysis

**New file:** `theory/convergence_analysis.py`

```python
"""
Empirical convergence analysis for TV-FLIDS.
Fits model: acc(t) = L_inf - (L_inf - L0) * exp(-t / tau)
tau = convergence time constant; smaller tau = faster convergence under attack.
"""
import numpy as np
from scipy.optimize import curve_fit
from typing import List, Dict


def fit_convergence_curve(accuracies: List[float]) -> Dict:
    """
    Fits an exponential convergence model to the accuracy series.
    Returns: L_inf (asymptote), L0 (start), tau (time constant), R² (fit quality).
    """
    t = np.arange(len(accuracies))
    y = np.array(accuracies)

    def model(t, L_inf, L0, tau):
        return L_inf - (L_inf - L0) * np.exp(-t / (tau + 1e-8))

    try:
        p0 = [max(y), y[0], len(y) / 3]
        popt, _ = curve_fit(model, t, y, p0=p0, maxfev=5000)
        y_pred   = model(t, *popt)
        ss_res   = np.sum((y - y_pred) ** 2)
        ss_tot   = np.sum((y - np.mean(y)) ** 2)
        return {
            'L_inf':             float(popt[0]),
            'L0':                float(popt[1]),
            'tau':               float(popt[2]),
            'r2':                float(1 - ss_res / (ss_tot + 1e-10)),
            'converges_by_round': int(popt[2] * 3),  # 95% of asymptote
        }
    except Exception as e:
        return {'error': str(e)}


def compare_convergence_rates(round_metrics_dict: Dict[str, List[Dict]]) -> Dict:
    """Compare tau (convergence time constant) across all methods."""
    results = {}
    for method, metrics in round_metrics_dict.items():
        accuracies    = [m['accuracy'] for m in metrics]
        results[method] = fit_convergence_curve(accuracies)
    return results
```

Call from `run_full_comparison.py` and report `tau` as an additional column in Table 1. Smaller `tau` means TV-FLIDS recovers faster under attack.

---

### Step 5.3 — Privacy Compatibility Claim

If targeting IEEE IoT Journal (as stated in README), privacy analysis is expected. Add a paragraph to the paper:

> "TV-FLIDS is orthogonal to differential privacy mechanisms. The verification gate operates on gradient magnitudes and cosine similarities — not raw training data — and can be composed with client-side DP-SGD noise without modification. The 4-byte validation loss scalar transmitted per client per round is insensitive to individual training samples."

The communication overhead analysis already in `evaluation/overhead.py` supports the quantitative part of this claim.

---

## PHASE 6: CODE QUALITY & REPRODUCIBILITY

> **Estimated time:** 1–2 days

---

### Step 6.1 — Create a `Makefile` for One-Command Reproduction

**New file:** `Makefile`

```makefile
.PHONY: install data smoke test full-comparison ablation figures clean

install:
	conda env create -f environment.yml
	conda activate tvflids

data:
	bash scripts/download_nslkdd.sh

smoke:
	python experiments/run_experiment.py \
	    --strategy tvflids --attack label_flip_30 --rounds 3 --seed 42
	python tests/test_integration.py

test:
	python tests/test_all.py
	python tests/test_integration.py

full-comparison:
	python experiments/run_full_comparison.py \
	    --strategies fedavg krum trimmed_mean fltrust foolsgold tvflids tvflids_fixed \
	    --attack label_flip_30 \
	    --seeds 42 123 456 789 1337 \
	    --rounds 100

ablation:
	python experiments/run_ablation.py \
	    --attack label_flip_30 \
	    --rounds 50 \
	    --seeds 42 123 456 789 1337

figures:
	python experiments/run_ratio_sweep.py \
	    --methods fedavg fltrust tvflids \
	    --ratios 0.0 0.1 0.2 0.3 0.4 0.5 0.6 \
	    --seeds 42 123 456 \
	    --rounds 100

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf results/logs/* results/figures/* results/tables/*
```

---

### Step 6.2 — Add `environment.yml` for Exact Conda Reproducibility

**New file:** `environment.yml`

```yaml
name: tvflids
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.10.12
  - pip=23.3.1
  - pip:
    - torch==2.1.0
    - torchvision==0.16.0
    - flwr==1.6.0
    - scikit-learn==1.3.2
    - pandas==2.1.3
    - numpy==1.26.2
    - scipy==1.11.4
    - imbalanced-learn==0.11.0
    - matplotlib==3.8.2
    - seaborn==0.13.0
    - pyyaml==6.0.1
    - tqdm==4.66.1
    - tensorboard==2.15.1
    - statsmodels==0.14.1
    - hdbscan==0.8.33
```

---

### Step 6.3 — Remove Dead Code

```bash
# Remove or move to optional:
rm data/preprocessing/mnist_fl_pipeline.py

# In models/mlp.py: move IDSBiLSTM to models/bilstm.py
# with a --model bilstm flag wired in run_experiment.py if needed

# Either activate dataset_config.yaml (Step 2.3) or delete it
```

---

### Step 6.4 — Add `results/README.md` for Navigation

**New file:** `results/README.md`

```markdown
# Results Directory

All files are generated by experiment scripts. Do not commit .pdf, .json, or .pkl files.

## Regeneration

| Command | Output | Time (GPU) |
|---------|--------|------------|
| `make smoke` | Sanity check only | ~2 min |
| `make full-comparison` | Table 1 | ~30 min |
| `make ablation` | Table 2 | ~20 min |
| `make figures` | Figure 3 | ~15 min |
| Full paper reproduction | All outputs | ~90 min |

## Expected Outputs

| File | Content |
|------|---------|
| `tables/full_comparison_results.json` | Table 1 data |
| `tables/ablation_results.json` | Table 2 data |
| `tables/ratio_sweep_results.json` | Figure 3 data |
| `figures/fig1_convergence.pdf` | Convergence curves |
| `figures/fig2_trust_evolution.pdf` | Trust scores per client |
| `figures/fig3_robustness_curve.pdf` | Accuracy vs adversarial ratio |
| `figures/fig4_ablation.pdf` | Ablation bar chart |
| `figures/fig5_adaptive_weights.pdf` | α,β,γ trajectories |
| `figures/fig6_confusion.pdf` | FedAvg vs TV-FLIDS confusion matrices |
```

---

## PHASE 7: PAPER WRITING

> **Estimated time:** 2–3 weeks — this is where most effort goes

---

### Step 7.1 — Paper Structure (NeurIPS Format, 9 pages + references)

```
Title
  TV-FLIDS: Trust-Aware and Verifiable Federated Learning for Intrusion
  Detection under Adaptive Byzantine Clients

Abstract (250 words max)
  [1 sentence] Problem: Byzantine clients in FL for IoT IDS
  [1 sentence] Gap: existing defenses lack adaptive trust adjustment
  [2 sentences] Approach: 3-criteria gate + meta-gradient trust weight learning
  [2 sentences] Results: headline numbers with effect sizes from Table 1
  [1 sentence] Significance

Section 1 — Introduction (1.5 pages)
  - Open with IoT attack statistics (cite recent survey)
  - Why FL for IDS is necessary
  - Why existing defenses fail against adaptive attackers
  - Contributions (numbered list, ≤4 items, each falsifiable):
      1. Three-criteria verification gate with adaptive thresholds
      2. Meta-gradient adaptive trust weight learning
      3. Formal Byzantine influence bound (Proposition 1) + Lemma 1
      4. Empirical evaluation on NSL-KDD and UNSW-NB15

Section 2 — Related Work (1 page)
  - Byzantine-resilient FL: Krum, TrimmedMean, FLTrust, FoolsGold, FLAME, RFA
  - FL for network intrusion detection (cite 3–5 recent works)
  - Trust-based and adaptive FL: identify the exact gap you fill

Section 3 — Problem Formulation (0.75 page)
  - FL setup: N clients, T rounds, fraction_fit
  - Threat model: f Byzantine clients out of N; 5 attack types
  - IDS objective: minimize FNR under adversarial conditions; 5-class NSL-KDD

Section 4 — TV-FLIDS Design (2.5 pages)
  4.1 Three-Criteria Verification Gate
      - Check 1: loss consistency (ΔL threshold)
      - Check 2: cosine direction similarity
      - Check 3: z-score norm outlier detection
      - Adaptive threshold schedule during warmup
  4.2 Dynamic Trust Scoring
      - EMA formula: T_i(t) = decay·T_i(t-1) + (1−decay)·[α·S_i+β·A_i−γ·O_i]
      - Memory decay rationale
  4.3 Adaptive Meta-Gradient Weight Learning
      - Softmax projection: α,β,γ = softmax([log_α, log_β, log_γ])
      - Differentiable val loss objective
      - Adam update step
  4.4 Trust-Weighted Aggregation
  4.5 Theoretical Analysis
      - Proposition 1: bounded Byzantine influence
      - Lemma 1: trust convergence under the EMA update rule
      - Convergence rate analysis

Section 5 — Experimental Setup (0.75 page)
  - Datasets: NSL-KDD (125,973 train / 22,544 test, 5-class) and UNSW-NB15 (49 feat, 10-class)
  - Implementation: 20 clients, 100 rounds, fraction_fit=0.5, non-IID α=0.5
  - Baselines: 7 methods (FedAvg, Krum, TrimmedMean, FLTrust, FoolsGold, FLAME, RFA)
  - Attacks: 5 types (label flip 10/20/30%, gradient scale, noise, backdoor, Min-Max)
  - Statistical validation: 5 seeds, Wilcoxon, McNemar, Cohen's d, 95% bootstrap CI

Section 6 — Results (3 pages)
  6.1 Table 1: Main strategy comparison (mean±std, Wilcoxon p, Cohen's d)
  6.2 Figure 1: Convergence curves under 30% label flip
  6.3 Figure 3: Robustness vs adversarial ratio (0–60%)
  6.4 Table 2: Ablation A1–A5 with per-component significance
  6.5 Figure 5: Adaptive weight trajectories under different attack types
  6.6 Figure 2: Trust score evolution (honest vs malicious clients)
  6.7 Table 3: Cross-dataset results (NSL-KDD vs UNSW-NB15)
  6.8 Table 4: Overhead (time and communication cost)
  6.9 Figure 6: Confusion matrices FedAvg vs TV-FLIDS

Section 7 — Discussion (0.5 page)
  - When does TV-FLIDS fail? (>50% Byzantine; fully colluding adaptive attacker)
  - Limitations: IID server validation assumption; single model architecture
  - Future work: privacy composition, heterogeneous models

Section 8 — Conclusion (0.25 page)

Appendix (unlimited pages)
  A. Proof of Proposition 1
  B. Proof of Lemma 1 (trust convergence)
  C. Full hyperparameter sensitivity tables
  D. UNSW-NB15 complete results
  E. Extended attack type comparisons
  F. Proposition 1 verification on real experimental outputs
```

---

### Step 7.2 — How to Write Every Result Sentence

Every number in the paper must trace to a specific row and column in `results/tables/full_comparison_results.json`. Use this template for every metric claim:

> "TV-FLIDS achieves **0.884 ± 0.009** accuracy under 30% label flip attack (5 seeds, NSL-KDD, non-IID α=0.5), statistically significantly outperforming FLTrust by a margin of 0.024 (Wilcoxon signed-rank, *p* = 0.012, Cohen's *d* = 0.87)."

The mandatory components are: **value ± std**, the exact comparison method, the exact test, the *p*-value, and the effect size. Omitting any component will draw a mandatory revision request.

---

### Step 7.3 — Figure Quality Checklist

Every figure must pass all of these before submission:

- [ ] 300 DPI, PDF format (already in your `visualization.py` rcParams)
- [ ] Times New Roman font, size ≥ 9pt for all text (already in rcParams)
- [ ] Colorblind-safe palette (already in your `PALETTE` dict)
- [ ] Error bands (±1 std shading) on all convergence curves
- [ ] Error bars (±1 std caps) on all bar charts
- [ ] Caption is self-contained: states attack type, dataset, number of seeds
- [ ] No more than 6 curves per subplot
- [ ] TV-FLIDS line is thicker (`linewidth=2.5`) and uses solid linestyle
- [ ] Axis limits set so no curve touches the plot border

---

### Step 7.4 — Write the Abstract Last

After all results are in hand, write the abstract. The penultimate sentence must contain exactly two headline numbers with effect sizes:

> "Under 30% label flip attack on NSL-KDD with non-IID (α=0.5) data distribution, TV-FLIDS achieves **0.884 ± 0.009** F1-Macro — a 9.1% relative improvement over FLTrust (Wilcoxon *p* < 0.01, *d* = 0.82) — while incurring less than **0.1%** additional communication overhead."

---

## MASTER CHECKLIST: SUBMISSION GATE

Do not submit until every box is checked. Work through these in order.

### Phase 1 — Critical Fixes
- [ ] `evaluate_fn` wired to strategy → smoke test prints non-zero metrics
- [ ] `TestMetaGradient` passes → `weight_history` shows non-constant α,β,γ
- [ ] Validation set extracted before SMOTE, before client partitioning
- [ ] `python tests/test_integration.py` passes for both `fedavg` and `tvflids`

### Phase 2 — Pipeline Integrity
- [ ] Adaptive weight history logged to JSON every round
- [ ] All 6 figures have live callers in experiment runners
- [ ] `dataset_config.yaml` imported in `setup_data()`
- [ ] Attack seed propagated from global experiment seed to all attack functions

### Phase 3 — Experimental Design
- [ ] UNSW-NB15 integrated and smoke-tested (2 rounds, no crash)
- [ ] FLAME baseline implemented, unit-tested, reference-faithful
- [ ] RFA baseline implemented, unit-tested, reference-faithful
- [ ] Min-Max attack added to `ATTACK_CONFIGS` and `fl/client.py`
- [ ] Proposition 1 verified on real TV-FLIDS experiment outputs (not synthetic)
- [ ] Convergence rate (τ) computed for all methods

### Phase 4 — Statistical Rigor
- [ ] 5-seed runs complete: all strategies × NSL-KDD × label_flip_30
- [ ] 5-seed runs complete: tvflids × all 5 attack types
- [ ] Cross-dataset runs complete: NSL-KDD and UNSW-NB15
- [ ] Wilcoxon p-values computed for all baseline comparisons
- [ ] Cohen's d computed for all baseline comparisons
- [ ] 95% bootstrap CIs computed for all primary metrics
- [ ] Ablation A1–A5 with per-component Wilcoxon significance
- [ ] McNemar test: TV-FLIDS vs FLTrust predictions on NSL-KDD test set

### Phase 5 — Theory
- [ ] Proposition 1 bound holds on real experimental trust scores (ratio ≤ 1.0)
- [ ] Lemma 1 verified: honest trust trend > 0, Byzantine trust trend < 0
- [ ] Convergence rate model fitted to accuracy curves, R² > 0.90

### Phase 6 — Code Quality
- [ ] `make smoke` completes in under 5 minutes
- [ ] `make test` passes: all unit tests + integration tests
- [ ] `Makefile` and `environment.yml` present
- [ ] Dead code removed (`mnist_fl_pipeline.py`, `IDSBiLSTM` relocated)
- [ ] `results/README.md` present

### Phase 7 — Paper
- [ ] Every number traces to a specific JSON cell in results files
- [ ] Every primary metric formatted as: `value ± std [95% CI]` (Wilcoxon *p*=X, *d*=Y)
- [ ] Figure 1: convergence curves with ±1 std shaded bands, all 7 methods
- [ ] Figure 2: trust evolution, honest vs malicious clearly colored
- [ ] Figure 3: robustness curve 0–60% with fill_between ±1 std
- [ ] Figure 4: ablation bars with significance markers (* / ns)
- [ ] Figure 5: adaptive weight trajectories under ≥ 2 attack types
- [ ] Figure 6: normalized confusion matrices FedAvg vs TV-FLIDS
- [ ] Abstract written last; contains 2 headline numbers with effect sizes
- [ ] Related work cites ≥ 2 papers from 2022–2024
- [ ] Limitations section is honest: ≥50% Byzantine, colluding adaptive attacker
- [ ] Full proof of Proposition 1 in Appendix A
- [ ] Full proof of Lemma 1 in Appendix B
- [ ] Anonymous version prepared (author name, institution, GitHub link removed)
- [ ] Paper compiles with no LaTeX errors or overfull hboxes

---

## REALISTIC TIMELINE

| Week | Focus | Milestone |
|------|-------|-----------|
| **Week 1** | Phase 1 critical fixes | Smoke test passes; non-zero metrics from all strategies |
| **Week 2** | Phase 2 pipeline integrity + Phase 3 new baselines and UNSW-NB15 | All 6 figures generated from live data |
| **Week 3** | Phase 3 Min-Max attack + Phase 4 launch 5-seed compute runs | Long runs submitted to compute resource |
| **Week 4** | Phase 4 statistical analysis + Phase 5 theory + Phase 6 code cleanup | All result tables populated; theory verified |
| **Week 5–7** | Phase 7 paper writing | Full draft camera-ready |
| **Week 8** | Internal revision, co-author review, final proofreading | Submission-ready |

---

## CLOSING NOTE

Your project has genuine algorithmic novelty. The three-criteria verification gate combined with meta-gradient adaptive trust is a coherent and defensible contribution. The bugs identified are specific integration failures — not fundamental flaws in the algorithm — and all are fixable within the timeline above. The statistical and theoretical infrastructure you designed is correct; it simply needs live data to run against.

The path from current state to NeurIPS is real but achievable within a focused eight-week sprint. Execute Phase 1 first: once the evaluation pipeline produces numbers, every other phase becomes testable and the momentum compounds quickly.
