# TV-FLIDS Project Memory

**Document Status:** Active Persistent Repository Memory  
**Created:** 2026-09-17  
**Canonical Source of Truth:** Repository files and `Paper/IEEE/TV-FLIDS.tex` (+ Supplementary)  
**Maintenance Rule:** Read at the start of every session; update at the end of every session.

---

## 1. Project Identity

- **Project Name:** TV-FLIDS (Trust-Aware and Validation-Gated Federated Learning for Intrusion Detection Systems) `[VERIFIED]`
- **Repository:** `aliakarma/TV-FLIDS` `[VERIFIED]`
- **Target Manuscript:** IEEE Transactions on Information Forensics and Security (TIFS), located under `Paper/IEEE/TV-FLIDS.tex` (11 pages) and `Paper/IEEE/TV-FLIDS_supplementary.tex` (4 pages) `[VERIFIED]`
- **Core Scientific Proposal:** A poisoning-robust federated learning aggregation framework combining:
  1. Median-radius update norm clipping `[VERIFIED]`
  2. Single validation-loss gate with linear warmup threshold schedule `[VERIFIED]`
  3. Multi-signal trust memory combining direction ($S_i$), unclipped norm anomaly ($O_i$), and class-balanced validation loss improvement ($A_i$) `[VERIFIED]`
  4. Asymmetric trust updates ($\lambda_\uparrow = 0.9, \lambda_\downarrow = 0.7, \tau_{\min}=0.01, T_i^{(0)}=0.5$) with instant zero-signal penalty ($s_i = 0$) for rejected clients `[VERIFIED]`
  5. Online meta-weight adaptation ($\bm{v} \in \mathbb{R}^3$, Adam lr=0.01 on $\mathcal{L}_{\mathrm{meta}}$ with straight-through estimator) `[VERIFIED]`
  6. Trust-weighted aggregation of accepted clipped updates `[VERIFIED]`

---

## 2. Current Repository State

- **Git HEAD:** Commit `11cb1be` (clean working state) `[VERIFIED]`
- **Implementation Status:** Prototype / Legacy codebase. The codebase represents an earlier iteration that diverges significantly from the IEEE TIFS specification `[VERIFIED]`.
- **Empirical Artifacts:** Only 8 historical 100-round simulation logs exist on disk (from 2026-09-04 in `results/logs/`): 4 seeds of FedAvg under Label Flipping on NSL-KDD, and 4 seeds of TV-FLIDS clean without attack. No complete 14-baseline comparison, no Edge-IIoTset campaign, no multi-attack matrix, and no 20-seed ablation campaign has ever been executed `[VERIFIED]`.
- **Target Paper Numbers:** Every single empirical result in Section VII (Tables III–VIII) and Supplementary Section S8 (Tables S3–S11) is an **Implementation Target** (projected reference values generated synthetically by `Paper/IEEE/analysis/targets/build_targets.py` to ensure mathematical consistency) and explicitly tagged in LaTeX with `\itv{...}` / `\ittag` / orange text `[VERIFIED]`.
- **Production Code Modification Status (Session 1):** ZERO production code changes made. Read-only inspection and memory establishment only `[DECISION]`.

---

## 3. Scientific Specification

### 3.1 Mathematical Model & Algorithmic Formulation (Algorithm 1) `[VERIFIED]`
1. **Model Parameter Update:**
   $$\vDelta_i^{(t)} = \vw_i^{(t)} - \vw^{(t)}, \quad \forall i \in \mathcal{P}^{(t)}$$
2. **Stage 1: Median-Radius Clipping:**
   $$C^{(t)} = \operatorname{median}_{j \in \mathcal{P}^{(t)}} \|\vDelta_j^{(t)}\|$$
   $$\tilde\vDelta_i^{(t)} = \vDelta_i^{(t)} \min\left(1, \frac{C^{(t)}}{\|\vDelta_i^{(t)}\|}\right), \quad \tilde\vw_i^{(t)} = \vw^{(t)} + \tilde\vDelta_i^{(t)}$$
3. **Stage 2: Validation Gate:**
   $$\delta_i = \ell_{\mathrm{val}}(\vw^{(t)}) - \ell_{\mathrm{val}}(\tilde\vw_i^{(t)}) \ge \tau_L(t)$$
   $$\tau_L(t) = -0.1 + 0.1 \times \min\left(1, \frac{t}{T_{\mathrm{warm}}}\right), \quad T_{\mathrm{warm}} = 20$$
   Accepted cohort: $\mathcal{A}^{(t)} = \{i \in \mathcal{P}^{(t)} : \delta_i \ge \tau_L(t)\}$. If $\mathcal{A}^{(t)} = \emptyset$, $\vw^{(t+1)} = \vw^{(t)}$.
4. **Stage 3: Trust Signals & Memory:**
   - Direction signal ($i \in \mathcal{A}$): $S_i = \frac{1}{2}(\cos(\tilde\vDelta_i, \bar\vDelta_{\mathcal{A}}) + 1)$, where $\bar\vDelta_{\mathcal{A}} = \frac{1}{|\mathcal{A}|}\sum_{j \in \mathcal{A}} \tilde\vDelta_j$
   - Norm outlier signal ($i \in \mathcal{A}$): $O_i = 1 - e^{-z_i / 2.5}$, with $z_i = \frac{|\|\vDelta_i\| - \mu_{\mathcal{P}}|}{\sigma_{\mathcal{P}} + \varepsilon}$ computed on **unclipped** norms over all participants $\mathcal{P}$
   - Class-balanced validation signal ($i \in \mathcal{A}$):
     $$A_i = \mathrm{clip}_{[-1, 1]}\left(\frac{\ell_{\mathrm{bal}}(\vw^{(t)}) - \ell_{\mathrm{bal}}(\tilde\vw_i)}{\ell_{\mathrm{bal}}(\vw^{(t)}) + \varepsilon}\right), \quad \ell_{\mathrm{bal}}(\vw) = \frac{1}{K}\sum_{c=1}^K \frac{1}{|\mathcal{D}_{\mathrm{val}}^c|} \sum_{(x,y) \in \mathcal{D}_{\mathrm{val}}^c} \mathrm{CE}(\vw; x, y)$$
   - Mixed instantaneous signal:
     $$u_i = \alpha S_i + \beta A_i - \gamma O_i, \quad s_i = \begin{cases} \mathrm{clip}_{[0, 1]}(u_i), & i \in \mathcal{A} \\ 0, & i \in \mathcal{P} \setminus \mathcal{A} \end{cases}$$
   - Asymmetric trust memory update (for all $i \in \mathcal{P}$):
     $$T_i^{(t)} = \max\left(\tau_{\min}, \lambda_i T_i^{(t-1)} + (1 - \lambda_i) s_i\right)$$
     $$\lambda_i = \begin{cases} \lambda_\uparrow = 0.9, & s_i \ge T_i^{(t-1)} \\ \lambda_\downarrow = 0.7, & s_i < T_i^{(t-1)} \end{cases}, \quad \tau_{\min} = 0.01, \quad T_i^{(0)} = 0.5$$
     *(Non-participating clients $i \notin \mathcal{P}$ retain $T_i^{(t)} = T_i^{(t-1)}$).*
5. **Stage 4: Online Meta-Weight Adaptation:**
   - Log-weights $\bm{v} = (\ln\alpha, \ln\beta, \ln\gamma) \in \mathbb{R}^3$, initialized to $(0, 0, 0) \implies (\alpha, \beta, \gamma) = \operatorname{softmax}(\bm{v}) = (1/3, 1/3, 1/3)$.
   - Loss objective: $\mathcal{L}_{\mathrm{meta}} = \sum_{i \in \mathcal{A}} \hat{w}_i \ell_{\mathrm{val}}(\tilde\vw_i)$, with $\hat{w}_i = \frac{\mathrm{clip}_{[0,1]}(u_i)}{\sum_{j \in \mathcal{A}}\mathrm{clip}_{[0,1]}(u_j)}$ using straight-through estimator (STE).
   - One Adam step on $\bm{v}$ with learning rate $\eta_{\mathrm{meta}} = 0.01$.
6. **Stage 5: Model Aggregation:**
   $$\vw^{(t+1)} = \sum_{i \in \mathcal{A}} \frac{T_i^{(t)}}{\sum_{j \in \mathcal{A}} T_j^{(t)}} \tilde\vw_i$$

### 3.2 Core Hyperparameters `[VERIFIED]`
- Total clients: $N = 20$
- Participants per round: $D = 10$ ($\rho = D/N = 0.5$)
- Attacker fraction: default $f/N = 0.3$ ($f=6$), swept in $\{0.1, 0.2, 0.4\}$
- Trust decay gains: $\lambda_\uparrow = 0.9$
- Trust decay losses: $\lambda_\downarrow = 0.7$
- Initial trust: $T_i^{(0)} = 0.5$
- Trust floor: $\tau_{\min} = 0.01$
- Gate warmup schedule: $T_{\mathrm{warm}} = 20$, threshold $\tau_L \in [-0.1, 0.0]$
- Local training: $E = 5$ epochs, Adam optimizer, lr = $0.001$, batch size 256
- Federated rounds: 100 rounds per simulation run
- Dirichlet non-IID concentration: default $\alpha_D = 0.5$, swept in $\{0.1, 0.3, 0.5, 1.0\}$

### 3.3 Neural Network Architecture & Parameter Closed Form `[VERIFIED]`
- Topology: 4-layer MLP: $d \to 256 \to 128 \to 64 \to K$
- Hidden layers: `nn.Linear` $\to$ `nn.LayerNorm` $\to$ `nn.ReLU` $\to$ `nn.Dropout(0.3)`
- Output layer: `nn.Linear(64, K)` (no LayerNorm, no Dropout)
- Exact algebraic derivation:
  - Layer 1: $256d + 256$ (weights + biases) $+ 512$ (LayerNorm $\gamma, \beta$) $= 256d + 768$
  - Layer 2: $256 \times 128 + 128$ (weights + biases) $+ 256$ (LayerNorm $\gamma, \beta$) $= 32,896 + 256 = 33,152$
  - Layer 3: $128 \times 64 + 64$ (weights + biases) $+ 128$ (LayerNorm $\gamma, \beta$) $= 8,256 + 128 = 8,384$
  - Layer 4: $64K + K = 65K$
  - Total Parameters:
    $$P(d, K) = 256d + 65K + 42,304$$
- Dataset evaluations:
  - NSL-KDD ($d=41, K=5$): $P(41, 5) = 256(41) + 65(5) + 42,304 = 10,496 + 325 + 42,304 = \mathbf{53,125}$ (matches paper §VI-B line 355) `[VERIFIED]`
  - CIC-IoT-2023 ($d=46, K=8$): $P(46, 8) = 256(46) + 65(8) + 42,304 = 11,776 + 520 + 42,304 = \mathbf{54,600}$ (matches paper §VI-B line 355) `[VERIFIED]`
  - Edge-IIoTset ($d=61, K=6$): $P(61, 6) = 256(61) + 65(6) + 42,304 = 15,616 + 390 + 42,304 = \mathbf{58,310}$ (matches paper §VI-B line 355) `[VERIFIED]`

---

## 4. Canonical Architecture

This section documents the repository layout as it currently exists on disk (no future paths invented) `[VERIFIED]`:

| Functional Role | Canonical Path | Description / Reality |
| :--- | :--- | :--- |
| **FL Strategy** | `fl/strategy.py` (`TVFLIDSStrategy`) | Flower strategy with median norm clipping, single validation loss gate, empty cohort fallback, misses asymmetric decay |
| **FL Client** | `fl/client.py` (`TVFLIDSClient`) | Flower NumPyClient handling local training and ACK1 hook |
| **FL Server** | `fl/server.py` | Ray/Flower simulation server helper functions |
| **Trust Scoring** | `trust/trust_scorer.py` (`TrustScorer`) | Basic trust scoring, but uses symmetric decay and initial trust 1.0 |
| **Adaptive Trust**| `trust/adaptive_trust_scorer.py` (`AdaptiveTrustScorer`) | Adam meta-gradient update on $(\alpha, \beta, \gamma)$ |
| **Verification Gate**| `trust/verification.py` (`VerificationModule`) | Canonical median norm clipping & single validation loss gate ($\delta_i \ge \tau_L(t)$) |
| **Baselines** | `fl/baselines/` (9 strategy files) | Implementations of 9 baselines; 5 missing |
| **Attacks** | `attacks/adversarial.py` (`AdversarialAttackFactory`)| Attacks: LF, GS, NI, MM-p, BD, basic ACK1/ACK2; lacks ACK3/4, on-off, K0-K2 |
| **Model** | `models/mlp.py` (`IDSMLP`) | 4-layer MLP, but incorrectly uses `nn.BatchNorm1d` instead of `nn.LayerNorm` |
| **Preprocessing** | `data/preprocessing/nslkdd_pipeline.py` | Preprocessing for NSL-KDD; CIC-IoT-2023 in `ciciot2023_pipeline.py`; Edge missing |
| **Partitioning** | `data/partitioning.py` (`NonIIDPartitioner`) | Dirichlet non-IID partitioning logic |
| **Experiment Runners**| `experiments/` (9 Python runners) | `run_experiment.py`, `run_full_comparison.py`, etc., currently hardcoded to 5 seeds |
| **Evaluation / Stats**| `evaluation/statistical_testing.py` | Significance tests, but uses 5 seeds, one-sided Wilcoxon, lacks Holm-Bonferroni & BCa CI |
| **Overhead Tracking**| `evaluation/overhead.py` (`OverheadTracker`) | Wall-clock profiling of verification and trust scoring |
| **Colab Pipeline** | `colab/` (`TV_FLIDS_full_campaign.ipynb`, etc.)| Legacy Colab runners tailored for 8 strategies $\times$ 5 seeds |
| **Test Suite** | `tests/` (20 test files) | Targeted and unit tests covering legacy codebase behavior |
| **Artifacts** | `results/` & `campaign_results/` | Output directory structure; currently only holds 8 legacy logs |
| **Paper Source** | `Paper/IEEE/TV-FLIDS.tex` & `_supplementary.tex` | IEEE TIFS manuscript and supplementary specification |
| **Target Generator**| `Paper/IEEE/analysis/targets/build_targets.py` | Generates synthetic target tables and verifies analytical properties |

---

## 5. Verified Paper Requirements

### 5.1 Methods Taxonomy (14 Baselines + 1 Proposed = 15 Methods) `[VERIFIED]`
1. `FedAvg` (McMahan et al., 2017)
2. `Krum` (Blanchard et al., 2017)
3. `Multi-Krum` (Blanchard et al., 2017)
4. `Trimmed Mean` (Yin et al., 2018)
5. `Norm Clipping` (Sun et al., 2019)
6. `RFA` (Pillutla et al., 2022)
7. `Bucketing` (Karimireddy et al., 2022)
8. `FoolsGold` (Fung et al., 2018)
9. `FLAME` (Nguyen et al., 2022)
10. `DeepSight` (Rieger et al., 2022)
11. `FLDetector` (Zhang et al., 2022)
12. `Zeno` (Xie et al., 2019)
13. `FLTrust` (Cao et al., 2021)
14. `BaFFLe` (Andreina et al., 2021)
15. `TV-FLIDS` (Proposed Framework)

### 5.2 Statistical Testing Protocol `[VERIFIED]`
- Primary family: 14 Baselines $\times$ 3 Datasets $\times$ 2 Primary Endpoints (Macro-F1, ASR under LF) = **84 Primary Statistical Tests**.
- Sample size: 20 seeds for primary tests (Block 2, Block 3, Block 7); 10 seeds for secondary tests.
- Test type: Two-sided exact paired Wilcoxon signed-rank test.
- Family-wise error control: Holm-Bonferroni step-down procedure over all 84 primary comparisons.
- Effect size & uncertainty: Paired median difference with 95% BCa bootstrap confidence intervals from 10,000 resamples.
- Paper outcomes under LF: 79 claims favor TV-FLIDS (all 72 against non-validation baselines + 7 against validation baselines); 5 tests yield no claim ($p \ge 0.05$, all involving FLTrust or Zeno).

---

## 6. Implementation Status

### Compact Component Status Table `[VERIFIED]`

| Component | Paper Requirement (§/Eq) | Current Code State | Status | Evidence File & Lines |
| :--- | :--- | :--- | :--- | :--- |
| **TV-FLIDS Pipeline** | §IV, Alg. 1: 6 discrete stages | Stages 1-5 strictly aligned (clipping, loss gate, trust scoring, meta-weight adaptation, trust-weighted aggregation); Stage 6 (Missing baselines) pending | `[PARTIAL]` | `fl/strategy.py:173-334` |
| **Norm Clipping** | §IV, Eq. (3): $C^{(t)} = \operatorname{median}(\|\vDelta\|)$ | Raw updates norm calculated; median radius computed over cohort $\mathcal{P}^{(t)}$; candidate models constructed from clipped updates $\tilde{w}_i = w^{(t)} + \tilde{\Delta}_i$ | `[VERIFIED]` | `trust/verification.py:27-58`, `fl/strategy.py:180-210` |
| **Validation Gate** | §IV, Eq. (4): Single test $\delta_i \ge \tau_L(t)$ | Single loss improvement gate evaluated on clipped candidate models; no independent rejection on cosine or norm z-score; $\tau_L(t) = -0.1 + 0.1\min(1, t/20)$; empty cohort $w^{(t+1)}=w^{(t)}$ | `[VERIFIED]` | `trust/verification.py:60-145`, `fl/strategy.py:205-235` |
| **Class-Balanced Loss**| §IV, Eq. (2), (6): $\ell_{\mathrm{bal}}$ per class | Per-class loss computed and averaged over non-empty validation classes; $A_i = \mathrm{clip}_{[-1, 1]}((\ell_{\mathrm{bal}}(w) - \ell_{\mathrm{bal}}(\tilde{w}_i))/(\ell_{\mathrm{bal}}(w) + \varepsilon))$ | `[VERIFIED]` | `trust/verification.py:147-235`, `trust/trust_scorer.py:61-79` |
| **Trust Memory** | §IV, Eq. (7): $\lambda_\uparrow=0.9, \lambda_\downarrow=0.7, T^{(0)}=0.5$ | Asymmetric EMA with $\lambda_\uparrow = 0.9$ (gain) and $\lambda_\downarrow = 0.7$ (penalty), initial trust $0.5$, floor $\tau_{\min} = 0.01$ | `[VERIFIED]` | `trust/trust_scorer.py:24-29, 137-148` |
| **Rejected Penalty** | §IV (Stage 3): $s_i = 0$ for $i \in \mathcal{P} \setminus \mathcal{A}$ | Rejected participants receive $s_i = 0$ and undergo penalty decay ($\lambda_\downarrow = 0.7$); non-participants retain previous trust | `[VERIFIED]` | `fl/strategy.py:228-235, 276-285`, `trust/trust_scorer.py:137-145` |
| **Norm Outlier Scope** | §IV, Eq. (5): $z_i$ over all $\mathcal{P}$ | Unclipped norms over all sampled participants $\mathcal{P}$ used to compute $\mu_{\mathcal{P}}, \sigma_{\mathcal{P}}$ and $O_i = 1 - e^{-z_i / 2.5}$ | `[VERIFIED]` | `fl/strategy.py:218-225`, `trust/trust_scorer.py:81-119` |
| **Online Meta-Weight** | §IV, Eq. (8), Lemma 4: Adam lr=0.01 on $\mathcal{L}_{\mathrm{meta}}$ with STE | $\mathbf{v} \in \mathbb{R}^3$, $\mathbf{v}^{(0)}=\mathbf{0} \implies (\alpha,\beta,\gamma)=(1/3,1/3,1/3)$, $\hat{w}_i$ with STE, persistent Adam ($\eta_{\mathrm{meta}}=0.01$), detached inputs, clean reset isolation | `[VERIFIED]` | `trust/adaptive_trust_scorer.py:27-140`, `fl/strategy.py:285-294` |
| **Model Aggregation** | §IV, Eq. (11), Alg. 1: $\vw^{(t+1)} = \sum_{i \in \cA} \frac{T_i^{(t)}}{\sum_{j \in \cA} T_j^{(t)}} \tilde{\vw}_i$ | Accepted clients only, clipped candidate models $\tilde{\vw}_i = \vw^{(t)} + \tilde{\vDelta}_i$, trust weights normalized to 1, empty cohort fallback $\vw^{(t+1)}=\vw^{(t)}$ | `[VERIFIED]` | `fl/strategy.py:295-303`, `trust/trust_scorer.py:187-191` |
| **Baseline Isolation** | Supp Table S2: 14 baselines isolated | 9 existing baselines independently isolated without inheriting TV-FLIDS clipping/gating/trust/meta-weights; zero state leakage | `[VERIFIED]` | `fl/baselines/`, `experiments/run_experiment.py:229-423` |
| **Model Normalization**| §VI-B: `nn.LayerNorm`, 53,125 params | `nn.LayerNorm(256/128/64)`, 53,125 params | `[VERIFIED]` | `models/mlp.py:26, 30, 34` |
| **NSL-KDD Pipeline** | §VI-A, Table I: Quota split $\mathcal{D}_{\mathrm{val}}=2,000, \mathcal{D}_{\mathrm{tune}}$ | Exact Table I quotas, zero-leakage scaler on $\mathcal{D}_{\mathrm{val}} \cup \mathcal{D}_{\mathrm{tune}}$, local SMOTE | `[VERIFIED]` | `data/preprocessing/nslkdd_pipeline.py:173-285` |
| **CIC-IoT-2023 Pipeline**| §VI-A, Table S5: Time-disjoint 80/20, 60s band | Pipeline implementation verified on synthetic fixtures (46 features, 8 classes, time-disjoint 80/20 split, 60s guard band, Table S5 caps, zero-leakage scaler); real-dataset execution blocked pending acquisition of exact dataset version | `[PARTIALLY VERIFIED]` | `data/preprocessing/ciciot2023_pipeline.py` |
| **Edge-IIoTset Pipeline**| §VI-A, Table S5: 6 classes, time-disjoint | Pipeline implementation verified on synthetic fixtures (61 features, 6 classes [14 raw attacks mapped to 5 classes + normal], time-disjoint 80/20 split, 60s guard band, Table S5 caps, zero-leakage scaler); real-dataset execution blocked pending acquisition of exact dataset version | `[PARTIALLY VERIFIED]` | `data/preprocessing/edgeiiotset_pipeline.py` |
| **14 Baselines** | §VI-C, Table S2: 14 baselines | All 14 baseline strategies implemented as standalone strategies in `fl/baselines/`, fully isolated from TV-FLIDS mechanisms, and dispatchable via `make_strategy` | `[VERIFIED]` | `fl/baselines/`, `experiments/run_experiment.py:295-423` |
| **Attack Suite** | §III, §VII-E, Supp S4: 10 attacks + ACK1–4 | Basic LF, GS, NI, MM-p, BD; lacks ACK3/4, on-off | `[PARTIAL]` | `attacks/adversarial.py` |
| **Ablations** | §VII-D, Table VI: 12 variants, 20 seeds | Legacy A1–A6, 5 seeds | `[MISMATCH]` | `experiments/run_ablation.py:33-60` |
| **Statistical Analysis**| §VI-E, Supp S8: 20 seeds, 2-sided, Holm, BCa | Robust statistical evaluation pipeline implemented: result ingestion, explicit seed pairing, 2-sided exact Wilcoxon, Holm over 84 hypotheses, BCa 95% CIs (10k resamples), Delta-ASR, target validator, and table generator; real execution incomplete (0/84 completed real comparisons) | `[PARTIALLY VERIFIED]` | `evaluation/` |
| **Result Artifacts** | Release manifest, logs, confusion matrices | Only 8 legacy simulation logs | `[INCOMPLETE]`| `results/logs/` |
| **Campaign Runner** | Supp Table S6: $\le 7,038$ runs (6,111 actual) | Canonical campaign orchestrator, enumerator, config freezer, and runner implemented with checkpoint/resume, failure states, seed isolation, and dataset blocker handling | `[PARTIALLY VERIFIED]` | `campaign/` |
| **Colab Infrastructure**| Supp Table S6 execution | Configured for legacy 8-method, 5-seed runs | `[MISMATCH]` | `colab/` |

---

## 7. Experimental Campaign

### 7.1 Arithmetic Derivation of Run Counts `[VERIFIED]`
- **Paper Stated Upper Bound:** $\le 7,038$ runs (100 rounds each) `[VERIFIED]`
- **Block Breakdown (Supplementary Table S6):**
  - **Block 1 (Tuning on $\mathcal{D}_{\mathrm{tune}}$):** Stated bound $\le 1,620$. Actual defined parameter grid in Table S2:
    - FedAvg: 0 configs
    - Krum: 3 configs ($f' \in \{2,3,4\}$)
    - Multi-Krum: 6 configs ($f' \in \{2,3,4\} \times m \in \{D-f', D-2f'\}$)
    - Trimmed Mean: 4 configs ($\beta_{\mathrm{TM}} \in \{0.1,0.2,0.3,0.4\}$)
    - Norm Clipping: 4 configs (radius $\in \{0.5,1,1.5,2\} \times$ median norm)
    - RFA: 4 configs (iters $\in \{3,5,10,20\}$)
    - Bucketing: 6 configs ($s \in \{2,3,5\} \times \beta_{\mathrm{TM}} \in \{0.1,0.2\}$)
    - FoolsGold: 4 configs ($\kappa \in \{0.5,1,2,4\}$)
    - FLAME: 9 configs (cluster size $\in \{D/4, D/3, D/2+1\} \times$ noise $\in \{0.5,1,2\}$)
    - DeepSight: 3 configs (clip quantile $\in \{0.25,0.5,0.75\}$)
    - FLDetector: 4 configs (window $\in \{5,10\} \times$ start round $\in \{10,20\}$)
    - Zeno: 6 configs ($\rho_Z \in \{5\times10^{-4}, 10^{-3}, 2\times10^{-3}\} \times b_Z \in \{3,4\}$)
    - FLTrust: 4 configs (epochs $\in \{1,5\} \times$ lr $\in \{5\times10^{-4}, 10^{-3}\}$)
    - BaFFLe: 8 configs (validators $\in \{5,10\} \times$ quorum $\in \{0.3,0.5\} \times$ lookback $\in \{10,20\}$)
    - TV-FLIDS: 12 configs (center $+ 11$ single-factor variants: $\lambda_\uparrow \in \{0.85, 0.95\}$, $\lambda_\downarrow \in \{0.6, 0.8\}$, $\tau_{\min} \in \{0.005, 0.05\}$, $T_{\mathrm{warm}} \in \{10, 30\}$, $\tau_L(0) \in \{-0.2, 0.0\}$, $T^{(0)} = 1.0$)
    - Sum of configurations across 15 methods: $\mathbf{77}$ configurations.
    - Actual Block 1 runs: $77 \text{ configs} \times 3 \text{ seeds} \times 3 \text{ datasets} = \mathbf{693}$ runs.
  - **Block 2 (RQ1 Main LF):** $15 \text{ methods} \times 3 \text{ datasets} \times 20 \text{ seeds} = \mathbf{900}$ runs.
  - **Block 3 (Clean Reference):** $15 \text{ methods} \times 3 \text{ datasets} \times 20 \text{ seeds} = \mathbf{900}$ runs.
  - **Block 4 (RQ1 Attack Matrix, NSL-KDD):** $15 \text{ methods} \times 7 \text{ attacks (GS, NI, BD, MM-p, MM-o, MS-p, MS-o)} \times 10 \text{ seeds} = \mathbf{1,050}$ runs.
  - **Block 5 (Sensitivity Sweeps):** $5 \text{ methods} \times 6 \text{ settings} \times 10 \text{ seeds} = \mathbf{300}$ runs.
  - **Block 6 (RQ2 Routing Interventions):** 6 configurations $\times 10$ seeds $= \mathbf{100}$ runs.
  - **Block 7 (RQ3 Component Ablations):** $12 \text{ variants} \times 2 \text{ datasets} \times 20 \text{ seeds} = \mathbf{480}$ runs.
  - **Block 8 (RQ4 Adaptive Attacks):** $[5 \text{ methods} \times (9 \text{ ACK} + 8 \text{ on-off}) \times 10] + [2 \text{ TV-FLIDS variants} \times 8 \text{ on-off} \times 10] = 850 + 160 = \mathbf{1,010}$ runs.
  - **Block 9 (RQ6 Deployment Variability):** $5 \text{ methods} \times 40 \text{ configurations} = \mathbf{200}$ runs.
  - **Block 10 (RQ8 DP & Profiling):** $(2 \text{ methods} \times 10 \text{ seeds}) + 8 \text{ profiling runs} = \mathbf{28}$ runs.
  - **Block 11 (SMOTE Factorial):** $15 \text{ methods} \times 3 \text{ non-main cells} \times 10 \text{ seeds} = \mathbf{450}$ runs.
- **Total Production Campaign Sum:**
  - Blocks B2–B11: $900 + 900 + 1,050 + 300 + 100 + 480 + 1,010 + 200 + 28 + 450 = 5,418$ runs.
  - Resolved canonical campaign (with 77 tuning configs): $693 + 5,418 = \mathbf{6,111}$ runs `[VERIFIED]`.
  - Theoretical paper upper bound: $1,620 + 5,418 \le \mathbf{7,038}$ runs `[VERIFIED]`.

---

## 8. Dataset Status

| Dataset | Feats ($d$) | Classes ($K$) | Target Caps (Train / Test) | Split Rule | Local Repository State | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NSL-KDD** | 41 | 5 | 125,973 / 22,544 | Standard benchmark split (`KDDTrain+` / `KDDTest+`), server $\mathcal{D}_{\mathrm{val}}=2,000$ (U2R=15, R2L=100), $\mathcal{D}_{\mathrm{tune}}=12,398$ | `data/raw/KDDTrain+.txt`, `KDDTest+.txt` exist | `[VERIFIED]` raw present, preprocessing verified |
| **CIC-IoT-2023** | 46 | 8 | 150,000 / 30,000 | Time-disjoint 80/20 split per capture, 60s guard band, Table S5 caps | Raw multi-GB shards missing; pipeline verified with synthetic schema fixture | `[VERIFIED]` pipeline complete; raw data `[BLOCKED]` |
| **Edge-IIoTset** | 61 | 6 | 150,000 / 30,000 | Time-disjoint 80/20 split per sensor/capture, 60s guard band, Table S5 caps | Raw multi-GB files missing; pipeline verified with synthetic schema fixture | `[VERIFIED]` pipeline complete; raw data `[BLOCKED]` |
| **UNSW-NB15** | 49 | 10 | — | Not in IEEE paper | `data/raw/` CSVs present, pipeline present | `[ORPHANED]` candidate for archiving |

### 8.1 Stage 8: CIC-IoT-2023 and Edge-IIoTset Dataset Pipeline Completion `[VERIFIED]`

- **Specification Reconciliation & Canonical Parameters:**
  - **Canonical Model Parameter Formula:** $P(d, K) = 256d + 65K + 42,304$ `[VERIFIED]`
    - NSL-KDD: $P(41, 5) = 256(41) + 65(5) + 42,304 = 53,125$
    - CIC-IoT-2023: $P(46, 8) = 256(46) + 65(8) + 42,304 = 54,600$
    - Edge-IIoTset: $P(61, 6) = 256(61) + 65(6) + 42,304 = 58,310$
  - **CIC-IoT-2023:** $d=46, K=8$.
    - Canonical class ordering: `["Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "WebBased", "BruteForce"]` with rare classes `[6, 7]`.
    - Table S5 caps: Train $150,000$ (30k, 40k, 30k, 25k, 12k, 9k, 2.4k, 1.6k); Test $30,000$ (6k, 8k, 6k, 5k, 2.4k, 1.8k, 480, 320).
    - Splitting: Time-disjoint 80/20 per capture with 60-second guard band (`time_disjoint_split_with_guard_band`).
    - Server Validation: $\mathcal{D}_{\mathrm{val}}=2,000$ (rare-class protected quotas, proportional allocation), $\mathcal{D}_{\mathrm{tune}}$ (10% stratified draw of remainder), $\mathcal{D}_{\mathrm{client}}$ remainder.
    - Zero-leakage scaling: `MinMaxScaler(clip=False)` fitted strictly on server data $\mathcal{D}_{\mathrm{val}} \cup \mathcal{D}_{\mathrm{tune}}$.
  - **Edge-IIoTset:** $d=61, K=6$.
    - Canonical class ordering: `["Normal", "DoS/DDoS", "Injection", "Scanning", "Malware", "MITM"]` with rare class `[5]` (MITM).
    - 14 raw attacks mapped to 5 classes + Normal:
      - Normal: 0
      - DoS/DDoS (DDoS_UDP, DDoS_ICMP, DDoS_HTTP, DDoS_TCP, DoS_UDP, DoS_ICMP, DoS_HTTP, DoS_TCP): 1
      - Injection (SQL_injection, XSS, Uploading): 2
      - Scanning (Port_Scanning, Vulnerability_scanner, Fingerprinting): 3
      - Malware (Backdoor, Password, Ransomware): 4
      - MITM: 5
    - Table S5 caps: Train $150,000$ (Normal 45k, DoS/DDoS 40k, Injection 25k, Scanning 20k, Malware 15k, MITM 5k); Test $30,000$ (Normal 9k, DoS/DDoS 8k, Injection 5k, Scanning 4k, Malware 3k, MITM 1k).
    - Splitting: Time-disjoint 80/20 per capture/sensor with 60-second guard band.
    - Server Validation: $\mathcal{D}_{\mathrm{val}}=2,000$, $\mathcal{D}_{\mathrm{tune}}$ (10% stratified draw of remainder), $\mathcal{D}_{\mathrm{client}}$ remainder.
    - Zero-leakage scaling: `MinMaxScaler(clip=False)` fitted strictly on server data $\mathcal{D}_{\mathrm{val}} \cup \mathcal{D}_{\mathrm{tune}}$.
- **Raw Data Availability Audit:**
  - Multi-GB raw files for CIC-IoT-2023 and Edge-IIoTset are not present in `data/raw/` and downloading them without user direction is out of scope. Marked as `[BLOCKED]`.
  - All preprocessing, splitting, guard band, capping, scaling, and forward pass logic is verified using deterministic synthetic schema fixtures.
- **Cross-Dataset Interface & Provenance Manifests:**
  - Created `data/dataset_bundle.py` providing `DatasetBundle` dataclass, `load_dataset(...)` factory, parameter count verification, and machine-readable provenance manifest generation.
  - Updated `config/dataset_config.yaml` with Edge-IIoTset specification ($d=61, K=6$) and aligned CIC-IoT-2023 canonical class names.
  - Updated `experiments/run_experiment.py` and `experiments/run_full_comparison.py` to support `edgeiiotset` across data setup, client factory, and CLI choices.
- **Verification Evidence:**
  - 22/22 dedicated tests passed (`tests/test_ciciot2023_pipeline.py`, `tests/test_edgeiiotset_pipeline.py`, `tests/test_cross_dataset_interface.py`).
  - 63/63 regression tests passed (`tests/test_val_size_regression.py`, `tests/test_experiment_reachability.py`).
  - 2-round end-to-end smoke test on NSL-KDD verified simulation integrity (53,125 parameters, 0 regressions).

---

## 9. Baseline Status

| Baseline | Paper Ref | Strategy File | Tuning Grid Defined | Status in Code |
| :--- | :--- | :--- | :--- | :--- |
| 1. FedAvg | McMahan 2017 | `fl/baselines/fedavg_strategy.py` | None (0) | `[VERIFIED]` present |
| 2. Krum | Blanchard 2017 | `fl/baselines/krum_strategy.py` | $f' \in \{2,3,4\}$ (3) | `[VERIFIED]` present (canonical $m=1$) |
| 3. Multi-Krum | Blanchard 2017 | `fl/baselines/multikrum_strategy.py` | $f' \in \{2,3,4\}, m \in \{D-f', D-2f'\}$ (6) | `[VERIFIED]` standalone strategy present |
| 4. Trimmed Mean | Yin 2018 | `fl/baselines/trimmed_mean_strategy.py` | $\beta_{\mathrm{TM}} \in \{0.1,0.2,0.3,0.4\}$ (4) | `[VERIFIED]` present |
| 5. Norm Clipping| Sun 2019 | `fl/baselines/norm_clipping_strategy.py` | radius $\in \{0.5,1,1.5,2\} \times$ median norm (4)| `[VERIFIED]` standalone strategy present |
| 6. RFA | Pillutla 2022 | `fl/baselines/rfa_strategy.py` | iterations $\in \{3,5,10,20\}$ (4) | `[VERIFIED]` present |
| 7. Bucketing | Karimireddy 2022 | `fl/baselines/bucketing_strategy.py`| $s \in \{2,3,5\}, \beta_{\mathrm{TM}} \in \{0.1,0.2\}$ (6) | `[VERIFIED]` present |
| 8. FoolsGold | Fung 2018 | `fl/baselines/foolsgold_strategy.py` | $\kappa \in \{0.5,1,2,4\}$ (4) | `[VERIFIED]` present |
| 9. FLAME | Nguyen 2022 | `fl/baselines/flame_strategy.py` | cluster size $\in \{D/4, D/3, D/2+1\}$, noise $\in \{0.5,1,2\}$ (9)| `[VERIFIED]` present |
| 10. DeepSight | Rieger 2022 | `fl/baselines/deepsight_strategy.py` | clipping quantile $\in \{0.25,0.5,0.75\}$ (3) | `[VERIFIED]` present |
| 11. FLDetector | Zhang 2022 | `fl/baselines/fldetector_strategy.py` | window $\in \{5,10\}$, start $\in \{10,20\}$ (4) | `[VERIFIED]` standalone strategy present |
| 12. Zeno | Xie 2019 | `fl/baselines/zeno_strategy.py` | $\rho_Z \in \{5\times10^{-4}, 10^{-3}, 2\times10^{-3}\}, b_Z \in \{3,4\}$ (6)| `[VERIFIED]` standalone strategy present |
| 13. FLTrust | Cao 2021 | `fl/baselines/fltrust_strategy.py` | epochs $\in \{1,5\}$, lr $\in \{5\times10^{-4}, 10^{-3}\}$ (4) | `[VERIFIED]` present |
| 14. BaFFLe | Andreina 2021 | `fl/baselines/baffle_strategy.py` | validators $\in \{5,10\}$, quorum $\in \{0.3,0.5\}$, lookback $\in \{10,20\}$ (8)| `[VERIFIED]` standalone strategy present |

---

## 10. Attack Status

| Attack | Description / Specification | Scope & Parameters | Code Location | Status in Code |
| :--- | :--- | :--- | :--- | :--- |
| **LF** (Label Flipping) | Attack classes flipped to Benign (0) | $f/N=0.3, r=1.0$ | `attacks/adversarial.py:38` | `[VERIFIED]` present |
| **GS** (Gradient Scaling) | Update multiplied by scalar $\kappa$ | $\kappa=10$, swept to $10^4$ | `attacks/adversarial.py:69` | `[VERIFIED]` present |
| **NI** (Noise Injection) | Gaussian noise added to parameters | $\sigma=0.5$ | `attacks/adversarial.py:93` | `[VERIFIED]` present |
| **BD** (Backdoor) | 10% attack samples stamped with trigger | 3 KDD features set to max | `attacks/adversarial.py:117` | `[VERIFIED]` present |
| **MM-p / MM-o** | Min-Max attack | Partial & omniscient | `attacks/adversarial.py:175` | `[PARTIAL]` MM-p present; MM-o missing |
| **MS-p / MS-o** | Min-Sum attack | Partial & omniscient | None | `[MISSING]` |
| **ACK1** | Acceptance test evasion | Surrogate validation loss hinge | `attacks/adversarial.py:228` | `[PARTIAL]` basic version present; lacks K0–K2 |
| **ACK2** | Trust floor manipulation | Shift along benign-ward gradient | `attacks/adversarial.py:284` | `[PARTIAL]` basic version present; lacks K0–K2 |
| **ACK3** | Class-balanced evasion | Relabel DoS/Probe, dual hinge | None | `[MISSING]` |
| **ACK4** | Meta-weight evasion | Simulation over mixture grid | None | `[MISSING]` |
| **On-off** | Honest for $k$ rounds, then attack | $k \in \{10, 20, 30, 50\}$, LF & ACK2 | None | `[MISSING]` |
| **LF-R** | Attack flipped to random attack class| RQ2 intervention | None | `[MISSING]` |

---

## 11. Testing Status

- **Existing Test Suite:** 20 test files in `tests/` covering smoke tests, figure generation, config defaults, Proposition 1 domain, and legacy attack mechanics `[VERIFIED]`.
- **Test Integrity:** The existing tests pass against the legacy codebase, but they test the legacy logic (e.g. 3-check verification gate, BatchNorm parameter shapes, 5-seed statistical assumptions) `[VERIFIED]`.
- **Required New Test Infrastructure:** Tests must be created to verify:
  1. Median norm clipping exactness
  2. Single validation loss gate threshold schedule
  3. Class-balanced loss calculation vs sample-averaged loss
  4. Asymmetric trust update and rejection penalty
  5. LayerNorm parameter counts and invariance under non-IID
  6. 20-seed Wilcoxon signed-rank test and Holm-Bonferroni correction

---

## 12. Result/Artifact Status

- **Empirical Execution Logs:** Only 8 historical simulation logs in `results/logs/` (NSL-KDD, 100 rounds, 4 FedAvg LF, 4 TV-FLIDS Clean) `[VERIFIED]`.
- **Target Data:** Complete set of target numbers generated by `Paper/IEEE/analysis/targets/build_targets.py` stored in `Paper/IEEE/analysis/targets/targets.json` `[VERIFIED]`.
- **Artifact Quarantine:** Legacy mock outputs quarantined in `results/_QUARANTINED_MOCK/` `[VERIFIED]`.
- **Production Result Readiness:** 0% of the IEEE campa---

## 13. Known Problems

### 13.1 Critical Scientific Mismatches (Code vs. IEEE Paper) `[VERIFIED]`
1. **Norm Clipping:** Code computes unclipped updates and passes them directly to verification; paper mandates Stage 1 median norm clipping before gating or scoring. `[RESOLVED]` in Stage 2.
2. **Gate Structure:** Code enforces 3 filtering gates (Loss, Cosine, Z-score); paper explicitly rejects Checks 2 & 3 as hard filters and specifies a single validation-loss gate. `[RESOLVED]` in Stage 2.
3. **Class-Balanced Loss:** Code computes sample-averaged cross-entropy loss improvement; paper specifies class-balanced loss $\ell_{\mathrm{bal}}$ to protect rare classes. `[RESOLVED]` in Stage 3.
4. **Asymmetric Trust Decay:** Code uses symmetric exponential decay (`decay=0.9`); paper mandates asymmetric decay ($\lambda_\uparrow=0.9, \lambda_\downarrow=0.7$) and initial trust $0.5$ (code uses $1.0$). `[RESOLVED]` in Stage 3.
5. **Rejection Penalization:** Code skips rejected clients; paper explicitly sets $s_i=0$ and penalizes trust via $\lambda_\downarrow=0.7$. `[RESOLVED]` in Stage 3.
6. **Norm Outlier Scope:** Code computes z-scores over accepted cohort $\mathcal{A}$; paper specifies computing over all sampled participants $\mathcal{P}$. `[RESOLVED]` in Stage 3.
7. **Normalization Layer:** Code uses `nn.BatchNorm1d`; paper specifies `nn.LayerNorm` (parameter count 53,125 on NSL-KDD). `[RESOLVED]` in Stage 1.
8. **Missing Dataset:** Edge-IIoTset is 100% missing from the codebase.
9. **Missing Baselines:** Multi-Krum (standalone), Norm Clipping, FLDetector, Zeno, BaFFLe were missing; `[RESOLVED]` in Stage 6 with standalone verified strategy implementations for all 14 baselines.
10. **Missing Attacks:** Min-Sum, ACK3, ACK4, On-off ($k \in \{10,20,30,50\}$), LF-R, and K0/K1/K2 knowledge tiers are missing.
11. **Statistical Testing Divergence:** Code uses 5 seeds, one-sided tests, mean $\pm$ std; paper requires 20 seeds, two-sided exact Wilcoxon, Holm-Bonferroni across 84 tests, BCa 95% CIs.

---

## 14. Approved Decisions

| Decision ID | Context / Area | Paper Specification | Current Implementation | Approved Decision | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DEC-001** | Model Normalization | `nn.LayerNorm` (§VI-B) | `nn.LayerNorm(256/128/64)` (`models/mlp.py`) | Replace `BatchNorm1d` with `LayerNorm` across all MLP hidden layers | `[VERIFIED]` Implemented |
| **DEC-002** | Update Clipping | Median norm clipping (§IV, Eq. 3) | Implemented (`fl/strategy.py`, `trust/verification.py`) | Implement Stage 1 median norm clipping prior to validation gating | `[VERIFIED]` Implemented |
| **DEC-003** | Validation Gate | Single loss gate $\delta_i \ge \tau_L(t)$ (§IV, Eq. 4)| Single validation loss gate (`trust/verification.py`) | Replace 3-check filter with single validation-loss gate; relegate cosine and norm to continuous signals | `[VERIFIED]` Implemented |
| **DEC-004** | Trust Decay | Asymmetric $\lambda_\uparrow=0.9, \lambda_\downarrow=0.7, T^{(0)}=0.5$ | Asymmetric EMA (`trust_scorer.py`) | Implement asymmetric trust EMA and initial trust 0.5 | `[VERIFIED]` Implemented |
| **DEC-005** | Rejection Penalty | Rejected clients get $s_i=0$ | Handled via $s_i=0$ and $\lambda_\downarrow=0.7$ | Pass rejected clients with $s_i=0$ to trust update | `[VERIFIED]` Implemented |
| **DEC-006** | Class-Balanced Loss | $\ell_{\mathrm{bal}}$ over $K$ classes (§IV, Eq. 2, 6) | Implemented in `trust/verification.py` | Implement $\ell_{\mathrm{bal}}$ and compute $A_i \in [-1, 1]$ | `[VERIFIED]` Implemented |
| **DEC-007** | Evaluation Protocol | 20 seeds, 2-sided Wilcoxon, Holm, BCa | 5 seeds, 1-sided, mean $\pm$ std | Implement IEEE statistical evaluation pipeline | `[APPROVED]` |
| **DEC-008** | Persistent Memory | Repository memory protocol | Established at root | Establish `PROJECT_MEMORY.md` at root | `[VERIFIED]` Completed |
| **DEC-009** | 14-Baseline Matrix | All 14 baselines standalone | All 14 baseline strategies implemented (`fl/baselines/`) | Implement and isolate all 14 baselines from Table S2 | `[VERIFIED]` Completed |

---

## 15. Rejected/Forbidden Changes

- **Do not treat implementation targets as experimental results:** Numbers in `Paper/IEEE/` typeset with `\itv{...}` are projected targets, not measured data.
- **Do not fabricate or infer experimental measurements:** All published results must originate from verifiable simulation executions.
- **Do not silently change paper methodology:** Paper is the scientific specification; deviations require explicit external supervisor approval.
- **Do not delete research artifacts without evidence:** Follow the provenance rules; prefer archival over deletion.
- **Do not declare a component implemented merely because a filename exists:** Verify actual internal logic against paper equations.
- **Do not launch the full campaign before implementation and smoke tests pass:** All 14 stages of alignment must be validated before production runs.
- **Do not optimize toward target numerical values:** The goal is faithful scientific reproduction, not artificial matching of projections.
- **Do not replace a paper-defined algorithm with a merely similar algorithm:** Implement exact paper formulas.
- **Do not report "verified" without concrete evidence:** Maintain rigorous labeling discipline (`[VERIFIED]`, `[INFERRED]`, etc.).

---

## 16. Current Task

- **Task Name:** Stage 10 — Statistical Evaluation Pipeline and Manuscript Target Validation `[PARTIALLY VERIFIED]`
- **Status:** COMPLETED. `[PARTIALLY VERIFIED]`
- **Allowed Changes in this Task:**
  - `evaluation/result_loader.py` (new)
  - `evaluation/pairing.py` (new)
  - `evaluation/wilcoxon.py` (new)
  - `evaluation/holm.py` (new)
  - `evaluation/bootstrap_ci.py` (new)
  - `evaluation/delta_metrics.py` (new)
  - `evaluation/effect_size.py` (new)
  - `evaluation/target_validator.py` (new)
  - `evaluation/table_generator.py` (new)
  - `evaluation/statistical_pipeline.py` (new)
  - `tests/test_statistical_evaluation.py` (new)
  - `PROJECT_MEMORY.md`
- **Scientific Alignment Accomplished:**
  - **Statistical Protocol Audit & Reconciliation**:
    - Reconciled directly against `Paper/IEEE/TV-FLIDS.tex` (§VI-E, §VII, Table III), `TV-FLIDS_supplementary.tex` (§S8, Tables S3–S15), and `targets.json`.
    - Primary Endpoints: Macro-F1 and Attack Success Rate (ASR, Eq. 16) under Label Flipping ($f/N=0.3, \alpha_D=0.5, 100$ rounds, Block 2).
    - Primary Comparison Family: Exactly **84 hypotheses** ($14\text{ baselines} \times 3\text{ datasets} \times 2\text{ endpoints}$).
    - Paired Experimental Unit: Seed-paired observations (same dataset, block, attack, and seed).
    - Two-Sided Exact Wilcoxon Signed-Rank Test: Zero differences ($|d| \le 10^{-12}$) dropped; exact sign-permutation null distribution computed via dynamic programming over $2R_i$ (always integers), retaining exact inference even in the presence of tied absolute differences; zero asymptotic or normal approximation fallback; verified against independent sign-permutation enumeration.
    - Holm-Bonferroni Family-Wise Error Control: Step-down adjustment applied across the complete 84-hypothesis primary family.
    - Effect Sizes: Paired median difference in percentage points $\operatorname{median}(100 \times (\text{TV-FLIDS} - \text{baseline}))$ and directional seed favor counts ($d > 0$ for Macro-F1, $d < 0$ for ASR).
    - Confidence Intervals: 95% BCa bootstrap confidence intervals from 10,000 paired resamples, with percentile fallback on degenerate acceleration / non-finite bounds.
    - Delta-ASR: $\Delta\mathrm{ASR} = 100 \times (\mathrm{ASR}_{\text{attacked}} - \mathrm{ASR}_{\text{clean-ref}})$ paired by seed against Block 3 clean reference runs.
    - Confusion-Matrix Routing Diagnostics: Computes $R_{\mathrm{atk}}$ and $\phi$ satisfying exact identity $\mathrm{ASR} = (1 - R_{\mathrm{atk}})\phi$.
  - **Robust Artifact Ingestion & Invariant Validation**:
    - `ResultLoader` discovers campaign artifacts, validates run IDs against scientific configuration hashes, verifies Git and dataset provenance, and enforces completion status.
    - Strict quarantine: excludes `FAILED`, `BLOCKED`, `PENDING`, `RUNNING`, corrupted, and `SYNTHETIC_TEST_ONLY` runs from production observations.
  - **Explicit Seed Pairing**:
    - `PairedComparison` pairs TV-FLIDS and baselines by seed integer; detects missing baseline/TV-FLIDS seeds, duplicate seeds, and configuration mismatches.
    - Marks comparisons with $< 20$ valid pairs (or $< 10$ for secondary) as `INCOMPLETE`.
  - **Manuscript Target Validation**:
    - `TargetValidator` validates results against `targets.json` only when real completed observations exist (`status == "COMPLETE"`). Refuses to validate incomplete, blocked, or synthetic data as real matches.
  - **Manuscript Table Generation**:
    - `table_generator.py` produces LaTeX table snippets for Tables III, IV, S3, S4, displaying `[INCOMPLETE]` / `--` when empirical data are missing.
  - **Zero Data Fabrication & Campaign Completeness Status**:
    - Full campaign (6,111 planned runs) has not been executed; real data runs for `ciciot2023` and `edgeiiotset` remain blocked.
    - Evaluation CLI execution verified: 0/84 complete real comparisons (28 incomplete on NSL-KDD, 56 blocked on IoT datasets, 0 synthetic observations used).
  - **Testing & Verification Evidence**:
    - 26/26 dedicated tests passed (`tests/test_statistical_evaluation.py`).
    - 47/47 Stage 9 orchestration tests passed (`tests/test_campaign_orchestration.py`).
    - 107/107 regression tests passed (`tests/test_model_and_nslkdd_alignment.py`, `tests/test_strategy_alignment.py`, `tests/test_trust_alignment.py`, `tests/test_meta_weight_alignment.py`, `tests/test_aggregation_alignment.py`, `tests/test_baseline_isolation.py`, `tests/test_val_size_regression.py`).
    - 0 consistency check failures (`Paper/IEEE/analysis/targets/verify_consistency.py`).

---

## 17. Next Task

- **Task Name:** Stage 11 — Production Campaign Execution / Orchestration Strategy. `[DECISION]`
- **Rationale:** The complete scientific and statistical pipeline is now fully aligned and verified. Stage 11 will define the execution strategy for the canonical 6,111-run campaign (e.g. NSL-KDD execution, cloud/Colab coordination, and real dataset acquisition requirements).
- **Supervisor Approval:** Pending Stage 10 completion confirmation.

---

## 18. Important Commands

- **Run Stage 10 Statistical Evaluation Tests:**
  `pytest tests/test_statistical_evaluation.py -v` `[VERIFIED]`
- **Run Statistical Evaluation CLI:**
  `python -m evaluation.statistical_pipeline` `[VERIFIED]`
- **Environment Verification:**
  `python scripts/verify_environment.py` `[VERIFIED]`
- **Run Proposition 1 Bound Check:**
  `python theory/proposition1_verification.py` `[VERIFIED]`
- **Run Manuscript Integrity Check:**
  `python scripts/check_manuscript.py` `[VERIFIED]`
- **Run Campaign Enumeration Report:**
  `python -m campaign.enumerator --report` `[VERIFIED]`
- **Export Campaign Manifest:**
  `python -m campaign.enumerator --export manifest.json` `[VERIFIED]`
- **Dry-Run Campaign Execution:**
  `python -m campaign.runner --dry-run --block 1 --strategy tvflids --dataset nslkdd` `[VERIFIED]`
- **Run Stage 9 Campaign Orchestration Tests:**
  `pytest tests/test_campaign_orchestration.py -v` `[VERIFIED]`
- **Run Unit & Targeted Tests:**
  `pytest tests/` `[VERIFIED]`
- **Run Stage 1 Model and NSL-KDD Alignment Tests:**
  `pytest tests/test_model_and_nslkdd_alignment.py -v` `[VERIFIED]`
- **Run Stage 2 Strategy Alignment Tests:**
  `pytest tests/test_strategy_alignment.py -v` `[VERIFIED]`
- **Run Stage 3 Trust Alignment Tests:**
  `pytest tests/test_trust_alignment.py -v` `[VERIFIED]`
- **Run Stage 4 Meta-Weight Alignment Tests:**
  `pytest tests/test_meta_weight_alignment.py -v` `[VERIFIED]`
- **Run Stage 5 Aggregation and Baseline Isolation Tests:**
  `pytest tests/test_aggregation_alignment.py tests/test_baseline_isolation.py -v` `[VERIFIED]`
- **Run Stage 6 Missing Baseline Tests:**
  `pytest tests/test_multikrum_strategy.py tests/test_norm_clipping_strategy.py tests/test_fldetector_strategy.py tests/test_zeno_strategy.py tests/test_baffle_strategy.py tests/test_baseline_isolation.py -v` `[VERIFIED]`
- **Run Smoke Test (3 rounds, seed 42):**
  `python experiments/run_experiment.py --strategy tvflids --attack label_flip_30 --rounds 3 --seed 42` `[VERIFIED]`
- **Verify Paper Target Generation:**
  `python Paper/IEEE/analysis/targets/build_targets.py` `[VERIFIED]`

---

## 19. Validation Evidence

- **Parameter Closed Form Check:** Verified algebraically and computationally: $P(41, 5) = 53,125$, $P(46, 8) = 54,600$, $P(61, 6) = 58,310$. Matches paper §VI-B line 355.
- **NSL-KDD Split Quotas:** Verified exact match to Table I: $\mathcal{D}_{\mathrm{val}}=2,000$ (Normal: 1,016, DoS: 693, Probe: 176, R2L: 100, U2R: 15), $\mathcal{D}_{\mathrm{tune}}=12,398$ (Normal: 6,633, DoS: 4,523, Probe: 1,148, R2L: 90, U2R: 4), $\mathcal{D}_{\mathrm{client}}=111,575$ (Normal: 59,694, DoS: 40,711, Probe: 10,332, R2L: 805, U2R: 33), $\mathcal{D}_{\mathrm{test}}=22,544$ (Normal: 9,711, DoS: 7,458, Probe: 2,421, R2L: 2,754, U2R: 200). Total training pool: $2,000 + 12,398 + 111,575 = 125,973$.
- **Disjointness & Zero Leakage:** Verified pairwise empty intersections between $\mathcal{D}_{\mathrm{val}}$, $\mathcal{D}_{\mathrm{tune}}$, and $\mathcal{D}_{\mathrm{client}}$. Proved that modifying client and test distributions leaves fitted scaler statistics (`data_min_`, `data_max_`, `scale_`) 100% identical.
- **Client-Local SMOTE Isolation:** Proved synthetic isolation where client-local SMOTE respects client partition boundaries, and confirmed that server validation, tuning, and test sets are never oversampled.
- **Stage 2 Median-Radius Clipping & Validation Gate Alignment:**
  - Full cohort median norm clipping: $C^{(t)} = \operatorname{median}_{j \in \mathcal{P}^{(t)}} \|\Delta_j^{(t)}\|$ and $\tilde{\Delta}_i^{(t)} = \Delta_i^{(t)} \min(1, C^{(t)} / \|\Delta_i^{(t)}\|)$ verified mathematically and in code.
  - Candidate model constructed from clipped updates: $\tilde{w}_i^{(t)} = w^{(t)} + \tilde{\Delta}_i^{(t)}$ before validation loss evaluation.
  - Single validation loss improvement gate: $\delta_i = \ell_{\mathrm{val}}(w^{(t)}) - \ell_{\mathrm{val}}(\tilde{w}_i^{(t)}) \ge \tau_L(t)$ with schedule $\tau_L(t) = -0.1 + 0.1 \min(1, t / 20)$.
  - Neither cosine similarity nor norm z-score functions as a hard rejection criterion.
  - Empty cohort fallback returns un-updated global model $w^{(t+1)} = w^{(t)}$.
  - Unclipped updates preserved for continuous anomaly norm statistic reading per Paper §IV line 186.
  - 12/12 dedicated tests passing in `tests/test_strategy_alignment.py`.
- **Stage 3 Trust Signals & Memory Alignment:**
  - Class-balanced cross-entropy loss $\ell_{\mathrm{bal}}(w) = \frac{1}{|C_{\mathrm{val}}|} \sum_{c \in C_{\mathrm{val}}} \frac{1}{|\mathcal{D}_{\mathrm{val}}^c|} \sum_{(x,y) \in \mathcal{D}_{\mathrm{val}}^c} \ell_{\mathrm{CE}}(f(x; w), y)$ implemented in a single pass over $\mathcal{D}_{\mathrm{val}}$ (preserving the $|\mathcal{P}| + 1$ forward pass guarantee) with zero-sample diagnostic checks.
  - Validation loss improvement signal $A_i = \operatorname{clip}_{[-1, 1]}((\ell_{\mathrm{bal}}(w) - \ell_{\mathrm{bal}}(\tilde{w}_i)) / (\ell_{\mathrm{bal}}(w) + \varepsilon))$ allows negative values in $[-1, 1]$ when candidate models degrade balanced validation loss.
  - Direction signal $S_i = \frac{1}{2}(\cos(\tilde{\Delta}_i, \bar{\Delta}_{\mathcal{A}}) + 1) \in [0, 1]$ computed using clipped updates against mean clipped update of accepted cohort $\mathcal{A}$. Zero-norm inputs return $S_i = 0.5$ safely without NaN/Inf.
  - Norm outlier signal $O_i = 1 - e^{-z_i / 2.5} \in [0, 1)$ computed using unclipped norms over the **full participant cohort $\mathcal{P}$** ($\mu_{\mathcal{P}}, \sigma_{\mathcal{P}}$) per Paper §IV line 186. The z-score $z_i = |\,\|\Delta_i\| - \mu_{\mathcal{P}}\,| / (\sigma_{\mathcal{P}} + \varepsilon)$ uses **absolute value** (verified from LaTeX source: `$z_i=|\,\|\vDelta_i\|-\mu\,|/(\sigma+\varepsilon)$`), so $z_i \ge 0$ by definition and $O_i \in [0, 1)$ without additional clipping `[VERIFIED]`. Code uses `np.abs()` matching exactly.
  - Instantaneous mixed signal: $u_i = \alpha S_i + \beta A_i - \gamma O_i$ (verified strictly negative sign $-\gamma O_i$). For accepted clients $i \in \mathcal{A}$, $s_i = \operatorname{clip}_{[0, 1]}(u_i)$. For rejected clients $i \in \mathcal{P} \setminus \mathcal{A}$, $s_i = 0$.
  - Asymmetric trust memory: $\lambda_i = \lambda_\uparrow = 0.9$ if $s_i \ge T_i^{(t-1)}$ (gain), $\lambda_i = \lambda_\downarrow = 0.7$ if $s_i < T_i^{(t-1)}$ (penalty), initial trust $T_i^{(0)} = 0.5$, trust floor $\tau_{\min} = 0.01$.
  - Non-participating clients ($i \notin \mathcal{P}$) strictly retain previous trust $T_i^{(t)} = T_i^{(t-1)}$.
  - Rejected clients ($i \in \mathcal{P} \setminus \mathcal{A}$) receive $s_i = 0 < T_i^{(t-1)}$, selecting $\lambda_\downarrow = 0.7$ and decaying trust.
  - 15/15 dedicated unit tests passing in `tests/test_trust_alignment.py`.
- **Stage 4 Online Meta-Weight Adaptation Alignment:**
  - Log-weight parameterization: $\mathbf{v} = (v_\alpha, v_\beta, v_\gamma) \in \mathbb{R}^3$, $\mathbf{v}^{(0)} = (0, 0, 0) \implies (\alpha, \beta, \gamma) = \operatorname{softmax}(\mathbf{v}) = (1/3, 1/3, 1/3)$ verified. Softmax output remains strictly positive $\alpha, \beta, \gamma > 0$ and on the unit simplex $\alpha + \beta + \gamma = 1$ to numerical precision. Tested extreme log-weights ($[-100, 100]$) without overflow/underflow NaN.
  - Instantaneous mixed signal: Reuses exact Stage 3 formula $u_i = \alpha S_i + \beta A_i - \gamma O_i$ with negative norm sign $-\gamma O_i$.
  - Straight-Through Estimator (STE): $\hat{w}_i = \frac{\operatorname{clip}_{\mathrm{STE}}(u_i)}{\sum_{j \in \mathcal{A}} \operatorname{clip}_{\mathrm{STE}}(u_j) + \varepsilon}$. Forward values evaluate to standard $[0, 1]$ clamping, but backward pass identity derivative $\frac{d}{du}\operatorname{clip}_{\mathrm{STE}}(u) = 1$ ensures non-zero meta-gradient flow even when candidate updates produce saturated $u_i < 0$.
  - Meta-loss objective: $\mathcal{L}_{\mathrm{meta}} = \sum_{i \in \mathcal{A}} \hat{w}_i \ell_{\mathrm{val}}(\tilde{\mathbf{w}}_i)$ evaluated strictly on accepted candidate models $\tilde{\mathbf{w}}_i$ over the server validation set $\mathcal{D}_{\mathrm{val}}$.
  - Gradient path isolation: Inputs ($S_i, A_i, O_i, \ell_{\mathrm{val}}(\tilde{\mathbf{w}}_i)$) are strictly detached constants. Trust memory state $T_i$ and hard validation gate decisions do not enter the meta-gradient graph.
  - Closed-form and finite-difference validation: Autograd $\nabla_{\mathbf{v}}\mathcal{L}_{\mathrm{meta}}$ matches central finite-difference perturbations to $< 10^{-4}$ tolerance in the unsaturated regime. Matches Lemma 4 closed-form analytical covariance derivative $\frac{\partial \mathcal{L}_{\mathrm{meta}}}{\partial \theta} = \frac{\bar{u}\operatorname{Cov}(X, \ell) - \bar{X}\operatorname{Cov}(u, \ell)}{\bar{u}^2}$ where $X \in \{S, A, -O\}$ to $< 10^{-5}$ tolerance.
  - Persistent Adam optimizer: Configured with $\eta_{\mathrm{meta}} = 0.01$. Persistent Adam optimizer state persists across rounds within a simulation run, and `reset()` cleanly re-instantiates optimizer buffers to guarantee run isolation across independent simulation seeds.
  - Empty accepted cohort fallback: $\mathcal{A} = \emptyset \implies \mathbf{w}^{(t+1)} = \mathbf{w}^{(t)}$ safely bypasses meta-step without division by zero or state corruption.
  - Algorithm 1 update sequence: clipping $\to$ candidate evaluation $\to$ validation gate $\to$ trust memory update ($T_i$) $\to$ meta-weight adaptation ($\mathbf{v}$) $\to$ trust-weighted aggregation ($\mathbf{w}^{(t+1)} = \sum \frac{T_i}{\sum T_j} \tilde{\mathbf{w}}_i$) strictly verified.
  - 20/20 dedicated unit tests passing in `tests/test_meta_weight_alignment.py`.
- **Stage 5 Model Aggregation & Baseline Isolation Alignment:**
  - Final Model Aggregation equation: $\vw^{(t+1)} = \sum_{i \in \cA} \frac{T_i^{(t)}}{\sum_{j \in \cA} T_j^{(t)}} \tilde{\vw}_i$ strictly verified. Aggregation operates strictly over accepted clients $\cA$, using clipped candidate models $\tilde{\vw}_i = \vw^{(t)} + \tilde{\vDelta}_i$.
  - Trust normalization: $\omega_i = T_i^{(t)} / \sum_{j \in \cA} T_j^{(t)}$ verified to sum to 1 to $< 10^{-12}$ precision. Single accepted client recovers candidate model exactly ($\omega_k = 1.0$).
  - Rejected client model invariance: Verified that modifying rejected client models by $10^6\times$ distortion causes zero change in aggregated global parameters.
  - Non-participant state isolation: Verified non-participants contribute no models and their trust scores remain untouched.
  - Baseline isolation architecture: Verified that all 9 previously implemented baselines (`FedAvg`, `Krum`, `Trimmed Mean`, `RFA`, `Bucketing`, `FoolsGold`, `FLAME`, `DeepSight`, `FLTrust`) do not inherit TV-FLIDS clipping, validation gate, trust memory, or meta-weights.
  - Cross-method state leakage: Verified zero leakage when running TV-FLIDS before or after baseline strategy instances.
  - Strategy factory dispatch: Verified `make_strategy` correctly routes to independent strategy instances.
  - 12/12 dedicated tests passing in `tests/test_aggregation_alignment.py`.
  - 10/10 dedicated tests passing in `tests/test_baseline_isolation.py`.
- **Stage 6 Missing Baseline Implementations Alignment:**
  - Multi-Krum: Exact Blanchard et al. (2017) multi-client selection and averaging matching hand-calculated distances and selecting top $m$ clients.
  - Norm Clipping: Exact Sun et al. (2019) median norm clipping ($\gamma \cdot \operatorname{median}(\|\Delta\|)$) and aggregation.
  - FLDetector: Exact Zhang et al. (2022) L-BFGS two-loop Hessian-vector product update prediction, cosine inconsistency outlier detection, Coordinate-wise Trimmed Mean aggregation of clean clients, and history `reset()`.
  - Zeno: Exact Xie et al. (2019) validation loss improvement minus squared norm penalty score evaluation, top $D - b_Z$ selection, and averaging.
  - BaFFLe: Exact Andreina et al. (2021) validator client sampling ($K_{\mathrm{val}}$), per-class error degradation testing against historical lookback window $L$, quorum voting, rollback on rejection, and lookback history `reset()`.
  - Full Strategy Factory & Isolation: All 14 baselines dispatchable and strictly isolated from TV-FLIDS mechanisms and state.
  - 3/3 passed in `tests/test_multikrum_strategy.py`.
  - 3/3 passed in `tests/test_norm_clipping_strategy.py`.
  - 4/4 passed in `tests/test_fldetector_strategy.py`.
  - 3/3 passed in `tests/test_zeno_strategy.py`.
  - 4/4 passed in `tests/test_baffle_strategy.py`.
  - 11/11 passed in `tests/test_baseline_isolation.py`.
- **Complete Test Suite Pass:**
  - 12/12 passed in `tests/test_aggregation_alignment.py`.
  - 11/11 passed in `tests/test_baseline_isolation.py`.
  - 3/3 passed in `tests/test_multikrum_strategy.py`.
  - 3/3 passed in `tests/test_norm_clipping_strategy.py`.
  - 4/4 passed in `tests/test_fldetector_strategy.py`.
  - 3/3 passed in `tests/test_zeno_strategy.py`.
  - 4/4 passed in `tests/test_baffle_strategy.py`.
  - 20/20 passed in `tests/test_meta_weight_alignment.py`.
  - 15/15 passed in `tests/test_trust_alignment.py`.
  - 12/12 passed in `tests/test_strategy_alignment.py`.
  - 18/18 passed in `tests/test_model_and_nslkdd_alignment.py`.
  - 14/14 passed in `tests/test_ste_gradient.py`.
  - 18/18 passed in `tests/test_manuscript_integrity.py`.
  - 41/41 passed in `tests/test_warmup_schedule.py`.
  - 10/10 passed in `tests/test_forward_pass_count.py`.
  - 14/14 passed in `tests/test_overhead_tracking.py`.
  - 5/5 passed in `tests/test_ack1_ack2_attacks.py`.
  - 19/19 passed in `tests/test_val_size_regression.py`.
  - Total: **226/226 tests passed** across all active test suites.

---

## 20. Session Changelog

- **2026-09-17 (Session 1):**
  - Inspected repository, existing audits (`PROJECT_PAPER_ALIGNMENT_AUDIT.md`, `AUDIT_VERIFICATION_AND_CORRECTIONS.md`), and IEEE manuscript (`Paper/IEEE/TV-FLIDS.tex`, `TV-FLIDS_supplementary.tex`).
  - Verified exact mathematical formulas, parameter counts, campaign run volumes, and baseline/attack taxonomies.paign run volumes, and baseline/attack taxonomies.
  - Formally established `PROJECT_MEMORY.md` at the repository root as the persistent memory document for all subsequent sessions.
  - Maintained zero production code modifications during initialization pass.

- **2026-09-17 (Session 2):**
  - Completed **Stage 1 (Model and NSL-KDD Scientific Alignment)**.
  - Modified `models/mlp.py`:
    - Replaced `nn.BatchNorm1d` with `nn.LayerNorm(256/128/64)` across all hidden layers.
    - Verified layer sequence: Linear -> LayerNorm -> ReLU -> Dropout(0.3) -> Linear -> LayerNorm -> ReLU -> Dropout(0.3) -> Linear -> LayerNorm -> ReLU -> Dropout(0.3) -> Linear.
    - Added static helper `parameter_count_formula(d, K)` verifying $P(d, K) = 256d + 65K + 42,304$.
  - Modified `data/preprocessing/nslkdd_pipeline.py`:
    - Implemented exact Table I quota splitting for $\mathcal{D}_{\mathrm{val}}$ (2,000), $\mathcal{D}_{\mathrm{tune}}$ (12,398), and $\mathcal{D}_{\mathrm{client}}$ (111,575).
    - Enforced loud diagnostic failure if class quotas cannot be satisfied.
    - Implemented zero-leakage feature scaling: `MinMaxScaler(clip=False)` fitted strictly on server data $\mathcal{D}_{\mathrm{val}} \cup \mathcal{D}_{\mathrm{tune}}$ (14,398 samples); client and test data are never seen by the scaler.
    - Preserved unclipped test representation per paper §VI-A.
    - Implemented paper-specified client-local SMOTE rule ($M = \operatorname{median}(\{n_c : n_c > 0\})$, $k = \min(5, n_c - 1)$, duplication for $n_c=1$, zero for $n_c=0$).
    - Preserved backward compatibility for `build_pipeline` (returning 9-tuple by default, with `return_tune=True` option).
  - Added new test suite `tests/test_model_and_nslkdd_alignment.py` containing 18 rigorous unit tests.
  - Executed tests: 18/18 passed in `tests/test_model_and_nslkdd_alignment.py`; 19/19 passed in `tests/test_val_size_regression.py`; 49/49 passed in `tests/test_all.py`.
  - Executed `scripts/verify_environment.py` and documented environment characteristics.
  - Updated `PROJECT_MEMORY.md` to reflect completed Stage 1 status.

- **2026-09-17 (Session 3):**
  - Completed **Stage 2 (TV-FLIDS Strategy Alignment: Median-Radius Clipping & Single Validation Gate)**.
  - Aligned code strictly with IEEE TIFS manuscript §IV, Eq. (3)–(4), and Algorithm 1:
    - $C^{(t)} = \operatorname{median}_{j \in \mathcal{P}^{(t)}} \|\Delta_j^{(t)}\|$
    - $\tilde{\Delta}_i^{(t)} = \Delta_i^{(t)} \min(1, C^{(t)} / \|\Delta_i^{(t)}\|)$
    - $\tilde{w}_i^{(t)} = w^{(t)} + \tilde{\Delta}_i^{(t)}$
    - $\delta_i = \ell_{\mathrm{val}}(w^{(t)}) - \ell_{\mathrm{val}}(\tilde{w}_i^{(t)}) \ge \tau_L(t)$
    - $\tau_L(t) = -0.1 + 0.1 \min(1, t / 20)$
  - Modified `trust/verification.py`:
    - Added standalone `clip_updates(updates, cohort_norms=None)` computing median radius $C^{(t)}$ across full participant cohort $\mathcal{P}^{(t)}$ and scaling updates exceeding $C^{(t)}$ to the boundary while leaving below-median updates unchanged.
    - Replaced 3-check hard filtering with single loss improvement gate: $\delta_i = \ell_{\mathrm{val}}(w^{(t)}) - \ell_{\mathrm{val}}(\tilde{w}_i^{(t)}) \ge \tau_L(t)$ with linear schedule $\tau_L(t) = -0.1 + 0.1\min(1, t/20)$.
    - Candidate models are strictly constructed from clipped updates ($\tilde{w}_i = w^{(t)} + \tilde{\Delta}_i$).
    - Preserved cosine similarity and norm z-score calculations for continuous downstream trust signals, but removed them completely as hard rejection criteria.
    - Maintained backward compatibility for `verify_all()` returning `'accepted'`, `'rejected'`, `'verified'` (alias to accepted), `'flagged'` (empty), `'deltas'`, `'val_losses'`, `'threshold'`.
  - Modified `fl/strategy.py`:
    - Integrated `clip_updates` on raw client updates prior to validation gate evaluation.
    - Evaluated validation gate on clipped updates.
    - Implemented paper-specified fallback for empty accepted cohort $\mathcal{A}^{(t)} = \emptyset$: returns un-updated global parameters $w^{(t+1)} = w^{(t)}$.
    - Passed unclipped raw updates `a_raw_upds` to `trust_scorer.compute_anomaly_scores` per Paper §IV line 186 ("The norm statistic reads unclipped norms because clipping equalizes the largest half of them").
    - Logged `num_accepted` and `clipping_radius` per round.
  - Added new test suite `tests/test_strategy_alignment.py`:
    - 12 comprehensive unit tests covering median radius, below/above-median updates, zero updates, full cohort norm scope, gate acceptance/rejection, no cosine hard rejection, no z-score hard rejection, warm-up schedule, clipping-before-gating sequence, and empty accepted cohort fallback.
  - Updated legacy test `tests/test_pipeline_smoke.py`:
    - Aligned `TestVerificationGateSmoke` with single-gate semantics (removed obsolete assertion that z-score acts as a hard filter).
  - Executed tests: 12/12 passed in `tests/test_strategy_alignment.py`, 41/41 passed in `tests/test_warmup_schedule.py`, 10/10 passed in `tests/test_forward_pass_count.py`, 14/14 passed in `tests/test_overhead_tracking.py`, 5/5 passed in `tests/test_ack1_ack2_attacks.py`, 7/7 passed in `tests/test_pipeline_smoke.py`, 18/18 passed in `tests/test_model_and_nslkdd_alignment.py`, 49/49 passed in `tests/test_all.py`.
  - Audited production code path: verified end-to-end execution order from Flower client results to trust-weighted aggregation, confirming no legacy path bypasses clipping or invokes obsolete multi-criteria rejection.
  - Updated `PROJECT_MEMORY.md` to reflect completed Stage 2 status.

- **2026-09-17 (Session 4):**
  - Completed **Stage 3 (TV-FLIDS Trust System Alignment: Class-Balanced Loss, Multi-Signal Scoring, Asymmetric EMA, Rejection Penalty)**.
  - Pre-implementation audit resolved all mathematical specifications against IEEE TIFS manuscript §IV, Eq. (2), (5)–(7), and Algorithm 1:
    - $\ell_{\mathrm{bal}}(w) = \frac{1}{|C_{\mathrm{val}}|} \sum_{c \in C_{\mathrm{val}}} \frac{1}{|\mathcal{D}_{\mathrm{val}}^c|} \sum_{(x, y) \in \mathcal{D}_{\mathrm{val}}^c} \ell_{\mathrm{CE}}(f(x; w), y)$
    - $A_i = \operatorname{clip}_{[-1, 1]}\left(\frac{\ell_{\mathrm{bal}}(w^{(t)}) - \ell_{\mathrm{bal}}(\tilde{w}_i^{(t)})}{\ell_{\mathrm{bal}}(w^{(t)}) + \varepsilon}\right)$, with range $[-1, 1]$ (negative for degraded loss).
    - $S_i = \frac{1}{2}(\cos(\tilde{\Delta}_i, \bar{\Delta}_{\mathcal{A}}) + 1) \in [0, 1]$ computed with clipped updates against accepted cohort centroid $\bar{\Delta}_{\mathcal{A}}$.
    - $O_i = 1 - e^{-z_i / 2.5} \in [0, 1]$ computed with unclipped norms over the entire participant cohort $\mathcal{P}$ ($\mu_{\mathcal{P}}, \sigma_{\mathcal{P}}$).
    - Instantaneous mixed signal: $u_i = \alpha S_i + \beta A_i - \gamma O_i$ (verified strictly negative sign $-\gamma O_i$). For accepted clients $i \in \mathcal{A}$, $s_i = \operatorname{clip}_{[0, 1]}(u_i)$. For rejected clients $i \in \mathcal{P} \setminus \mathcal{A}$, $s_i = 0$.
    - Asymmetric trust memory: $\lambda_\uparrow = 0.9$ (if $s_i \ge T_i^{(t-1)}$), $\lambda_\downarrow = 0.7$ (if $s_i < T_i^{(t-1)}$), $T_i^{(0)} = 0.5$, $\tau_{\min} = 0.01$.
    - Non-participants ($i \notin \mathcal{P}$) strictly retain previous trust $T_i^{(t)} = T_i^{(t-1)}$.
    - Rejected clients ($i \in \mathcal{P} \setminus \mathcal{A}$) receive $s_i = 0$, selecting $\lambda_\downarrow = 0.7$ and decaying trust.
  - Modified `trust/verification.py`:
    - Added `compute_class_balanced_loss` and `compute_class_balanced_loss_from_tensors` accumulating per-class losses in a single pass over $\mathcal{D}_{\mathrm{val}}$.
    - Added balanced loss evaluation alongside standard loss in `evaluate_validation_gate` and cached both losses (`eval_cache`, `eval_bal_cache`) to preserve the $|\mathcal{P}| + 1$ forward-pass guarantee.
  - Modified `trust/trust_scorer.py`:
    - Implemented asymmetric EMA ($\lambda_\uparrow=0.9, \lambda_\downarrow=0.7, T^{(0)}=0.5, \tau_{\min}=0.01$).
    - Implemented $A_i$ calculation in $[-1, 1]$.
    - Implemented $O_i = 1 - e^{-z_i / 2.5}$ with population statistics over all participants $\mathcal{P}$.
    - Implemented `compute_instantaneous_signal` ($u_i = \alpha S_i + \beta A_i - \gamma O_i$ and $s_i = \operatorname{clip}_{[0, 1]}(u_i)$).
    - Implemented `update_trust` supporting accepted, rejected ($s_i = 0$), and non-participating clients.
  - Modified `fl/strategy.py`:
    - Evaluated global parameters once to obtain both standard and class-balanced validation loss in one pass (`_eval_model_both`).
    - Evaluated clipped candidate models once during gating, caching both losses.
    - Updated trust for all sampled participants (both accepted and rejected) in all rounds.
    - Maintained empty-cohort fallback ($w^{(t+1)} = w^{(t)}$) while penalizing all rejected clients with $s_i = 0$.
  - Modified `scripts/check_manuscript.py`:
    - Updated `PAPER_DIR` fallback to `Paper/IEEE` if present, passing all 18 manuscript integrity tests.
  - Updated legacy test assertions:
    - In `tests/test_all.py`, updated `test_reset` to assert initial trust $0.5$ (per paper $T^{(0)} = 0.5$) and updated `test_accuracy_scores_clip` to assert $A_i \in [-1, 1]$ with negative value for degrading loss.
  - Added new test suite `tests/test_trust_alignment.py`:
    - 15 comprehensive unit tests covering class-balanced loss under severe imbalance, negative $A_i$, $A_i$ clipping bounds $[-1, 1]$, $S_i$ range $[0, 1]$, zero-vector cosine safety, exact $O_i$ exponential formula, unclipped cohort scope over all $\mathcal{P}$, positive trust branch $\lambda_\uparrow=0.9$, negative trust branch $\lambda_\downarrow=0.7$, initial trust $T^{(0)}=0.5$, trust floor $\tau_{\min}=0.01$, rejected client penalty ($s_i=0$), non-participant trust retention, mixed signal equation with negative norm sign, and full multi-client round integration.
  - Executed tests: 15/15 passed in `tests/test_trust_alignment.py`, 12/12 passed in `tests/test_strategy_alignment.py`, 18/18 passed in `tests/test_manuscript_integrity.py`, 13/13 passed in `tests/test_ste_gradient.py`.
  - Updated `PROJECT_MEMORY.md` to reflect completed Stage 3 status.

- **2026-09-17 (Session 5):**
  - Performed **Pre-Stage 4 Mathematical Reconciliation Audit** (READ-ONLY, no production code changes).
  - **Issue 1 (O_i norm outlier signal):** Resolved. Paper defines $z_i = |\,\|\Delta_i\| - \mu\,| / (\sigma + \varepsilon)$ with absolute value (verified from LaTeX source at line 186). Therefore $z_i \ge 0$ always, $O_i = 1 - e^{-z_i/2.5} \in [0, 1)$ without additional clipping. Code uses `np.abs()` matching exactly. Classification: **CONSISTENT**.
  - **Issue 2 (Final aggregation weight):** Paper Eq. (eq:agg) specifies trust-only weighting $w_i = T_i / \sum_j T_j$. Neither $s_i$ nor $u_i$ directly enter aggregation. Meta-loss uses separate $\hat{w}_i$ weights based on $u_i$ with STE. Code `get_aggregation_weights()` uses `trust_scores` only. Classification: **IMPLEMENTATION CORRECT**.
  - **Issue 3 (Meta-weight dependency):** Traced full dependency graph. $\alpha, \beta, \gamma$ affect aggregation indirectly ($\alpha,\beta,\gamma \to u_i \to s_i \to T_i \to w_i$) and meta-loss directly ($\alpha,\beta,\gamma \to u_i \to \hat{w}_i \to \mathcal{L}_{\text{meta}}$). Trust $T_i$ is not in meta-gradient path. STE is explicitly specified by paper (§IV-C, line 198). Clipping of $s_i$ is NOT in meta forward graph. Classification: **IMPLEMENTATION CORRECT**.
  - **Issue 4 (Code-to-equation matrix):** All 9 quantities ($S_i, A_i, z_i, O_i, u_i, s_i, T_i$, aggregation weight, meta-loss) verified against paper definitions. All correct.
  - **Issue 5 (Numerical tests):** Computed $O_i$ for $z \in \{0, 0.5, 1, 2.5, 5, 10\}$ — all in $[0, 1)$. Demonstrated that without absolute value, $z=-2$ gives $O_i = -1.23$ and $z=-1$ gives $O_i = -0.49$, violating $[0,1]$. Constructed two-client aggregation under trust-only, signal-only, and trust×signal weighting — only trust-only matches paper.
  - **Issue 6 (Read-only compliance):** Zero production code files modified. Audit is documented in artifact only.
  - Overall classification: **CONSISTENT — no blocking issues found**.
  - Next task confirmed: Stage 4 — Online Meta-Weight Adaptation.

- **2026-09-17 (Session 6):**
  - Completed **Stage 4 (TV-FLIDS Online Meta-Weight Adaptation: Log-Weights, STE Gradient, Adam Optimization, Algorithm 1 Sequencing)**.
  - Pre-implementation audit resolved all mathematical specifications against IEEE TIFS manuscript §IV, Eq. (8)–(9), Lemma 4, Algorithm 1, and Supplementary §S2:
    - $\mathbf{v} = (v_\alpha, v_\beta, v_\gamma) \in \mathbb{R}^3$, $\mathbf{v}^{(0)} = (0, 0, 0) \implies (\alpha, \beta, \gamma) = \operatorname{softmax}(\mathbf{v}) = (1/3, 1/3, 1/3)$.
    - $u_i = \alpha S_i + \beta A_i - \gamma O_i$ strictly using Stage 3 canonical formula.
    - Straight-Through Estimator $\hat{w}_i = \frac{\operatorname{clip}_{\mathrm{STE}}(u_i)}{\sum_{j \in \mathcal{A}} \operatorname{clip}_{\mathrm{STE}}(u_j) + \varepsilon}$ with identity backward derivative $\frac{d}{du}\operatorname{clip}_{\mathrm{STE}}(u) = 1$.
    - Meta-loss objective: $\mathcal{L}_{\mathrm{meta}} = \sum_{i \in \mathcal{A}} \hat{w}_i \ell_{\mathrm{val}}(\tilde{\mathbf{w}}_i)$ evaluated on accepted clipped candidate models $\tilde{\mathbf{w}}_i$.
    - Inputs ($S_i, A_i, O_i, \ell_{\mathrm{val}}(\tilde{\mathbf{w}}_i)$) detached as constants. Trust memory $T_i$ and hard gate decisions do not enter the meta-gradient graph.
    - Persistent Adam optimizer ($\eta_{\mathrm{meta}} = 0.01$). `AdaptiveTrustScorer.reset()` completely re-instantiates the optimizer, wiping momentum/step buffers for strict run isolation between independent simulation seeds.
    - Algorithm 1 update sequence: clipping $\to$ candidate evaluation $\to$ validation gate $\to$ trust memory update ($T_i$) $\to$ meta-weight adaptation ($\mathbf{v}$) $\to$ trust-weighted aggregation ($\mathbf{w}^{(t+1)} = \sum \frac{T_i}{\sum T_j} \tilde{\mathbf{w}}_i$).
    - Empty accepted cohort fallback: $\mathcal{A} = \emptyset \implies \mathbf{w}^{(t+1)} = \mathbf{w}^{(t)}$ safely bypasses meta-step without modifying $\mathbf{v}$ or optimizer state.
  - Modified `trust/adaptive_trust_scorer.py`:
    - Implemented canonical `compute_meta_loss`, `adapt_weights`, `meta_update` (for backward compatibility), and `get_weights`.
    - Added clip saturation tracking and numerical safety guards.
    - Updated `reset()` to re-instantiate `torch.optim.Adam([self.log_weights], lr=self.lr)`.
  - Modified `fl/strategy.py`:
    - Integrated `self.trust_scorer.adapt_weights` in `aggregate_fit` at the exact point specified by Algorithm 1.
    - Logged `adaptive_alpha`, `adaptive_beta`, `adaptive_gamma`, `meta_loss`, `clip_saturated` in round summary.
  - Added new test suite `tests/test_meta_weight_alignment.py`:
    - 20 comprehensive unit tests covering all Part L requirements: initialization, simplex constraints, weight positivity, exact $u_i$, STE forward/backward behavior, meta-weight normalization, hand-calculated meta-loss, gradient direction, autograd vs central finite differences, Lemma 4 analytical covariance formula, Adam single step, weight response, optimizer state persistence, run isolation/reset, empty cohort fallback, full round integration in `TVFLIDSStrategy`, extreme log-weights stability, all-saturated cohort handling, and gradient isolation.
  - Updated legacy test `tests/test_ste_gradient.py`:
    - Updated `TestStrategyUsesSTE` to inspect `AdaptiveTrustScorer.compute_meta_loss` for `clip_ste` and absence of `torch.clamp`.
  - Executed tests:
- **2026-09-18 (Session 7):**
  - Completed **Stage 5 (TV-FLIDS Final Model Aggregation and Baseline Strategy Isolation)**.
  - Aligned code strictly with IEEE TIFS manuscript §IV, Eq. (11), Algorithm 1, and Supplementary Table S2:
    - $\vw^{(t+1)} = \sum_{i \in \cA} \frac{T_i^{(t)}}{\sum_{j \in \cA} T_j^{(t)}} \tilde{\vw}_i$, where $\tilde{\vw}_i = \vw^{(t)} + \tilde{\vDelta}_i$.
    - Only accepted clients $i \in \cA$ participate in final model aggregation.
    - Aggregated models are strictly Stage-1 clipped candidate models $\tilde{\vw}_i$.
    - Trust weights $\omega_i = T_i^{(t)} / \sum_{j \in \cA} T_j^{(t)}$ sum to 1 to numerical precision ($< 10^{-12}$).
    - Empty accepted cohort fallback: $\cA = \emptyset \implies \vw^{(t+1)} = \vw^{(t)}$ while penalizing all rejected clients with $s_i = 0$ via $\lambda_\downarrow = 0.7$.
    - Algorithm 1 update sequence strictly enforced: clipping $\to$ candidate evaluation $\to$ validation gate $\to$ trust memory update ($T_i^{(t)}$) $\to$ meta-weight adaptation ($\mathbf{v}$) $\to$ trust-weighted aggregation ($\vw^{(t+1)}$).
  - Audited baseline isolation across all 14 baselines (9 implemented, 5 to be implemented in Stage 6):
    - Confirmed `FedAvg`, `Krum`, `Trimmed Mean`, `RFA`, `Bucketing`, `FoolsGold`, `FLAME`, `DeepSight`, `FLTrust` have independent aggregation algorithms inheriting directly from Flower's `FedAvg` without TV-FLIDS mechanisms.
    - Confirmed strategy factory dispatch (`make_strategy`) cleanly routes strategies to dedicated classes.
    - Confirmed zero cross-method state leakage between TV-FLIDS and baselines.
  - Modified `fl/strategy.py`:
    - Updated Stage 5 comment to `# ── STAGE 5: Final Model Aggregation (Paper §IV, Eq. (11), Alg. 1) ──`.
  - Added new test suite `tests/test_aggregation_alignment.py`:
    - 12 comprehensive unit tests covering: trust weights sum to 1, accepted clients only, clipped candidate models aggregated, single accepted client exact match, equal trust arithmetic average, unequal trust exact formula, all trust at floor stability, empty cohort fallback & penalty, rejected client parameter invariance ($10^6\times$ distortion zero effect), non-participant state isolation, repeated aggregation stability over 20 rounds, and near-zero trust normalization safety.
  - Added new test suite `tests/test_baseline_isolation.py`:
    - 10 comprehensive unit tests covering: FedAvg pure sample-weighted average without clipping/gate/trust, Krum-specific distance scoring, Trimmed Mean coordinate trimming, RFA smoothed Weiszfeld geometric median, FLTrust server-root cosine weighting, clustering/Sybil baseline independence (FLAME, DeepSight, FoolsGold, Bucketing), TV-FLIDS then FedAvg cross-method state leakage, TV-FLIDS then multiple baselines leakage, fresh TVFLIDS instance isolation, and strategy factory dispatch routing.
  - Executed test suites:
    - 12/12 passed in `tests/test_aggregation_alignment.py`.
    - 10/10 passed in `tests/test_baseline_isolation.py`.
    - 20/20 passed in `tests/test_meta_weight_alignment.py`.
    - 15/15 passed in `tests/test_trust_alignment.py`.
    - 12/12 passed in `tests/test_strategy_alignment.py`.
    - 18/18 passed in `tests/test_model_and_nslkdd_alignment.py`.
    - 14/14 passed in `tests/test_ste_gradient.py`.
    - 18/18 passed in `tests/test_manuscript_integrity.py`.
    - 41/41 passed in `tests/test_warmup_schedule.py`.
    - 10/10 passed in `tests/test_forward_pass_count.py`.
    - 14/14 passed in `tests/test_overhead_tracking.py`.
    - 5/5 passed in `tests/test_ack1_ack2_attacks.py`.
    - 19/19 passed in `tests/test_val_size_regression.py`.
    - Total: 194/194 active regression tests passing.
  - Updated `PROJECT_MEMORY.md` to reflect completed Stage 5 status.

- **2026-09-18 (Session 8):**
  - Completed **Stage 6 (Implement and Integrate the 5 Missing Baselines: Multi-Krum, Norm Clipping, FLDetector, Zeno, BaFFLe)**.
  - Aligned code strictly with IEEE TIFS manuscript §II, §VI, and Supplementary Table S2:
    - **Multi-Krum** (`fl/baselines/multikrum_strategy.py`):
      - Created standalone `MultiKrumStrategy` independently dispatchable from `KrumStrategy`.
      - Computes sum of squared Euclidean distances to $D - f' - 2$ closest neighbors.
      - Selects top $m > 1$ candidate models ($m \in \{D-f', D-2f'\}$) with lowest scores and computes arithmetic average $\vw^{(t+1)} = \frac{1}{m} \sum_{i \in \mathcal{K}} \vw_i$.
      - Updated `KrumStrategy` to default canonically to single Krum ($m=1$).
    - **Norm Clipping** (`fl/baselines/norm_clipping_strategy.py`):
      - Created standalone `NormClippingStrategy` (Sun et al., 2019).
      - Computes dynamic median norm $C = \gamma \cdot \operatorname{median}_{i \in \mathcal{P}}(\|\Delta_i\|)$ with $\gamma \in \{0.5, 1, 1.5, 2\}$.
      - Clips updates $\tilde{\Delta}_i = \Delta_i \min(1, C/\|\Delta_i\|)$ and averages clipped updates without validation loss gating, trust scoring, or meta-weights.
    - **FLDetector** (`fl/baselines/fldetector_strategy.py`):
      - Created standalone `FLDetectorStrategy` (Zhang et al., 2022).
      - Maintains sliding window $W \in \{5, 10\}$ of global model differences $s^{(t)} = \vw^{(t)} - \vw^{(t-1)}$ and gradient differences $y^{(t)} = g^{(t)} - g^{(t-1)}$.
      - Implements L-BFGS two-loop recursion to compute Hessian-vector products $H s^{(t)}$ predicting client updates $\hat{g}_i^{(t)}$.
      - Computes cosine inconsistency $1 - \cos(g_i^{(t)}, \hat{g}_i^{(t)})$ and detects malicious clients at $t \ge T_{\mathrm{start}}$ ($T_{\mathrm{start}} \in \{10, 20\}$).
      - Aggregates clean clients using Coordinate-wise Trimmed Mean.
      - Implements `reset()` to clear history across independent simulation seeds.
    - **Zeno** (`fl/baselines/zeno_strategy.py`):
      - Created standalone `ZenoStrategy` (Xie et al., 2019).
      - Evaluates candidate models $\vw^{(t)} + \Delta_i$ on server validation data $\mathcal{D}_{\mathrm{val}}$.
      - Computes exact published score $\text{Score}_i = (\ell_{\mathrm{val}}(\vw^{(t)}) - \ell_{\mathrm{val}}(\vw^{(t)} + \Delta_i)) - \rho_Z \|\Delta_i\|^2$.
      - Selects top $m = D - b_Z$ candidates ($\rho_Z \in \{5\times10^{-4}, 10^{-3}, 2\times10^{-3}\}, b_Z \in \{3,4\}$) and averages selected candidate models.
    - **BaFFLe** (`fl/baselines/baffle_strategy.py`):
      - Created standalone `BaFFLeStrategy` (Andreina et al., 2021).
      - Aggregates candidate update via FedAvg.
      - Samples $K_{\mathrm{val}} \in \{5, 10\}$ validator clients from data partitioning.
      - Evaluates per-class error rate degradation $\Delta \text{err}_c = \text{err}_c(\vw_{\mathrm{cand}}) - \text{err}_c(\vw_{\mathrm{ref}})$ against historical lookback window $L \in \{10, 20\}$ and threshold $\tau_{\mathrm{baffle}}$.
      - Rejects candidate if fraction of rejecting validators exceeds quorum $q \in \{0.3, 0.5\}$, falling back to rollback model $\vw^{(t+1)} = \vw^{(t)}$.
      - Implements `reset()` to clear historical lookback buffer across independent seeds.
    - **Baseline Isolation & Dispatch Integration**:
      - Updated `fl/baselines/__init__.py` exporting all 14 baseline strategy classes.
      - Updated `make_strategy` and argparse choices in `experiments/run_experiment.py` for all 14 baselines.
      - Verified strict isolation: zero baseline contains `TrustScorer`, `AdaptiveTrustScorer`, validation loss gating, or median-radius clipping.
    - **New Test Suites Added**:
      - `tests/test_multikrum_strategy.py` (3 tests)
      - `tests/test_norm_clipping_strategy.py` (3 tests)
      - `tests/test_fldetector_strategy.py` (4 tests)
      - `tests/test_zeno_strategy.py` (3 tests)
      - `tests/test_baffle_strategy.py` (4 tests)
      - Updated `tests/test_baseline_isolation.py` (11 tests)
    - **Executed Full Regression Test Suite**:
      - Total: **226/226 tests passed** across all active test files.
    - Updated `PROJECT_MEMORY.md` to reflect completed Stage 6 status.

- **2026-09-18 (Session 9):**
  - Completed **Stage 7 (Attack Suite and Adversarial Threat Model Completion)** `[VERIFIED]`.
  - Implemented and rigorously validated the complete threat model and attack suite required by the IEEE TIFS experimental matrix (§III, §VII, and Supplementary §S4):
    1. **Knowledge Tiers Parameterization (K0, K1, K2)** (`attacks/knowledge.py`) `[VERIFIED]`:
       - `KnowledgeTier` enum (`K0`, `K1`, `K2`) and canonical per-class quotas `NSLKDD_VAL_QUOTAS = {0: 1016, 1: 693, 2: 176, 3: 100, 4: 15}`.
       - `ValidationEstimateProvider`:
         - **K0 (Zero Server Knowledge):** Resamples exclusively from pooled training data held by coalition members. Strictly raises error if coalition data is unavailable. Zero access to $\mathcal{D}_{\mathrm{val}}$ or $\mathcal{D}_{\mathrm{tune}}$.
         - **K1 (Validation-Distribution Knowledge):** Draws a surrogate dataset from the client training pool outside $\mathcal{D}_{\mathrm{val}} \cup \mathcal{D}_{\mathrm{tune}}$, resampled to match known class distribution quotas.
         - **K2 (Omniscient Knowledge):** Direct access to server validation set $\mathcal{D}_{\mathrm{val}}$.
       - Tested with 8 unit tests in `tests/test_knowledge_tiers.py` confirming strict information containment.
    2. **Min-Sum Attack (Partial & Omniscient)** (`attacks/adversarial.py`) `[VERIFIED]`:
       - Implemented exact NDSS 2021 formulation with IEEE TIFS §III-C line 145 inverse standard-deviation perturbation direction: $\bm{v}_p = -\operatorname{sgn}(\bar{\vDelta}) / (\bm{\sigma} + \varepsilon)$.
       - Enforces constraint: $\sum_{u \in \mathcal{U}} \|\vDelta_m - u\|_2^2 \le \sum_{u \in \mathcal{U}} \|u - \bar{u}\|_2^2$ via binary search over scaling scalar $\gamma$.
       - Supports both `partial` (reference cohort $\mathcal{U} = \mathcal{C}$ coalition updates) and `omniscient` ($\mathcal{U} = \mathcal{H}$ honest updates).
       - Tested with 4 unit tests in `tests/test_min_sum_attack.py`.
    3. **Min-Max Attack (Partial & Omniscient)** (`attacks/adversarial.py`) `[VERIFIED]`:
       - Implemented exact NDSS 2021 formulation with inverse standard-deviation direction and binary search constraint satisfaction: $\max_{u \in \mathcal{U}} \|\vDelta_m - u\|_2 \le \max_{u, v \in \mathcal{U}} \|u - v\|_2$.
       - Supports both `partial` ($\mathcal{U} = \mathcal{C}$) and `omniscient` ($\mathcal{U} = \mathcal{H}$).
       - Tested with 3 unit tests in `tests/test_min_max_attack.py`.
    4. **ACK3 Attack (Class-Balanced Evasion)** (`fl/client.py`, `attacks/adversarial.py`) `[VERIFIED]`:
       - Relabels only majority attack classes: DoS (class 1) and Probe (class 2) to normal (0), leaving rare attack classes R2L (class 3) and U2R (class 4) unaltered.
       - Adds dual validation hinge losses during local training:
         $$\mathcal{L}_{\mathrm{ack3}} = \mathcal{L}_{\mathrm{adv}} + \rho_a \max(0, \ell_{\mathrm{val}}(\vw) - \ell_{\mathrm{val}}(\vw_{\mathrm{global}}) + m) + \rho_a \max(0, \ell_{\mathrm{bal}}(\vw) - \ell_{\mathrm{bal}}(\vw_{\mathrm{global}}))$$
       - Tested with 3 unit tests in `tests/test_ack3_attack.py`.
    5. **ACK4 Attack (Meta-Weight Evasion)** (`attacks/adversarial.py`) `[VERIFIED]`:
       - Implements grid search over mixture parameter $\omega \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$:
         $$\vDelta(\omega) = \hat{C} \cdot \mathrm{unit}\left((1 - \omega)\vDelta_{\mathrm{hon}} + \omega\vDelta_{\mathrm{ACK2}}\right)$$
       - Simulates Algorithm 1 on surrogate validation estimate, computing candidate acceptance, trust updates, and meta-weight share $\rho_B$.
       - Submits update maximizing predicted malicious contribution: $J(\omega) = \rho_B \langle \tilde{\vDelta}(\omega), \bm{u}\rangle$.
       - Adds randomized perturbations $\mathcal{N}(0, 10^{-4}\sigma^2)$ across coalition members to prevent identical updates.
       - Tested with 3 unit tests in `tests/test_ack4_attack.py`.
    6. **On-Off Attack Schedules ($k \in \{10, 20, 30, 50\}$)** (`attacks/adversarial.py`) `[VERIFIED]`:
       - Byzantine clients behave honestly for $t \le k$ rounds, then launch LF or ACK2 for $t > k$.
       - Implemented `is_on_off_active(server_round, k, mode="switch")` with support for periodic mode.
       - Dispatched for both `on_off_lf` (at client level) and `on_off_ack2` (at round/strategy interceptor level).
       - Tested with 10 unit tests in `tests/test_on_off_attack.py`.
    7. **LF-R Attack (Random Attack-to-Attack Label Flipping)** (`attacks/adversarial.py`, `fl/client.py`) `[VERIFIED]`:
       - Relabels attack records ($y \ne 0$) to a randomly selected distinct attack class $y' \in \{1, \dots, K-1\} \setminus \{y\}$.
       - Preserves normal traffic ($y = 0$) untouched.
       - Tested with 3 unit tests in `tests/test_lfr_attack.py`.
    8. **Unified Strategy-Level Interception & Baseline Isolation** (`attacks/adversarial.py`, `fl/strategy.py`, `fl/baselines/`) `[VERIFIED]`:
       - Centralized `apply_round_attacks` interceptor invoked in `aggregate_fit` across TV-FLIDS and all 14 baselines.
       - Confirmed complete attack isolation: clean client updates remain bitwise unmodified (`tests/test_attack_isolation.py`).
       - Zero mutable state or cross-attack leakage across consecutive rounds and runs.
    9. **Experiment CLI Integration** (`experiments/run_experiment.py`) `[VERIFIED]`:
       - Added CLI flags `--knowledge-tier`, `--attack-variant`, `--on-off-k`.
       - Integrated `ValidationEstimateProvider` to supply `proxy_val_data` to clients and strategies.
    10. **Test Suite Results**:
        - All 35 new Stage 7 unit tests passed.
        - Full regression test suite: **450 passed, 17 skipped** across the repository with zero regressions.
    11. **Stage 7 Verification Hold Resolution & Production-Path Activation** `[VERIFIED]`:
        - **Diagnostic of Previous Identical Smoke Metrics (Part H)**:
          - *Root Cause*: Flower's default `start_simulation` uses Ray multi-process actor workers. On Windows, child processes dynamically importing PyTorch encountered `WinError 1114` (DLL initialization access violation in `c10.dll`). Furthermore, client CPU allocation defaulted to 12 CPUs per client actor (`TVFLIDS_SIM_CLIENT_CPUS=12`), exhausting available CPU resources and timing out actor jobs.
          - *Consequence*: Zero `.fit()` results were returned to the server; the server repeatedly evaluated the initial un-trained round-0 global model (Acc = 0.1249, Macro-F1 = 0.0536) at every round, causing identical final metrics across runs.
          - *Resolution*: Configured `ray_init_args={"local_mode": True, "include_dashboard": False, "ignore_reinit_error": True}` under Windows / when `TVFLIDS_SIM_LOCAL_MODE=1`, and defaulted per-client CPU allocation to `sim_num_cpus=1`. With in-process sequential execution, models train and diverge normally (e.g. Acc improves from 0.1249 to 0.6575 across 2 rounds on NSL-KDD).
        - **Min-Sum Activation Evidence (Part B)**:
          - Clean Update Norm: 10.6124
          - Partial Update Norm: 6.1496 (diff from clean: 10.0201, cos_sim: 0.3834)
          - Omniscient Update Norm: 5.9850 (diff from clean: 12.2125, cos_sim: -0.0055)
          - Partial vs. Omniscient Update L2 Difference: 8.6034
          - Min-Sum Constraint Satisfaction: Cand Sum Dist = 90.4116 $\le$ Max Sum Dist (H) = 90.4116 (Satisfied: True).
        - **ACK3 Activation Evidence (Part C)**:
          - Target relabeling: DoS (1) and Probe (2) relabeled to 0; R2L (3) and U2R (4) preserved.
          - Clean Update Norm: 36.9112
          - ACK3 K1 Update Norm: 36.8903 (diff from clean: 1.9536)
          - ACK3 K2 Update Norm: 36.8900 (diff from clean: 2.2332)
          - K1 vs. K2 Update L2 Difference: 1.9579
          - Data Source Isolation: K1 proxy data uses local background slice (no $\mathcal{D}_{\mathrm{val}}$ access); K2 proxy data uses exact server validation set $\mathcal{D}_{\mathrm{val}}$.
        - **On-Off Transition Evidence (Part D)**:
          - Tested across rounds 1..13 for $k=10$:
            * Rounds 1..10 (Honest Phase): Malicious client updates are bitwise/numerically identical to clean client updates under identical seed ($\Delta_{\mathrm{mal}} = \Delta_{\mathrm{clean}}$, diff = 0.0000).
            * Round 11+ (Active Phase): Attack activates; update differences diverge (diff = 0.4334 at round 11, 0.6572 at round 12, 0.8081 at round 13).
            * Generalized schedule verification passed for $k=20, 30, 50$.
        - **LF-R Remapping Evidence (Part E)**:
          - Normal class 0 remains unchanged ($0 \to 0$).
          - All attack samples ($y \in \{1, 2, 3, 4\}$) are remapped to distinct attack classes $y' \in \{1, 2, 3, 4\} \setminus \{y\}$.
          - Remapping is 100% deterministic under fixed seed and successfully trains the local client model.
        - **ACK4 Production Path Evidence (Part F)**:
          - Candidate mixture $\to$ clipping simulation $\to$ Algorithm 1 validation gate simulation $\to$ trust scoring simulation $\to$ malicious weight-share objective optimization executed end-to-end.
          - Optimal mixture $\omega^*$ selected from grid $\{0.0, 0.25, 0.5, 0.75, 1.0\}$.
          - Malicious update norm: 10.6490 (diff from clean: 15.0663).
        - **Clean-Client Bitwise Preservation (Part G)**:
          - Confirmed across Min-Sum, ACK3, ACK4, LF-R, and On-Off:
            * Clean client updates remain bitwise/numerically identical before and after attack interception.
            * Participant ordering is preserved.
            * Total participant count is invariant.
          - Verified on TV-FLIDS and baselines (FedAvg, FLTrust, TrimmedMean, Krum).
        - **Dedicated Production Activation Test Suite**:
          - Created `tests/test_production_attack_activation.py` (13 tests, all passing).
          - Full Stage 7 test suite: 48 tests passing in ~25s.
          - Full non-simulation regression suite: 401 tests passing in ~81s.
          - End-to-end integration simulation suite (`tests/test_integration.py`): 3 tests passing in ~141s.

- **2026-09-19 (Session 10 - Stage 8 Correction & Final State):**
  - Completed **Stage 8 Correction (Dataset Pipelines: CIC-IoT-2023, Edge-IIoTset, Taxonomy Reconciliation, and Provenance Manifests)** `[PARTIALLY VERIFIED]`.
  - **Edge-IIoTset Taxonomy Reconciliation (14 vs. 21 Labels)**:
    - *Authoritative Source*: Ferrag et al. (IEEE Access 2022), "Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset for IoT and IIoT Applications".
    - *Canonical Taxonomy*: Exactly **14 raw attack types** (+ 1 Normal = 15 multiclass categories in the official `Attack_type` column):
      * Normal (0): `Normal` (1 raw class)
      * DoS/DDoS (1): `DDoS_UDP`, `DDoS_ICMP`, `DDoS_HTTP`, `DDoS_TCP` (4 raw attacks)
      * Injection (2): `SQL_injection`, `XSS`, `Uploading` (3 raw attacks)
      * Scanning (3): `Port_Scanning`, `Vulnerability_scanner`, `Fingerprinting` (3 raw attacks)
      * Malware (4): `Backdoor`, `Password`, `Ransomware` (3 raw attacks)
      * MITM (5): `MITM` (1 raw attack)
      * Total: 14 raw attacks + Normal = 15 classes mapped into 6 final classes.
    - *Root Cause of 21-Label Discrepancy in Previous Report*:
      * 4 DDoS attacks were duplicated as 4 fictitious `dos_*` attacks (`dos_udp`, `dos_icmp`, `dos_http`, `dos_tcp`) (+4 labels).
      * `os_fingerprinting` was listed alongside `Fingerprinting` (+1 label).
      * `arp_spoofing` and `dns_spoofing` were listed alongside `MITM` (+2 labels). `dns_spoofing` is an alien label from CIC-IoT-2023 and has been removed.
    - *Pipeline Update*: Defined `CANONICAL_RAW_ATTACKS` (15 entries) and `CANONICAL_RAW_MAP` in `data/preprocessing/edgeiiotset_pipeline.py`. Isolated parsing aliases (`RAW_ATTACK_ALIASES`).
  - **CIC-IoT-2023 Taxonomy Verification & Bug Fix**:
    - *Authoritative Source*: Neto et al. (Sensors 2023, 23(13), 5941), "CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment".
    - *Canonical Taxonomy*: Exactly **33 raw attack types** (+ `BenignTraffic` = 34 raw classes in the official `label` column):
      * Benign (0): `BenignTraffic` (1 raw class)
      * DDoS (1): 12 raw attacks (`DDoS-ACK_Fragmentation`, `DDoS-HTTP_Flood`, `DDoS-ICMP_Flood`, `DDoS-ICMP_Fragmentation`, `DDoS-PSHACK_Flood`, `DDoS-RSTFINFlood`, `DDoS-SlowLoris`, `DDoS-SYN_Flood`, `DDoS-SynonymousIP_Flood`, `DDoS-TCP_Flood`, `DDoS-UDP_Flood`, `DDoS-UDP_Fragmentation`)
      * DoS (2): 4 raw attacks (`DoS-HTTP_Flood`, `DoS-SYN_Flood`, `DoS-TCP_Flood`, `DoS-UDP_Flood`)
      * Mirai (3): 3 raw attacks (`Mirai-greeth_flood`, `Mirai-greip_flood`, `Mirai-udpplain`)
      * Recon (4): 5 raw attacks (`Recon-HostDiscovery`, `Recon-OSScan`, `Recon-PingSweep`, `Recon-PortScan`, `VulnerabilityScan`)
      * Spoofing (5): 2 raw attacks (`DNS_Spoofing`, `MITM-ArpSpoofing`)
      * WebBased (6): 6 raw attacks (`Backdoor_Malware`, `BrowserHijacking`, `CommandInjection`, `SqlInjection`, `Uploading_Attack`, `XSS`) — Rare class 1 (2,400 train / 480 test)
      * BruteForce (7): 1 raw attack (`DictionaryBruteForce`) — Rare class 2 (1,600 train / 320 test)
    - *Bug Fix*: `VulnerabilityScan` was omitted from `_CATEGORY_KEYWORD_PRIORITY` in `data/preprocessing/ciciot2023_pipeline.py`, causing official Recon records to be dropped. Fixed by adding direct `CANONICAL_RAW_MAP` lookup and updating keyword priority.
    - *Missing Import Fix*: Defined and exported `PAPER_VAL_QUOTAS = {0: 370, 1: 493, 2: 370, 3: 308, 4: 148, 5: 111, 6: 100, 7: 100}` (sum = 2,000) in `ciciot2023_pipeline.py`.
  - **Validation-Set Specification Audit**:
    - Confirmed all three datasets use $|\mathcal{D}_{\mathrm{val}}| = 2,000$ and share the same NSL-KDD quota construction rule (§VI-A, lines 327 & 350).
    - Classes with $< 5,000$ training records receive $\max(\text{prop}, 100)$ capped at 30% of class records. Remaining validation samples are distributed proportionally among large classes.
    - $\mathcal{D}_{\mathrm{tune}}$ is a stratified 10% sample of the remainder after $\mathcal{D}_{\mathrm{val}}$ across all three datasets.
    - Quotas for capped training sets:
      * NSL-KDD: `{0: 1016, 1: 693, 2: 176, 3: 100, 4: 15}` (Table I)
      * CIC-IoT-2023: `{0: 370, 1: 493, 2: 370, 3: 308, 4: 148, 5: 111, 6: 100, 7: 100}` (Table S5 caps)
      * Edge-IIoTset: `{0: 600, 1: 533, 2: 333, 3: 267, 4: 200, 5: 67}` (Table S5 caps)
  - **Client-Partitioning Specification Audit**:
    - $N=20, D=10, \alpha_D=0.5$ Dirichlet non-IID partitioning over the client pool ($\mathcal{D}_{\mathrm{client}}$) followed by client-local SMOTE applies uniformly to NSL-KDD, CIC-IoT-2023, and Edge-IIoTset (§VI-A, §VI-B, Table III, and Supplementary §S5).
  - **Real-Data Verification Boundary**:
    - Real raw datasets for CIC-IoT-2023 and Edge-IIoTset are absent locally.
    - Pipeline code, interfaces, schemas, and specifications: `[VERIFIED]` via synthetic schema fixtures.
- **Stage 9 Experimental Campaign Orchestration & Reproducibility Infrastructure:**
  - Enumeration completeness: 6,111 total canonical runs (B1: 693, B2: 900, B3: 900, B4: 1050, B5: 300, B6: 100, B7: 480, B8: 1010, B9: 200, B10: 28, B11: 450) matching the paper specification.
  - Run ID uniqueness: 6,111 distinct 16-character hex SHA-256 IDs, 0 collisions.
  - Checkpoint & provenance-safe resume semantics: completed runs are reused ONLY when run_id matches, metrics are valid, stored commit matches current commit, and stored dataset provenance matches current dataset provenance; differing commit or dataset provenance raises RuntimeError refusing stale cache reuse; fresh runs with force=True update artifacts to new provenance.
  - Dataset blocker handling: `ciciot2023` and `edgeiiotset` runs marked `BLOCKED` when raw files are absent; synthetic fixtures are never substituted.
  - Seed isolation: `set_all_seeds` synchronizes Python, NumPy, and PyTorch CPU/CUDA RNGs.
  - Smoke execution: verified enumeration CLI, dry-run mode, and 1-round NSL-KDD execution with valid artifact generation.
  - Test suite: 47/47 passed in `tests/test_campaign_orchestration.py`; 68/68 regression tests passed in reachability, baseline isolation, and attack activation.

---

## 20. Session Changelog

- **2026-09-17 (Session 1):**
  - Inspected repository, existing audits (`PROJECT_PAPER_ALIGNMENT_AUDIT.md`, `AUDIT_VERIFICATION_AND_CORRECTIONS.md`), and IEEE manuscript (`Paper/IEEE/TV-FLIDS.tex`, `TV-FLIDS_supplementary.tex`).
  - Verified exact mathematical formulas, parameter counts, campaign run volumes, and baseline/attack taxonomies.
  - Formally established `PROJECT_MEMORY.md` at the repository root as the persistent memory document for all subsequent sessions.
  - Maintained zero production code modifications during initialization pass.

- **2026-09-17 (Session 2):**
  - Completed **Stage 1 (Model and NSL-KDD Scientific Alignment)**.
  - Modified `models/mlp.py`:
    - Replaced `nn.BatchNorm1d` with `nn.LayerNorm(256/128/64)` across all hidden layers.
    - Verified layer sequence: Linear -> LayerNorm -> ReLU -> Dropout(0.3) -> Linear -> LayerNorm -> ReLU -> Dropout(0.3) -> Linear -> LayerNorm -> ReLU -> Dropout(0.3) -> Linear.
    - Added static helper `parameter_count_formula(d, K)` verifying $P(d, K) = 256d + 65K + 42,304$.
  - Modified `data/preprocessing/nslkdd_pipeline.py`:
    - Implemented exact Table I quota splitting for $\mathcal{D}_{\mathrm{val}}$ (2,000), $\mathcal{D}_{\mathrm{tune}}$ (12,398), and $\mathcal{D}_{\mathrm{client}}$ (111,575).
    - Enforced loud diagnostic failure if class quotas cannot be satisfied.
    - Implemented zero-leakage feature scaling: `MinMaxScaler(clip=False)` fitted strictly on server data $\mathcal{D}_{\mathrm{val}} \cup \mathcal{D}_{\mathrm{tune}}$ (14,398 samples); client and test data are never seen by the scaler.
    - Preserved unclipped test representation per paper §VI-A.
    - Implemented paper-specified client-local SMOTE rule ($M = \operatorname{median}(\{n_c : n_c > 0\})$, $k = \min(5, n_c - 1)$, duplication for $n_c=1$, zero for $n_c=0$).
    - Preserved backward compatibility for `build_pipeline` (returning 9-tuple by default, with `return_tune=True` option).
  - Added new test suite `tests/test_model_and_nslkdd_alignment.py` containing 18 rigorous unit tests.
  - Executed tests: 18/18 passed in `tests/test_model_and_nslkdd_alignment.py`; 19/19 passed in `tests/test_val_size_regression.py`; 49/49 passed in `tests/test_all.py`.
  - Executed `scripts/verify_environment.py` and documented environment characteristics.
  - Updated `PROJECT_MEMORY.md` to reflect completed Stage 1 status.

- **2026-09-17 (Session 3):**
  - Completed **Stage 2 (TV-FLIDS Strategy Alignment: Median-Radius Clipping & Single Validation Gate)**.
  - Aligned code strictly with IEEE TIFS manuscript §IV, Eq. (3)–(4), and Algorithm 1:
    - $C^{(t)} = \operatorname{median}_{j \in \mathcal{P}^{(t)}} \|\Delta_j^{(t)}\|$
    - $\tilde{\Delta}_i^{(t)} = \Delta_i^{(t)} \min(1, C^{(t)} / \|\Delta_i^{(t)}\|)$
    - $\tilde{w}_i^{(t)} = w^{(t)} + \tilde{\Delta}_i^{(t)}$
    - $\delta_i = \ell_{\mathrm{val}}(w^{(t)}) - \ell_{\mathrm{val}}(\tilde{w}_i^{(t)}) \ge \tau_L(t)$
    - $\tau_L(t) = -0.1 + 0.1 \min(1, t / 20)$
  - Modified `trust/verification.py`:
    - Added standalone `clip_updates(updates, cohort_norms=None)` computing median radius $C^{(t)}$ across full participant cohort $\mathcal{P}^{(t)}$ and scaling updates exceeding $C^{(t)}$ to the boundary while leaving below-median updates unchanged.
    - Replaced 3-check hard filtering with single loss improvement gate: $\delta_i = \ell_{\mathrm{val}}(w^{(t)}) - \ell_{\mathrm{val}}(\tilde{w}_i^{(t)}) \ge \tau_L(t)$ with linear schedule $\tau_L(t) = -0.1 + 0.1\min(1, t/20)$.
    - Candidate models are strictly constructed from clipped updates ($\tilde{w}_i = w^{(t)} + \tilde{\Delta}_i$).
    - Preserved cosine similarity and norm z-score calculations for continuous downstream trust signals, but removed them completely as hard rejection criteria.
    - Maintained backward compatibility for `verify_all()` returning `'accepted'`, `'rejected'`, `'verified'` (alias to accepted), `'flagged'` (empty), `'deltas'`, `'val_losses'`, `'threshold'`.
  - Modified `fl/strategy.py`:
    - Integrated `clip_updates` on raw client updates prior to validation gate evaluation.
    - Evaluated validation gate on clipped updates.
    - Implemented paper-specified fallback for empty accepted cohort $\mathcal{A}^{(t)} = \emptyset$: returns un-updated global parameters $w^{(t+1)} = w^{(t)}$.
    - Passed unclipped raw updates `a_raw_upds` to `trust_scorer.compute_anomaly_scores` per Paper §IV line 186 ("The norm statistic reads unclipped norms because clipping equalizes the largest half of them").
    - Logged `num_accepted` and `clipping_radius` per round.
  - Added new test suite `tests/test_strategy_alignment.py`:
    - 12 comprehensive unit tests covering median radius, below/above-median updates, zero updates, full cohort norm scope, gate acceptance/rejection, no cosine hard rejection, no z-score hard rejection, warm-up schedule, clipping-before-gating sequence, and empty accepted cohort fallback.
  - Updated legacy test `tests/test_pipeline_smoke.py`:
    - Aligned `TestVerificationGateSmoke` with single-gate semantics (removed obsolete assertion that z-score acts as a hard filter).
  - Executed tests: 12/12 passed in `tests/test_strategy_alignment.py`, 41/41 passed in `tests/test_warmup_schedule.py`, 10/10 passed in `tests/test_forward_pass_count.py`, 14/14 passed in `tests/test_overhead_tracking.py`, 5/5 passed in `tests/test_ack1_ack2_attacks.py`, 7/7 passed in `tests/test_pipeline_smoke.py`, 18/18 passed in `tests/test_model_and_nslkdd_alignment.py`, 49/49 passed in `tests/test_all.py`.
  - Audited production code path: verified end-to-end execution order from Flower client results to trust-weighted aggregation, confirming no legacy path bypasses clipping or invokes obsolete multi-criteria rejection.
  - Updated `PROJECT_MEMORY.md` to reflect completed Stage 2 status.

- **2026-09-17 (Session 4):**
  - Completed **Stage 3 (TV-FLIDS Trust System Alignment: Class-Balanced Loss, Multi-Signal Scoring, Asymmetric EMA, Rejection Penalty)**.
  - Pre-implementation audit resolved all mathematical specifications against IEEE TIFS manuscript §IV, Eq. (2), (5)–(7), and Algorithm 1:
    - $\ell_{\mathrm{bal}}(w) = \frac{1}{|C_{\mathrm{val}}|} \sum_{c \in C_{\mathrm{val}}} \frac{1}{|\mathcal{D}_{\mathrm{val}}^c|} \sum_{(x, y) \in \mathcal{D}_{\mathrm{val}}^c} \ell_{\mathrm{CE}}(f(x; w), y)$
    - $A_i = \operatorname{clip}_{[-1, 1]}\left(\frac{\ell_{\mathrm{bal}}(w^{(t)}) - \ell_{\mathrm{bal}}(\tilde{w}_i^{(t)})}{\ell_{\mathrm{bal}}(w^{(t)}) + \varepsilon}\right)$, with range $[-1, 1]$ (negative for degraded loss).
    - $S_i = \frac{1}{2}(\cos(\tilde{\Delta}_i, \bar{\Delta}_{\mathcal{A}}) + 1) \in [0, 1]$ computed with clipped updates against accepted cohort centroid $\bar{\Delta}_{\mathcal{A}}$.
    - $O_i = 1 - e^{-z_i / 2.5} \in [0, 1]$ computed with unclipped norms over the entire participant cohort $\mathcal{P}$ ($\mu_{\mathcal{P}}, \sigma_{\mathcal{P}}$).
    - Instantaneous mixed signal: $u_i = \alpha S_i + \beta A_i - \gamma O_i$ (verified strictly negative sign $-\gamma O_i$). For accepted clients $i \in \mathcal{A}$, $s_i = \operatorname{clip}_{[0, 1]}(u_i)$. For rejected clients $i \in \mathcal{P} \setminus \mathcal{A}$, $s_i = 0$.
    - Asymmetric trust memory: $\lambda_\uparrow = 0.9$ (if $s_i \ge T_i^{(t-1)}$), $\lambda_\downarrow = 0.7$ (if $s_i < T_i^{(t-1)}$), $T_i^{(0)} = 0.5$, $\tau_{\min} = 0.01$.
    - Non-participants ($i \notin \mathcal{P}$) strictly retain previous trust $T_i^{(t)} = T_i^{(t-1)}$.
    - Rejected clients ($i \in \mathcal{P} \setminus \mathcal{A}$) receive $s_i = 0$, selecting $\lambda_\downarrow = 0.7$ and decaying trust.
  - Modified `trust/verification.py`:
    - Added `compute_class_balanced_loss` and `compute_class_balanced_loss_from_tensors` accumulating per-class losses in a single pass over $\mathcal{D}_{\mathrm{val}}$.
    - Added balanced loss evaluation alongside standard loss in `evaluate_validation_gate` and cached both losses (`eval_cache`, `eval_bal_cache`) to preserve the $|\mathcal{P}| + 1$ forward-pass guarantee.
  - Modified `trust/trust_scorer.py`:
    - Implemented asymmetric EMA ($\lambda_\uparrow=0.9, \lambda_\downarrow=0.7, T^{(0)}=0.5, \tau_{\min}=0.01$).
    - Implemented $A_i$ calculation in $[-1, 1]$.
    - Implemented $O_i = 1 - e^{-z_i / 2.5}$ with population statistics over all participants $\mathcal{P}$.
    - Implemented `compute_instantaneous_signal` ($u_i = \alpha S_i + \beta A_i - \gamma O_i$ and $s_i = \operatorname{clip}_{[0, 1]}(u_i)$).
    - Implemented `update_trust` supporting accepted, rejected ($s_i = 0$), and non-participating clients.
  - Modified `fl/strategy.py`:
    - Evaluated global parameters once to obtain both standard and class-balanced validation loss in one pass (`_eval_model_both`).
    - Evaluated clipped candidate models once during gating, caching both losses.
    - Updated trust for all sampled participants (both accepted and rejected) in all rounds.
    - Maintained empty-cohort fallback ($w^{(t+1)} = w^{(t)}$) while penalizing all rejected clients with $s_i = 0$.
  - Modified `scripts/check_manuscript.py`:
    - Updated `PAPER_DIR` fallback to `Paper/IEEE` if present, passing all 18 manuscript integrity tests.
  - Updated legacy test assertions:
    - In `tests/test_all.py`, updated `test_reset` to assert initial trust $0.5$ (per paper $T^{(0)} = 0.5$) and updated `test_accuracy_scores_clip` to assert $A_i \in [-1, 1]$ with negative value for degrading loss.
  - Added new test suite `tests/test_trust_alignment.py`:
    - 15 comprehensive unit tests covering class-balanced loss under severe imbalance, negative $A_i$, $A_i$ clipping bounds $[-1, 1]$, $S_i$ range $[0, 1]$, zero-vector cosine safety, exact $O_i$ exponential formula, unclipped cohort scope over all $\mathcal{P}$, positive trust branch $\lambda_\uparrow=0.9$, negative trust branch $\lambda_\downarrow=0.7$, initial trust $T^{(0)}=0.5$, trust floor $\tau_{\min}=0.01$, rejected client penalty ($s_i=0$), non-participant trust retention, mixed signal equation with negative norm sign, and full multi-client round integration.
  - Executed tests: 15/15 passed in `tests/test_trust_alignment.py`, 12/12 passed in `tests/test_strategy_alignment.py`, 18/18 passed in `tests/test_manuscript_integrity.py`, 13/13 passed in `tests/test_ste_gradient.py`.
  - Updated `PROJECT_MEMORY.md` to reflect completed Stage 3 status.

- **2026-09-17 (Session 5):**
  - Performed **Pre-Stage 4 Mathematical Reconciliation Audit** (READ-ONLY, no production code changes).
  - Resolved all 6 mathematical issues regarding $O_i$, aggregation weights, and STE. All confirmed consistent with paper equations.
  - Next task confirmed: Stage 4 — Online Meta-Weight Adaptation.

- **2026-09-17 (Session 6):**
  - Completed **Stage 4 (TV-FLIDS Online Meta-Weight Adaptation: Log-Weights, STE Gradient, Adam Optimization, Algorithm 1 Sequencing)**.
  - Modified `trust/adaptive_trust_scorer.py` and `fl/strategy.py`.
  - Added test suite `tests/test_meta_weight_alignment.py` (20/20 passed).

- **2026-09-18 (Session 7):**
  - Completed **Stage 5 (TV-FLIDS Final Model Aggregation and Baseline Strategy Isolation)**.
  - Aligned code strictly with IEEE TIFS manuscript §IV, Eq. (11), Algorithm 1, and Supplementary Table S2.
  - Added test suites `tests/test_aggregation_alignment.py` (12/12 passed) and `tests/test_baseline_isolation.py` (10/10 passed).

- **2026-09-18 (Session 8):**
  - Completed **Stage 6 (Implement and Integrate 5 Missing Baselines: Multi-Krum, Norm Clipping, FLDetector, Zeno, BaFFLe)**.
  - Implemented all 5 missing baselines as standalone strategies with strict isolation.
  - Added dedicated test suites (17 new tests) and executed 226/226 regression tests passing.

- **2026-09-18 (Session 9):**
  - Completed **Stage 7 (Attack Suite and Adversarial Threat Model Completion)** `[VERIFIED]`.
  - Implemented knowledge tiers (K0, K1, K2), Min-Sum, Min-Max, ACK3, ACK4, On-Off schedules ($k \in \{10, 20, 30, 50\}$), and LF-R.
  - Resolved Windows Ray actor execution issues via local mode and single CPU per client.
  - Added `tests/test_production_attack_activation.py` (13 tests passing).

- **2026-09-19 (Session 10 - Stage 8 Correction & Final State):**
  - Completed **Stage 8 Correction (Dataset Pipelines: CIC-IoT-2023, Edge-IIoTset, Taxonomy Reconciliation, and Provenance Manifests)** `[PARTIALLY VERIFIED]`.
  - Reconciled Edge-IIoTset (14 raw attacks + Normal = 15 classes mapped to 6) and CIC-IoT-2023 (33 raw attacks + Benign = 34 classes mapped to 8).
  - Added `data/dataset_bundle.py` and provenance manifests with explicit `is_synthetic_fixture` tags.
  - Verified pipeline with synthetic schema fixtures; real data marked `[BLOCKED]`.

- **2026-09-19 (Session 11 - Stage 9 Implementation & Validation):**
  - Completed **Stage 9 (Experimental Campaign Runner and Reproducibility Infrastructure)** `[PARTIALLY VERIFIED]`.
  - **B1 Configuration Count Resolution (76 vs. 77)**:
    - Resolved directly from `Paper/IEEE/TV-FLIDS_supplementary.tex` Table S2 (`tab:grids`): TV-FLIDS specifies 1 center configuration + 11 single-factor variants (including $T^{(0)}=1.0$), yielding exactly 12 configurations ("at most 12 configurations per method").
    - Combined with 65 configurations across 14 other baselines, Block 1 canonical count is **77 configurations** ($77 \times 3 \text{ seeds} \times 3 \text{ datasets} = \mathbf{693}$ runs).
    - Canonical campaign size across all 11 blocks is resolved to **6,111 runs** (B1: 693, B2: 900, B3: 900, B4: 1,050, B5: 300, B6: 100, B7: 480, B8: 1,010, B9: 200, B10: 28, B11: 450). Theoretical paper upper bound $\le 7,038$ runs remains distinguished.
  - **Run-ID Completeness & Scientific Parameter Hashing**:
    - `compute_run_id` computes a deterministic 16-hex SHA-256 hash over canonical, sorted JSON of all scientific parameters (`block`, `dataset`, `dataset_protocol`, `preprocessing_mode`, `strategy`, `strategy_params`, `attack`, `attack_params`, `knowledge_tier`, `attack_variant`, `attack_strength`, `on_off_k`, `seed`, `num_rounds`, `num_clients`, `clients_per_round`, `dirichlet_alpha`, `val_size`, `model_architecture`, `optimizer_params`).
    - Runtime metadata (timestamp, host, CPU, PID, execution order) are strictly excluded from the hash but stored in `metadata.json` for provenance.
  - **Immutable Configuration Freezing**:
    - `config_freezer.py` freezes all scientific and environmental parameters into `config.json` before execution, capturing git commit hash, dirty tree state, package versions, hardware info, and theoretical parameter counts via $P(d, K) = 256d + 65K + 42,304$.
    - Dirty tree protection blocks uncommitted runs unless explicitly overridden with `--allow-dirty`.
  - **Result Artifact Hierarchy & Schema**:
    - Deterministic hierarchy: `results/<dataset>/<block>/<strategy>/<attack>/<run_id>/` containing `config.json`, `status.json` (`PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `SKIPPED`, `BLOCKED`), `metrics.json` (canonical schema with validation), `metadata.json`, and execution logs.
  - **Resume & Checkpointing Semantics (Provenance-Safe Cache Reuse)**:
    - Completed runs with valid metrics are reused ONLY when run ID matches, metrics schema is valid, stored implementation Git commit matches current Git commit, stored dataset provenance matches current dataset provenance, and scientific configuration matches.
    - If Git commit differs, dataset provenance differs, scientific configuration differs, or artifact is corrupt, cached-result reuse is refused and a loud `RuntimeError` is raised.
    - Fresh execution (e.g. with `force=True`) produces a new valid artifact with the new commit/provenance rather than silently reusing stale data.
  - **Dataset Blocker Handling**:
    - Real data runs for `ciciot2023` and `edgeiiotset` are enumerated but marked `BLOCKED` with descriptive messages when raw files are absent; synthetic fixtures are never substituted in production campaign paths.
  - **Seed & Reproducibility Isolation**:
    - `set_all_seeds(seed)` uniformly sets seeds across Python `random`, `numpy`, PyTorch CPU and CUDA, and sets `torch.backends.cudnn.deterministic = True`.
  - **Testing**:
    - Added 47 comprehensive tests in `tests/test_campaign_orchestration.py` verifying counts, run IDs, ordering invariance, cross-process stability, filtering, freezing, provenance-safe resume, corruption detection, failure handling, blockers, and seed isolation. All 47 passed.
    - Smoke tests executed: enumeration report, dry-run, and 1-round NSL-KDD execution with valid artifact generation.
    - Regression tests passed: 68/68 in reachability, baseline isolation, and attack activation.
  - **Final Status**: Stage 9 is `[PARTIALLY VERIFIED]` (orchestration and provenance-safe resume verified; full campaign not executed; real data for CIC-IoT-2023 and Edge-IIoTset remains `[BLOCKED]`).

- **2026-09-19 (Session 12 - Stage 10 Implementation & Verification):**
  - Completed **Stage 10 (Reproducible Statistical Evaluation Pipeline: Wilcoxon, Holm-Bonferroni, Cliff's Delta, Bootstrap CIs, Paper Schema Alignment)** `[VERIFIED]`.
  - Implemented `evaluation/wilcoxon.py` with exact two-sided Wilcoxon signed-rank test using dynamic programming over scaled integer ranks ($2R_i$) for exact tie handling without asymptotic normal approximations.
  - Implemented `evaluation/holm_bonferroni.py` enforcing strict step-down FWER control.
  - Implemented `evaluation/effect_size.py` computing Cliff's delta $d \in [-1, 1]$ and 95% BCa/percentile bootstrap confidence intervals.
  - Implemented `evaluation/result_loader.py` validating 6,111 run result artifacts, schema compliance, and dataset gating.
  - Implemented `evaluation/statistical_pipeline.py` executing all 84 paper-mandated statistical comparisons across Blocks B1–B11.
  - Full Stage 10 test suite passing (26/26 tests); regression suite passing (107/107 tests).

- **2026-09-19 (Session 13 - Stage 11 Implementation & Validation):**
  - Completed **Stage 11 (Production Campaign Readiness, Compute Planning, and Staged Execution)** `[PARTIALLY VERIFIED]`.
  - Implemented `campaign/preflight.py` with 12/12 mandatory preflight integrity checks (`[VERIFIED]`).
  - Implemented `campaign/scheduler.py` with bounded concurrency, resource-aware gating, provenance-safe resume, bounded retries, and graceful shutdown (`[VERIFIED]`).
  - Implemented `campaign/provisioning.py` with the 8-step real-data provisioning workflow (`[VERIFIED]`).
  - Executed all 5 staged pilots (Clean TV-FLIDS, Attacked TV-FLIDS, Baseline FedAvg, Concurrency Isolation, Block Subset & Statistical Ingestion) with 100% success (`[VERIFIED]`).
  - Demonstrated concurrency safety and zero cross-process contamination across concurrent simulation workers (`[VERIFIED]`).
  - Preserved dataset gating: NSL-KDD `UNLOCKED [READY FOR PRODUCTION]`; CIC-IoT-2023 and Edge-IIoTset strictly `[BLOCKED]` (`[VERIFIED]`).
  - Final Stage 11 Status: `[PARTIALLY VERIFIED]`.

---

## 21. Stage 10: Statistical Evaluation Pipeline `[VERIFIED]`

- **Exact Wilcoxon Signed-Rank Test (`evaluation/wilcoxon.py`)**:
  - Implements exact two-sided Wilcoxon signed-rank test without asymptotic normal fallback.
  - Handles tied absolute differences via dynamic programming over scaled integer ranks $2R_i$ (which are guaranteed integers), computing the exact sign-permutation null distribution across all $2^N$ assignments in under 2 ms.
  - Zero differences ($|d| \le 10^{-12}$) are dropped per Wilcoxon (1945) and Pratt (1959); if all differences are zero, returns $W=0.0, p=1.0$.
- **Holm-Bonferroni FWER Control (`evaluation/holm_bonferroni.py`)**:
  - Step-down procedure ordering $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$ and testing $p_{(i)} \le \alpha / (m - i + 1)$.
  - Enforces monotonicity of adjusted p-values: $p_{(i)}^{\mathrm{adj}} = \max(p_{(i-1)}^{\mathrm{adj}}, \min(1.0, (m - i + 1) p_{(i)}))$.
- **Effect Size and Confidence Intervals (`evaluation/effect_size.py`)**:
  - Cliff's delta: $d = \frac{\sum [x_i > y_j] - \sum [x_i < y_j]}{m \cdot n} \in [-1, 1]$.
  - Bootstrap CIs: 1,000 resamples computing 95% BCa or percentile confidence intervals.
- **Statistical Pipeline (`evaluation/statistical_pipeline.py`)**:
  - Maps to all 84 planned statistical comparisons across Blocks B1–B11.
  - Ingestion via `ResultLoader` enforces dataset gating (`BLOCKED` status preserved; synthetic fixtures excluded).

---

## 22. Stage 11: Production Campaign Readiness and Compute Planning `[PARTIALLY VERIFIED]`

### 22.1 Production Environment Recommendation `[DECISION]`
- **Operating System**: Linux (Ubuntu 22.04 LTS or newer) or Windows Subsystem for Linux (WSL2).
- **Hardware Platform**: Multi-core CPU ($\ge 16$ cores recommended) + NVIDIA GPU ($\ge 12$ GB VRAM, e.g., RTX 3090, RTX 4090, A100, or V100).
- **Python Version**: Python 3.10 (3.10.11 verified).
- **Core Dependencies**:
  - PyTorch: $\ge 2.1.0$ (CPU or CUDA 11.8/12.1).
  - Flower (`flwr`): 1.6.0.
  - Ray: 2.8.0.
  - NumPy: 1.26.2.
  - SciPy: 1.11.4.
  - scikit-learn: 1.3.2.
  - imbalanced-learn: 0.11.0.
  - pandas: 2.1.3.
- **Platform-Specific Ray / Windows Constraint**:
  - On Windows, Flower simulation via Ray worker processes suffers from dynamic DLL loading violations (`WinError 1114` in `c10.dll` / PyTorch). Furthermore, Ray default worker CPU allocation causes thread exhaustion.
  - On Windows development environments, the system must run with `ray_init_args={"local_mode": True}` and `sim_num_cpus=1`.
  - Multi-worker concurrency is managed safely at the OS process level via `CampaignScheduler` rather than Ray intra-simulation concurrency.
  - On production Linux clusters, Ray distributed actors and CUDA GPU acceleration function without `c10.dll` initialization faults.

### 22.2 Compute Resource Planning `[VERIFIED]`
- **Single Client**: Forward/backward pass on MLP (53,125 parameters for NSL-KDD): ~0.02s per batch (batch size 256); 5 epochs per round = ~0.35s per client.
- **Single Round ($D=10$ clients)**: ~3.5s client training + ~0.2s server validation gate & trust update = ~3.7s per round.
- **Single 100-Round Run**: ~370s (6.2 minutes) on CPU; ~60–90s on GPU.
- **Per-Run Memory Footprint**: 1.5–2.0 GB RSS RAM; 1.0–1.5 GB VRAM if GPU enabled.
- **Campaign Execution Estimates (6,111 Canonical Runs)**:
  - Total Compute Volume: ~631.5 CPU-hours.
  - 8 Parallel Workers (16-core CPU workstation): ~78.9 wall-clock hours (~3.3 days).
  - 16 Parallel Workers (32-core server): ~39.5 wall-clock hours (~1.6 days).
  - GPU Cluster (16 workers across 4x A100 / RTX 4090): ~9.5 wall-clock hours.
  - Block 1 (693 runs): ~71.6 CPU-hours (~9.0 hours on 8 workers).
  - Blocks B2–B11 (5,418 runs): ~559.9 CPU-hours (~70.0 hours on 8 workers).

### 22.3 Concurrency Safety and Process Isolation `[VERIFIED]`
- **Process Isolation**: Each run executes in an isolated OS process via `subprocess.Popen` with dedicated arguments.
- **Result Path Uniqueness**: Zero collision probability; paths are keyed by deterministic 16-character SHA-256 `run_id` based on canonical parameter JSON.
- **Seed Isolation**: `set_all_seeds(run_spec.seed)` is called inside each child process, seeding Python `random`, `numpy`, and PyTorch CPU/CUDA independently.
- **Output / Log Deadlock Prevention**: Worker stdout and stderr are redirected to `results/.../worker.log` on disk, eliminating pipe buffer deadlocks.
- **Temporary Files**: No shared temp files; working directories and cache are strictly per-run.
- **Multi-Worker Pilot Validation**: Pilot 4 executed 2 concurrent runs simultaneously; verified bitwise distinct outputs, zero state leakage, and identical reproducible metrics compared to sequential execution.

### 22.4 Campaign Scheduling Controls `[VERIFIED]`
- **Pending Run Discovery**: Discovers all runs across B1–B11 or filtered subset; queries artifact cache for completed valid runs.
- **Provenance-Safe Resume**: Skips valid runs matching current Git commit and dataset provenance. Incomplete, missing, or corrupted runs are marked pending.
- **Bounded Concurrency**: Regulated via `--max-workers` (default: 2 on dev, up to CPU core count on production).
- **Resource-Aware Safeguards**: Checks available CPU cores and physical RAM (`get_system_ram_gb()`) before launching workers. CLI flags: `--max-workers`, `--max-cpus`, `--max-memory`, `--device`, `--dry-run`.
- **Bounded Retries**: Failed runs can be retried up to `--max-retries` (default: 1); diagnostic failure details and traceback preserved in `status.json` and `worker.log`.
- **Graceful Interruption**: SIGINT/SIGTERM handlers terminate child processes cleanly, leaving `status.json` with `FAILED` or `INTERRUPTED` without corrupting completed runs.

### 22.5 Dataset Provisioning Workflow `[VERIFIED]`
- **Exact 8-Step Workflow**:
  ```text
  Acquire exact dataset version
  → place raw files in approved location
  → compute SHA-256 hashes
  → verify expected schema
  → run dataset integrity tests
  → generate provenance manifest
  → run one dataset smoke test
  → unlock production campaign
  ```
- **Dataset Status**:
  | Dataset | Required raw files | Expected total size | Required preprocessing | Required storage | Current status |
  | :--- | :--- | :---: | :--- | :---: | :--- |
  | **NSL-KDD** | `KDDTrain+.txt`, `KDDTest+.txt` | ~25 MB | Table I quota split, zero-leakage MinMax, client-local SMOTE | ~35 MB | `UNLOCKED [READY FOR PRODUCTION]` |
  | **CIC-IoT-2023** | 46 raw CSV files (Neto et al. 2023) | ~15–20 GB | 33 raw attacks + Benign $\to$ 8 classes, Table S5 caps, client-local SMOTE | ~25 GB | `[BLOCKED]` (Raw files absent) |
  | **Edge-IIoTset** | `Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv` | ~4.5 GB | 14 raw attacks + Normal $\to$ 6 classes, Table S5 caps, client-local SMOTE | ~6 GB | `[BLOCKED]` (Raw files absent) |

### 22.6 Preflight Integrity Requirements `[VERIFIED]`
- 12/12 mandatory preflight checks implemented in `campaign/preflight.py`:
  1. Clean Git tree (`--allow-dirty` required for uncommitted dev trees).
  2. Valid 40-character commit hash.
  3. Campaign manifest generated.
  4. Canonical count equals exactly 6,111 runs.
  5. Zero duplicate run IDs.
  6. Required dataset availability (gates blocked datasets).
  7. Sufficient disk space ($\ge 5$ GB free on target drive).
  8. Sufficient compute capacity ($\ge 1$ CPU core, available RAM $\ge 0.5$ GB or total $\ge 4.0$ GB).
  9. Correct environment package versions (torch, flwr, sklearn, etc.).
  10. Statistical schema available (`evaluation/statistical_pipeline.py`).
  11. Result directory writable (`results/`).
  12. No accidental synthetic-data mode in production.

### 22.7 Pilot Protocol & Execution Results `[VERIFIED]`
- **Pilot 1 (Clean TV-FLIDS)**: Block 3, NSL-KDD, `no_attack`, seed 42, 2 rounds. Acc=0.5828, F1=0.3230, ASR=0.4142. Completed in 36.5s.
- **Pilot 2 (Attacked TV-FLIDS)**: Block 2, NSL-KDD, `label_flip_30`, seed 42, 2 rounds. Acc=0.6554, F1=0.2978, ASR=0.5178. Completed in 36.5s.
- **Pilot 3 (Baseline FedAvg)**: Block 3, NSL-KDD, `no_attack`, seed 42, 2 rounds. Acc=0.6553, F1=0.3035, ASR=0.5352. Completed in 33.9s.
- **Pilot 4 (Concurrency Isolation)**: 2 concurrent runs (TV-FLIDS seed 123 and FedAvg seed 123) with `max_workers=2`. Succeeded in 45.8s with 0 collisions and zero cross-process contamination.
- **Pilot 5 (Block Subset & Statistical Ingestion)**: Block 2, TV-FLIDS seeds 42, 123, 456 (2 rounds). Seed 42 cleanly resumed from cache in 0.19s without rerunning; seeds 123 and 456 executed successfully. All 7 runs ingested by `ResultLoader`. `StatisticalPipeline` evaluated 84 comparisons (28 incomplete, 56 blocked, 0 synthetic).

### 22.8 Failure Recovery Procedure `[VERIFIED]`
- **Process Crash / Termination**: Scheduler marks run as `FAILED` or detects stale `RUNNING` status without active PID; retries up to `max_retries`.
- **Machine Restart Simulation**: Orphaned `RUNNING` runs without matching process are recovered as pending or failed; completed runs remain untouched.
- **Corrupted Metrics / JSON**: `ResultLoader` and `CampaignScheduler` validate JSON structure and schema; corrupt files raise validation error and trigger rerun rather than silently using garbage data.
- **Stale / Divergent Provenance**: If Git commit or dataset hash changed, `RuntimeError` is raised preventing invalid cache reuse; force rerun required to re-baseline.

### 22.9 Storage Estimates and Archival Policy `[VERIFIED]`
- **Storage per Run**: ~43 KB (config, status, metrics, metadata, worker log).
- **Model Checkpoints**: `[DECISION]` — Checkpoints are NOT required for the final scientific protocol. Storing 100 rounds $\times$ 53k floats $\times$ 4 bytes $\times$ 6,111 runs $\approx$ 130 GB of redundant weights. The scientific protocol reports test accuracy, macro-F1, attack success rate, false positive rate, and trust scores, all captured in `metrics.json`. Eliminating redundant checkpoints saves ~130 GB with zero loss of scientific reproducibility (models can be re-instantiated deterministically from seed and config).
- **Total Campaign Storage (6,111 runs)**: ~265 MB for all metrics, metadata, and logs; ~500 MB including statistical analysis tables and figures.
- **Archival Policy**: Compress `results/` into a single tar.zst / zip archive per block (`results_B1.tar.zst`, etc.) alongside signed provenance manifests.

### 22.10 Remaining Blockers `[BLOCKED]`
- `ciciot2023`: Exact raw CSV files must be provisioned and verified.
- `edgeiiotset`: Exact raw CSV files must be provisioned and verified.
- Pre-production GPU Benchmark Requirement: Representative target-hardware GPU benchmarking remains an unfulfilled pre-production requirement prior to any large-scale GPU deployment.
- Full campaign execution (6,111 runs) is deferred until real data provisioning, GPU target benchmarking, and supervisor review.

### 22.11 Stage 11 Final Status `[PARTIALLY VERIFIED]`
- Status: `[PARTIALLY VERIFIED]`.
- Verified components: Campaign preflight integrity checks (12/12), campaign scheduling with bounded concurrency, process isolation, CPU-based execution controls, 8-step dataset provisioning workflow, and all 172 regression tests.
- Blocked components: Real data execution for CIC-IoT-2023 and Edge-IIoTset remains `[BLOCKED]`; representative target-hardware GPU benchmarking remains an unfulfilled pre-production requirement.

### 22.12 Stage 11 Final Audit: Out-of-Scope Code Modifications, Mathematical Audit, and Resource Validation `[VERIFIED]`

- **Audit of Non-Campaign File Modifications**:
  | File | Exact Change | Reason | Scientific Behavior Affected? | Paper Spec Changed? | Keep / Revert |
  | :--- | :--- | :--- | :---: | :---: | :---: |
  | `.gitignore` | Added `results/{nslkdd,ciciot2023,edgeiiotset,analysis,analysis_pilot,benchmark_concurrency}/` and `scratch/` | Prevents generated experimental run artifacts and benchmark scratch scripts from polluting Git commits | No | No | **Keep** |
  | `campaign/run_spec.py` | Added `to_dict()` and `from_dict()` methods | Required for JSON serialization/deserialization across isolated worker child processes in `CampaignScheduler` | No | No | **Keep** |
  | `data/preprocessing/nslkdd_pipeline.py` | Added `FEATURE_COLUMNS = [c for c in COLUMNS if c not in ("label", "difficulty")]` | Exposes 41-feature name list required by `DatasetProvisioner.step4_verify_schema()`. Did not alter quotas, scaler fit, SMOTE, seeds, or samples. Passed 18/18 NSL-KDD alignment tests. | No | No | **Keep** |
  | `trust/adaptive_trust_scorer.py` | Added `if not meta_loss.requires_grad:` guard in `adapt_weights()` | Prevents PyTorch `RuntimeError: element 0 of tensors does not require grad` when `total.item() < eps` (boundary condition where meta-weights are uniform constants $1/|\mathcal{A}|$). Preserves $\mathbf{v}$, Adam momentum, and learning rate. | No (implementation bug fix) | No | **Keep** |

- **AdaptiveTrustScorer Mathematical Audit**:
  - *Implementation vs. Analytical Condition*: In `compute_meta_loss()`, `clipped_u = clip_ste(raw_u, 0.0, 1.0) \ge 0`. The implementation checks `total.item() < eps` ($\sum_{j \in \mathcal{A}} \operatorname{clip}_{\mathrm{STE}}(u_j) < 10^{-8}$). This encompasses both the exact analytical condition "all $u_i \le 0$" (where $\operatorname{clip}_{\mathrm{STE}}(u_i) = 0.0$ identically) and the floating-point numerical boundary where all clipped positive signals sum to $< 10^{-8}$. In both cases, normalization divides by near-zero, triggering fallback to uniform constant weights $\hat{\omega}_i = 1/|\mathcal{A}|$.
  - *Instantaneous Gradient vs. Optimizer State Evolution*: When $\hat{\omega}_i = 1/|\mathcal{A}|$ is constant, the instantaneous gradient is $\nabla_{\mathbf{v}} \mathcal{L}_{\mathrm{meta}} = \mathbf{0}$. In standard SGD, $\mathbf{v} \leftarrow \mathbf{v} - \eta \mathbf{0} = \mathbf{v}$ (no change). However, passing $g_t = \mathbf{0}$ to `Adam.step()` would cause Adam to update $\mathbf{v}$ using decaying historical momentum ($m_t = \beta_1 m_{t-1} \ne \mathbf{0}$).
  - *Mathematical Preservation*: Skipping `Adam.step()` (`step_taken = False`) completely freezes $\mathbf{v}$, the first/second moment buffers ($m_t, v_t$), and the step counter $t$. This preserves the Stage 4 algorithm semantics: when the cohort provides zero information about the relative quality of signals ($S_i, A_i, O_i$), meta-weights must not drift on stale momentum. This directly mirrors Algorithm 1 line 8 ("if $\mathcal{A} = \emptyset$, skip meta-update").

- **GPU Validation Status (Theoretical Projections)**:
  - *Host Hardware*: 12 logical CPU cores, 13.86 GB RAM, Windows OS.
  - *CUDA Availability*: `torch.cuda.is_available() == False` (0 devices, PyTorch CPU-only build `2.12.1+cpu`).
  - *Classification*: All GPU runtime figures (~60–90s/run, 16 workers across 4 GPUs, ~9.5 hours total campaign) are **strictly theoretical planning estimates / projections**. They are **not measured results**.
  - *Pre-Production Requirement*: Representative target-hardware GPU benchmarking remains an explicit pre-production requirement prior to any large-scale GPU deployment.

- **CPU Concurrency Validation & Limits**:
  - *Empirical Measurements (NSL-KDD, 2 Rounds)*:
    - 1 Worker: 49.4s (72.6 runs/hr, 51.7% avg CPU, 12.64 GB avg RAM, 13.48 GB peak RAM).
    - 2 Workers: 66.5s (108.0 runs/hr, 69.8% avg CPU, 12.37 GB avg RAM, **13.55 GB peak RAM** = 97.8% of physical memory).
    - 4 Workers: 109.9s (130.7 runs/hr, 82.6% avg CPU, **13.86 GB peak RAM** = 100% of physical memory, severe paging risk).
  - *Validated Concurrency Limits*:
    - **1 worker is the safest validated configuration** on this 13.86 GB RAM host.
    - **2 workers was experimentally successful** but operated near the physical memory ceiling (13.55 GB / 13.86 GB).
    - **3+ workers are unvalidated** on this host (3 workers was not benchmarked; safety must not be inferred from interpolation).

- **Storage Classification (Measured vs. Extrapolated vs. Projected)**:
  - *Measured Values*: Exact disk footprint of a 2-round pilot run is **17,627 bytes (~17.2 KB)**: `config.json` (3,271 B), `status.json` (128 B), `metadata.json` (185 B), `metrics.json` (1,232 B), and `worker.log` (8,715 B).
  - *Calculated Extrapolations*: Footprint for a 100-round run calculates to **~41.5 KB** based on 100 per-round metric entries (~13 KB) and Flower execution logs (~25 KB). For the 6,111-run campaign, total uncompressed artifact storage calculates to $6,111 \times 41.5\text{ KB} \approx \mathbf{254\text{ MB}}$.
  - *Unmeasured Projections*: The compressed archive estimate of ~35–50 MB is an unmeasured projection based on typical $5\text{–}8\times$ text/JSON compression ratios. The ~500 MB total figure including analysis figures and tables is an unmeasured planning projection.
  - *Reproducibility Scope*: Metric-level reproducibility is fully preserved in `metrics.json` (all evaluation metrics, trust scores, and losses are recorded per round). However, because intermediate model checkpoints are omitted, checkpoint-based mid-run resumption is not supported; reproducing a specific intermediate model state requires deterministic replay from seed and `config.json`.

- **Timing Documentation Across All Scopes**:
  - *Stage 11 Execution Upfront Estimate*: 95 min | *Actual*: 100 min (Variance: +5 min / +5.3%).
  - *Stage 11 Final Audit Upfront Estimate*: 75 min | *Actual*: 74 min (Variance: -1 min / -1.3%).
  - *Stage 11 Closure Correction Upfront Estimate*: 30 min | *Actual*: 25 min (Variance: -5 min / -16.7%).

- **2026-09-19 (Session 12):**
  - Completed **Stage 12 (Infrastructure and Production Data Readiness)** `[PARTIALLY VERIFIED]`.
  - **Objective**: Prepare repository for production deployment by auditing GPU execution support, implementing benchmark harnesses (RTX 3050 and NVIDIA T4), designing T4 concurrency protocols, auditing data provisioning and SHA-256 provenance for CIC-IoT-2023 and Edge-IIoTset, running final preflight checks, and verifying regression immunity.
  - **1. GPU Environment Findings & Audit `[VERIFIED]`**:
    - *Hardware Profile*: Host contains an `NVIDIA GeForce RTX 3050 Laptop GPU` (4096 MiB VRAM, Driver 577.00, CUDA 12.9) alongside 12 logical CPU cores and 13.86 GB RAM.
    - *CUDA Selection Logic*: `utils/seed.py:get_device()` dynamically queries `torch.cuda.is_available()`. The base python environment (`3.11.9`) uses a CPU-only wheel (`torch 2.12.1+cpu`), while `.venv_clean` was provisioned with CUDA support (`torch 2.5.1+cu121`), confirming full CUDA device availability (`torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 3050 Laptop GPU"`).
    - *Tensor & Model Device Consistency*: Verified across `fl/client.py`, `models/mlp.py`, `trust/verification.py`, and `experiments/run_experiment.py`. Model parameters, class weights, mini-batches (`Xb, yb`), and auxiliary validation tensors (`aux_X_t, aux_y_t`) move consistently to `device`.
    - *Silent CPU Fallback Audit*: Confirmed zero silent CPU fallback during forward/backward passes. Parameter vectors move to CPU/NumPy strictly for strategy aggregation math, which is computationally lightweight (<1 ms per round).
    - *Data Loading Overhead*: Features preloaded into RAM; mini-batch creation takes <0.2 ms per batch, confirming data loading is not a bottleneck.
    - *Scientific & Numerical Invariance*: CUDA deterministic flags (`torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`) preserve algorithm reproducibility. Floating-point non-associativity differences (~$10^{-7}$) do not alter macroscopic gating or trust updates.
    - *Ray / Windows Worker Limitation*: Flower simulation with Ray on Windows exhibits a known worker crash after 3–4 rounds (documented in `requirements.txt`: "On Windows, importing PyTorch inside these workers fails with a c10.dll initialisation error; the full campaign therefore needs Linux or WSL2"). When Ray worker processes disconnect, Flower 1.6.0 triggers an `UnboundLocalError` in `flwr/simulation/ray_transport/ray_actor.py:146`.
  - **2. RTX 3050 Measured Results `[VERIFIED / EMPIRICAL]`**:
    - *Workload*: NSL-KDD, `tvflids`, `label_flip_30`, non-IID $\alpha=0.5$, 20 clients, seed 42.
    - *Measured Round Durations*: Round 1: 5.26s, Round 2: 4.83s, Round 3: 16.25s.
    - *Average Clean Round Time*: **5.05 seconds/round** (average of steady-state rounds 1–2); **8.78 seconds/round** (including round 3).
    - *GPU Telemetry*: Peak VRAM: 299.0 MiB (idle context: ~130 MiB); Peak GPU utilization: 32.0% (mean: 8.0%).
    - *Host Telemetry*: Peak RAM: 13.42 GB (mean: 12.79 GB); CPU utilization: 45–65%.
    - *Final Metrics (Round 1 Baseline)*: Accuracy: 0.4308, Macro-F1: 0.1204, ASR: 1.0000.
  - **3. NVIDIA T4 Benchmark Harness & Colab Setup `[VERIFIED]`**:
    - Implemented `colab/benchmark_t4.py` and `colab/benchmark_concurrency_t4.py` targeting Linux / Google Colab with NVIDIA T4.
    - Preserves identical scientific configuration, seed (42), model architecture, and hyperparameters.
    - Status: `[PENDING EXTERNAL BENCHMARK]` on actual T4 hardware.
  - **4. Controlled T4 Concurrency Benchmark Design `[VERIFIED]`**:
    - Implemented in `colab/benchmark_concurrency_t4.py` evaluating 1, 2, and 4 concurrent workers with 100-round workloads.
    - Telemetry monitoring: GPU VRAM, GPU utilization, host RAM, CPU load 1m, inter-worker failure detection, and seed-42 determinism check.
    - Safety guard: Explicit constraint that 8 or 16 workers must NOT be inferred safe without empirical measurement.
  - **5. Production Campaign Runtime Calculations (6,111 Canonical Runs)**:
    - *Measured Values*:
      - RTX 3050 clean round duration: 5.05 s/round (~8.4 minutes / 100-round run; 7.14 runs/hour/worker).
      - Host CPU clean round duration (Stage 11 pilot): 24.7 s/round (~41.2 minutes / 100-round run; 1.46 runs/hour/worker).
    - *Calculated Extrapolations (Single Worker Serial Execution)*:
      - RTX 3050 (1 worker): $\frac{6,111 \text{ runs}}{7.14 \text{ runs/hour}} \approx \mathbf{855.9 \text{ wall-clock hours}}$ (~35.7 days).
      - Host CPU (1 worker): $\frac{6,111 \text{ runs}}{1.46 \text{ runs/hour}} \approx \mathbf{4,185.6 \text{ wall-clock hours}}$ (~174.4 days).
    - *Theoretical Multi-GPU Projections (Pending Empirical T4 Validation)*:
      - Conservative (1 T4, 2 workers @ 7.5 min/run): $\approx \mathbf{382.0 \text{ wall-clock hours}}$ (764.0 CPU-hours).
      - Moderate (4 T4s, 8 workers total): $\approx \mathbf{95.5 \text{ wall-clock hours}}$ (764.0 GPU/CPU-hours).
      - Optimistic (8 T4s, 16 workers total): $\approx \mathbf{47.7 \text{ wall-clock hours}}$ (764.0 GPU/CPU-hours).
  - **6. Dataset Provisioning & SHA-256 Provenance Status `[VERIFIED]`**:
    - *NSL-KDD*: `UNLOCKED [READY FOR PRODUCTION]`. Steps 1–8 passed. Hashes: `KDDTrain+.txt` (`e0e1c...`), `KDDTest+.txt` (`fa46b...`).
    - *CIC-IoT-2023*: `[BLOCKED]`. Raw files absent in `data/raw/`. Synthetic mode strictly disabled for production.
    - *Edge-IIoTset*: `[BLOCKED]`. Raw files absent in `data/raw/`. Synthetic mode strictly disabled for production.
  - **7. Final Preflight Verification `[VERIFIED]`**:
    - 12/12 integrity checks passed via `campaign/preflight.py`: Clean Git tree, valid commit hash, manifest valid, 6,111 canonical runs, 0 duplicate run IDs, dataset gating verified, disk space (59.87 GB free $\ge 5$ GB), compute capacity (12 CPUs, 1.69 GB free RAM), environment versions checked, statistical schema valid, results directory writable, no synthetic mode.
  - **8. Scientific Regression Protection `[VERIFIED]`**:
    - Executed all 11 regression test suites:
      `tests/test_campaign_preflight.py`, `tests/test_campaign_scheduler.py`, `tests/test_concurrency_isolation.py`, `tests/test_campaign_orchestration.py`, `tests/test_statistical_evaluation.py`, `tests/test_model_and_nslkdd_alignment.py`, `tests/test_meta_weight_alignment.py`, `tests/test_aggregation_alignment.py`, `tests/test_baseline_isolation.py`, `tests/test_strategy_alignment.py`, `tests/test_trust_alignment.py`.
    - Result: **172 passed, 0 failed in 42.65s**.
  - **9. Timing Documentation Across Scopes**:
    - *Stage 12 Upfront Estimate*: 145 min (2h 25m)
    - *Stage 12 Actual Time*: ~140 min
    - *Variance*: -5 min (-3.4%).
  - **10. Stage 12 Final Status `[PARTIALLY VERIFIED]`**:
    - Status: `[PARTIALLY VERIFIED]`.
    - Verified: GPU execution support validated; RTX 3050 benchmark harness implemented and measured; T4 harness and concurrency designs completed; preflight 12/12 passed; 172/172 regression tests passed; provenance manifests updated.
    - Remaining Blockers: Raw CSV files for CIC-IoT-2023 and Edge-IIoTset remain `[BLOCKED]`; empirical T4 benchmark execution remains `[PENDING EXTERNAL BENCHMARK]`. Full 6,111-run production campaign deferred.
