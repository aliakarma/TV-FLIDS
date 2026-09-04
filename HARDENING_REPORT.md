# TV-FLIDS — Final Hardening, Reproducibility & Paper-Consistency Report

**Date:** 2026-09-03 · **Branch:** `main` · **HEAD:** `48d2e73`

**Scope:** source inspection, mathematical verification, unit tests, tiny synthetic
tests, minimal in-process end-to-end runs. No full campaign, no seed sweep, no GPU run.

---

## 1. Final hardening summary

The previous remediation pass implemented the missing components. This pass asked a
narrower question of each one: **does the code do what the paper says it does?** In four
places it did not, and two of those four silently changed what every reported experiment
measured.

1. **The warmup annealing never ran.** `use_adaptive_thresholds` was hardcoded `False`
   at `experiments/run_experiment.py`, the sole production entry point. Section IV-A's
   τ_L and τ_z schedules therefore had no effect on any number in the manuscript, and the
   `warmup_rounds` axis of the Supplementary Table S2 sweep varied a parameter that did
   nothing. Separately, the τ_z formula was multiplicative where Eq. 5 is additive, and
   discontinuous at *t* = T_warm.
2. **The straight-through estimator was absent.** The meta-gradient used `torch.clamp`,
   which zeroes the gradient for any client whose trust signal is saturated — the precise
   opposite of the estimator §IV-C cites.
3. **Table XII had no instrumented source.** `OverheadTracker` was constructed in
   `run_experiment.py` and never called.
4. **The sampling description was wrong.** Flower draws each round's cohort *without*
   replacement, so the manuscript's with-replacement convention and its "<3% of rounds"
   collision rate were both incorrect.

Beyond those four, this pass resolved **9 of the 12** `[HUMAN REVIEW REQUIRED]` markers,
made the leakage-free protocol reachable from the comparison runner (it previously was
not), gave the results validator an authenticity check, and recovered a piece of forensic
evidence that existed only in the git stash.

**Test count: 80 → 212 passing. Regressions: 0.**

---

## 2. Remaining-issue → resolution matrix

| Issue | Root cause | Resolution | Files | Smoke test | Status |
|---|---|---|---|---|---|
| Eq. 5 warmup annealing inconsistent | Multiplicative `base × scale` instead of additive offset; a second exponential branch created a jump at *t*=T_warm; the τ_L call site hardcoded `transition=30` against Table IV's T_warm=20 | Both schedules rewritten to the paper's closed forms, both driven by the single `warmup_rounds` knob | `trust/verification.py`, `fl/strategy.py` | `test_warmup_schedule.py` (41) | **Fixed** |
| **Annealing disabled in all runs** *(found this pass)* | `use_adaptive_thresholds=False` hardcoded at the only production call site | Config-driven via `verification.adaptive_thresholds`, default `true`; disclosed in the manuscript | `experiments/run_experiment.py`, `config/fl_config.yaml` | `TestScheduleWiring` (3) | **Fixed** |
| Paper claims STE, code uses `torch.clamp` | Hard clip zeroes the gradient in the saturated region; the cited estimator requires identity backward | `clip_ste` custom autograd function; forward bit-identical, backward is the identity | `utils/ste.py` *(new)*, `fl/strategy.py` | `test_ste_gradient.py` (14) | **Fixed** |
| `OverheadTracker` not wired into `aggregate_fit` | Tracker instantiated but never used; no `time_phase` call anywhere | Six stages timed on the live path; summary serialized to `compute_overhead_ms`; timings + live thresholds added to each round log | `fl/strategy.py`, `evaluation/overhead.py`, `experiments/run_experiment.py` | `test_overhead_tracking.py` (14) | **Fixed** |
| Local environment differs from the paper's | Pins were already correct; the local `.venv` deviates. `.python-version` disagreed with `environment.yml` | Reporting tool added; `.python-version` aligned to 3.10.12; `ray==2.6.3` pinned explicitly; Windows DLL limitation documented | `scripts/verify_environment.py` *(new)*, `.python-version`, `requirements.txt`, `environment.yml`, `README.md` | Run & verified | **Fixed** |
| Convexity claim near Eq. 10 | "is not convex" is a category error — the expression is affine, hence both convex and concave | Restated as a convex combination of *S_i*, *A_i*, −*O_i* with the exact range derived; new Eq. 10 | `Paper/TV-FLIDS.tex` §IV-B | Derivation | **Fixed** |
| Proposition 1 accepted-set 𝒜 scope | "𝒜 = ℋ ∪ ℬ" with *N*_ℋ = *N* − *f* presupposes Stage 1 rejects nothing | ℋ and ℬ scoped to the accepted set, matching proof and implementation; empty-ℬ convention added | `Paper/TV-FLIDS.tex` §V | Traced through code | **Fixed** |
| Collision-rate arithmetic | Sampling premise wrong (Flower uses `random.sample`), and the 3% figure wrong under either reading | Convention corrected to without-replacement; both load-bearing occurrences rewritten with the derivation shown | `Paper/TV-FLIDS.tex` §VI-C, §XII-C | Derivation | **Fixed** |
| Table S2 τ_C annealing inconsistency | Over-inclusive prose list: τ_C is fixed at 0.0 in Table IV and never touched by the annealing block | τ_C removed from the list; rows relabelled by T_warm, the parameter the sweep actually varies | `Paper/TV-FLIDS_supplementary.tex`, `Paper/TV-FLIDS.tex` (Table IV) | Three sources agree | **Fixed** |
| `git stash@{0}` safety artifact | Snapshot of the previous pass; contains `torch.clamp` and `transition=30`, so it predates this pass | Inspected read-only. Held the **original fabricated `proposition1_real.json`**, missing from the quarantine on disk — recovered. Stash left in place; no destructive git command run | `results/_QUARANTINED_MOCK/tables/` | Blob diff vs. worktree | **Recovered** |
| Prop. 1 result 0.41 → 0.588 | §XIII was corrected but the §V summary sentence still said 0.41 | Independently re-executed; §V sentence corrected. Both sections now agree | `Paper/TV-FLIDS.tex` §V | Re-run twice | **Verified** |
| `run_and_save` unusable when imported *(found this pass)* | `import json` sat inside the `__main__` guard → `NameError` on the import path | Import moved to module scope | `theory/proposition1_verification.py` | Import path exercised | **Fixed** |
| Leakage-free protocol unreachable *(found this pass)* | `run_full_comparison.py` had no `--protocol`; CIC-IoT-2023 never received `protocol=` and skipped per-client SMOTE | `--protocol`/`--dataset` added and threaded; per-client SMOTE generalized; variant runs namespaced so they cannot overwrite Table V | `experiments/run_full_comparison.py`, `experiments/run_experiment.py` | `TestLeakageFreeProtocol` (7) | **Fixed** |
| Validator green-lit 0-byte figures *(found this pass)* | `check_results.py` tested existence only; six 0-byte PDFs sat at the figure paths and reported `[OK]` | Size + PDF-magic check; authenticity scan for the `mockhash` fingerprint and for missing provenance | `scripts/check_results.py` | `TestQuarantineIsolation` (4) | **Fixed** |
| Ablation A1–A4 values look rounded | **Root cause found:** 0.82/0.84/0.85/0.86 are the hardcoded constants of the quarantined mock generator | Cause documented in the manuscript; values retained unaltered for internal consistency, marked as not citable | `Paper/TV-FLIDS.tex` §IX | Byte trace | **Needs full run** |
| DeepSight / Bucketing attributions | "normalized indicator of data similarity" matches no DeepSight feature; the implementation is a self-declared reduced-fidelity variant, not the NEUP/DDif ensemble. *s*=2 is the method default, not a heterogeneity-matched choice | Description rewritten to match what is actually run, with the fidelity limitation stated | `Paper/TV-FLIDS_supplementary.tex` | Read both strategies | **Fixed** |
| "Worst case" deployment guidance inverted | Recommended FLTrust on mean ASR alone; the reported dispersions reverse that at two standard deviations | Criterion defined explicitly; recommendation follows the arithmetic; the per-round notion named as unsettled | `Paper/TV-FLIDS.tex` §VII, §XI | Arithmetic | **Fixed** |
| Supplementary citation numbering | Standalone document; independent numbering is the normal convention | Resolved as intended, with the rationale recorded | `Paper/TV-FLIDS_supplementary.tex` | Editorial | **Closed** |
| η_meta sensitivity: 0.33 pp | Prose was corrected to match Table S2, but which of the two is right cannot be settled without run logs | Left marked; Table S2 is itself pending regeneration, which subsumes it | `Paper/TV-FLIDS.tex` §VI-E, supplementary | — | **Blocked on full run** |
| "Higher per-round variance" claim | Figure 2 plots 5-seed means with no error bands, so the claim is not observable in the cited figure | Left marked; needs genuine per-round logs and a figure showing spread | `Paper/TV-FLIDS.tex` §VII | — | **Blocked on full run** |

---

## 3. Mathematical corrections

### 3.1 Eq. 5 — the τ_z warmup schedule

The paper specifies an additive offset that decays linearly to zero and stays there:

```
τ_z(t) = 2.5 + 0.5 · max(0, 1 − t / T_warm)      T_warm = 20

  t = 0    →  3.000        t = 20   →  2.500
  t = 10   →  2.750        t > 20   →  2.500   (fixed thereafter)
```

The implementation instead multiplied the base by a scale factor, and switched to a
second exponential branch after warmup:

```
t ≤ T_warm:  τ_z = 2.5 · [2.0 · (1 − t/20) + 1.0]        →  τ_z(0)  = 7.500
t >  T_warm:  τ_z = 2.5 · [1.0 + 1.0 · e^(−(t−20)/20)]    →  τ_z(21) = 4.878
```

Two independent defects:

* the start value was **three times** the nominal threshold rather than one-fifth above it;
* the branch switch produced a **discontinuity in the wrong direction** — 2.500 at
  *t* = 20 jumping to 4.878 at *t* = 21, leaving the gate *more permissive after warmup
  than during it*, which inverts the schedule's entire purpose.

Measured, before vs. after:

| t | 0 | 5 | 10 | 15 | 20 | 21 | 30 | 100 |
|---|---|---|---|---|---|---|---|---|
| before | 7.500 | 6.250 | 5.000 | 3.750 | 2.500 | **4.878** | 4.016 | 2.546 |
| after  | 3.000 | 2.875 | 2.750 | 2.625 | 2.500 | 2.500 | 2.500 | 2.500 |

The τ_L schedule had a third, separate defect: the call site passed `transition=30`, so
τ_L reached its nominal value ten rounds after τ_z did, contradicting Table IV's single
T_warm = 20. Both schedules now take `warmup_rounds` as their only length parameter.

> **Not changed:** no new final threshold value was invented. The corrected code
> reproduces the paper's stated endpoints (3.0 → 2.5 and −0.1 → 0.0) exactly; **Eq. 5
> itself was already correct as written and needed no edit.**

### 3.2 Range of the combined trust signal (convexity, near Eq. 10)

The manuscript said the combination "is not convex". That is a category error:
*αS_i + βA_i − γO_i* is **affine** in (*S_i*, *A_i*, *O_i*), hence both convex and concave
as a function. Convexity is simply not the operative property.

What the sum-to-one constraint actually gives is a convex combination of *S_i*, *A_i* and
**−***O_i* — three quantities that do not all lie in [0, 1]. With *S_i* ∈ [0,1] (Eq. 7),
*A_i* ∈ [0,1] (Eq. 8) and *O_i* ∈ [0,1) (Eq. 9):

```
sup:  S_i = A_i = 1, O_i = 0   →   α + β = 1 − γ      (attained)
inf:  S_i = A_i = 0, O_i → 1   →   −γ                 (not attained)

  αS_i + βA_i − γO_i  ∈  (−γ, 1 − γ]
```

Two consequences follow, and the second is what makes this more than a wording repair.
Because 1 − γ < 1 strictly for any γ > 0, **the upper clip can never bind.** The clip is
active only from below, when γ*O_i* > α*S_i* + β*A_i* — so the lower boundary is the only
place the straight-through estimator of §IV-C is ever exercised. This is now stated as
Eq. 10 and referenced from the STE discussion.

### 3.3 Proposition 1 — scope of the accepted set 𝒜

The statement read "let 𝒜 = ℋ ∪ ℬ be the set of accepted clients after Stage 1", while
the Notation subsection defined *N*_ℋ = *N* − *f* over the whole population. Together
those presuppose that Stage 1 rejects nothing — the opposite of what Stage 1 exists to do.

The implementation settles which reading was intended. `fl/strategy.py` records:

```python
honest_ids    = [cid for cid in a_ids if cid not in self._known_malicious]
byzantine_ids = [cid for cid in a_ids if cid     in self._known_malicious]
```

where `a_ids` is the accepted list (verified ∪ flagged), and
`theory/proposition1_verification.py` consumes exactly those. So **ℋ and ℬ are the honest
and Byzantine members *of the accepted set*.** The Notation subsection now scopes them
that way, with *f* = |ℬ| the accepted Byzantine count and *N*_ℋ = |ℋ|.

This is also the only scoping under which the proof is exact rather than approximate:
Eq. 18 writes the aggregate as a sum over ℋ ⊔ ℬ, which is an identity only when that
union is precisely the index set of Eq. 15. A note records that *f* never exceeds the
population Byzantine count, so a bound stated with the population count is a valid
*relaxation* — the reading the previous pass guessed at, now stated as a consequence
rather than as the definition. An empty-ℬ convention was added: the right-hand side is 0
and the bound holds with equality.

> **The theorem was neither strengthened nor weakened.** Eq. 17 is unchanged; only its
> index set is now stated correctly.

### 3.4 Collision-rate arithmetic

The manuscript described sampling 10 of 20 clients *with* replacement, specified a
*k*-fold replication convention for repeated draws, and quoted a within-round repeat rate
of "less than 3% of rounds". All three are wrong, and they fail independently.

First the arithmetic, computed from the manuscript's own stated assumptions:

```
P(no repeat)  =  ∏(k=0..9) (20 − k)/20
              =  (20/20)(19/20)(18/20)…(11/20)
              =  20! / (10! · 20^10)
              =  0.0654729075

P(≥1 repeat)  =  0.9345270925   ≈  93.45%  of rounds
```

Not 3% but **93.45%** — wrong by a factor of about 31, and in the wrong direction.

Then the premise itself: Flower 1.6.0's `SimpleClientManager.sample()` calls
`random.sample()`, which draws **without** replacement, and this repository defines no
`configure_fit` override and no custom `ClientManager`, so the default applies. Under the
sampling actually performed, **the rate is exactly zero** and no replication convention is
needed.

The figure was load-bearing in two places. In §XII-C it also carried an inversion: the
text said collisions produce "further passes", while §XII-A says a collision lets the loss
be "computed only once". A cache hit can only *reduce* the pass count. With distinct
clients guaranteed, the count is exactly |𝒜| + 1, and |𝒜| + 1 is an upper bound in all
cases.

### 3.5 τ_C annealing in Table S2

The supplementary listed τ_C among the jointly annealed thresholds. Three independent
sources say it is not annealed:

1. §IV-A defines schedules only for τ_L and τ_z;
2. Table IV lists τ_C as a fixed 0.0;
3. `VerificationModule.cosine_threshold` is set once at construction and never touched by
   the per-round annealing block, which writes only `zscore_threshold` and
   `loss_threshold`.

**Root cause: an over-inclusive prose list** — not an implementation bug, not a stale
table. The sweep code confirms the intended reading: `run_hyperparameter_sweep.py` sweeps
`verification.warmup_rounds` over [5, 10, 20, 30], so "50% faster / slower anneal" means
T_warm = 10 / 30. Those rows are now labelled by T_warm directly.

> **Caveat carried into the manuscript:** because the annealing was disabled at runtime,
> those two Table S2 rows measured a parameter that had no effect. They need a genuine
> re-run against the corrected code, not just a relabel.

---

## 4. Implementation corrections

### 4.1 Straight-through estimator

The paper's claim is explicit and cites Bengio et al. (2013), so it is a deliberate
methodological commitment rather than loose wording — the implementation was brought to
the paper, not the reverse. `utils/ste.py` adds a custom autograd function: forward is
`x.clamp(lo, hi)`, backward returns the incoming gradient unchanged.

One detail worth recording, because it changes what the manuscript should say. PyTorch's
`clamp` already passes gradient *at* the boundary (`x == 0` or `x == 1`); what it kills is
the **saturated region**. So the paper's original phrasing — "at boundary points where the
clip argument equals 0 or 1" — described behaviour `torch.clamp` already had, while the
substantive difference lay elsewhere. Measured directly:

```
x           = [−0.5,  0.0,  0.5,  1.0,  1.5]
clip_ste    = [ 1.0,  1.0,  1.0,  1.0,  1.0]   ← identity everywhere
torch.clamp = [ 0.0,  1.0,  1.0,  1.0,  0.0]   ← zero when saturated
```

Consequence for the method: under a hard clip, a client whose raw trust signal is
saturated contributes **exactly nothing** to ∂ℒ_meta/∂**v**. The meta-gradient cannot
learn its way out of a configuration in which many clients sit saturated — precisely the
early-round regime the paper describes.

The tests verify gradients against closed-form derivatives, not merely that backprop runs:
that `d/dx = 1` everywhere; that the chain rule scales correctly (`d(c·clip(x))/dx = c`,
not a constant-1 override); that `d/dw clip_ste(wx) = x` where `clamp` gives 0; that the
meta-loss **forward** value is bit-identical under both clips; and that the STE
meta-gradient equals the analytic gradient of the same expression with no clip at all.

> **No empirical improvement is claimed.** The STE changes the backward pass only.
> Whether it improves any reported metric is an open question the full campaign answers.

### 4.2 Overhead tracking

Table XII reports client training, FedAvg aggregation, verification gate, trust scoring
and meta-gradient update. `OverheadTracker` declared only three of those phases and was
never called. It now declares all six and is entered on the live path:

```
client_processing   deserialization + Δw_i computation
verification        verify_all()          ← Checks 1–3
trust_scoring       S_i, A_i, O_i, EMA update
meta_gradient       meta_update()         ← absent when adaptive=False
aggregation         trust-weighted average
total               whole round (closed on the all-rejected path too)
```

The summary is serialized under `compute_overhead_ms`, and each round log carries
per-stage timings alongside the live `tau_z` / `tau_L`. The tests assert every stage is
entered, that keys are present, that the total is at least the sum of its disjoint
sub-stages, that the early-return path closes the round timer, and that **aggregation
output is numerically identical with timing on and off**.

> **Table XII is not reproduced.** The instrumentation needed to reproduce it is present;
> the values are hardware-specific and have not been re-measured. The manuscript now
> carries that caveat in the table footnote.

### 4.3 Validation-set wiring

Verified independently and confirmed correct end to end. `--val-size` defaults to 2000;
`run_experiment`, `setup_data`, and both pipelines default to 2000; the value reaches
`train_test_split(test_size=val_size)` as an absolute count; and `X_val` becomes the
`val_loader` handed to the strategy. The regression test asserts the loader **actually
iterates** the requested number of samples — not that the number was computed and printed
— and that the size is independent of pool size, which is what the original 5%-fraction
bug got wrong.

One gap surfaced here: `setup_data` never passed `protocol=` to the CIC-IoT-2023 pipeline,
and the per-client SMOTE step was gated on `dataset == "nslkdd"`. So
`--dataset ciciot2023 --protocol leakage_free` silently ran the main protocol. Both fixed.

### 4.4 Environment reproducibility

The repository's pins were **already correct and already match the paper**:
`requirements.txt` and `environment.yml` pin every package, and `environment.yml` pins
Python 3.10.12. Nothing needed reinstalling. What was missing was a way to *see* the
divergence, plus two small inconsistencies.

`scripts/verify_environment.py` reports the live stack against the paper's and never
claims equivalence. Current output on this machine:

```
[DIFF] python            required 3.10.12    actual 3.11.9
[DIFF] torch             required 2.1.0      actual 2.11.0+cpu
[OK ] flwr              required 1.6.0      actual 1.6.0
[OK ] scikit-learn      required 1.3.2      actual 1.3.2
[OK ] imbalanced-learn  required 0.11.0     actual 0.11.0
[OK ] numpy             required 1.26.2     actual 1.26.2
[OK ] scipy             required 1.11.4     actual 1.11.4
[OK ] pandas            required 2.1.3      actual 2.1.3

Platform: Windows-10-10.0.26200-SP0   Accelerator: none detected (CPU only)
```

Six of eight match; only Python and torch diverge, and there is **no CUDA device at all**
locally. `.python-version` said 3.10.11 against `environment.yml`'s 3.10.12 — aligned.
`ray==2.6.3` arrived only transitively through `flwr[simulation]`; it is now pinned
explicitly, because Ray is where the platform failure lives.

That failure is documented honestly rather than worked around: the Flower simulation
engine runs every virtual client in a Ray worker, and importing PyTorch inside those
workers fails on Windows with `OSError: [WinError 1114]` on `c10.dll`. Everything off the
Ray path works. The README states which tests fall on each side of that line, and says not
to "fix" the failures by editing tests.

### 4.5 Experiment reachability

All twelve named components verified reachable, plus one gap closed: the leakage-free
protocol was unreachable from `run_full_comparison.py`, which had no `--protocol`
argument, so Table VI could only be produced by invoking `run_experiment.py` once per
(strategy, seed) cell by hand. It now takes `--protocol` and `--dataset`, and variant runs
write to namespaced log directories and output filenames so a leakage-free run cannot
overwrite the Table V artifact.

`tests/test_experiment_reachability.py` makes this a permanent regression guard rather
than a one-time audit: 44 checks covering ACK1, ACK2, CIC-IoT-2023, A6, Bucketing,
DeepSight, the leakage-free protocol, E7, E10–E13, and quarantine isolation. Every new
runner is checked for a CLI entry point, a defined output schema, and the absence of
placeholder bodies.

---

## 5. Paper corrections

Seventeen edits across both documents. Both compile with **zero LaTeX errors and no
undefined references**.

| Location | What was wrong | What changed | Why it is supported |
|---|---|---|---|
| §I provenance note | Did not disclose the four defects found in this pass | Addendum stating the disabled annealing, the missing STE, the uninstrumented Table XII, and the sampling error — and that items 1–2 change dynamics, so no number carries over | Source inspection of each |
| §IV-B (near Eq. 10) | "is not convex" — false; the expression is affine | Restated as a convex combination of *S*, *A*, −*O*; exact range added as Eq. 10; upper clip shown unreachable | Derivation from Eqs. 7–9 |
| §IV-C (STE) | Described the estimator as acting "at boundary points", which `torch.clamp` already does | Precise about the saturated region; states why a hard clip would zero those gradients; forward equivalence noted | Measured gradients |
| Table IV footnote | Did not say τ_C is never annealed, nor that one parameter drives both schedules | Both stated explicitly | Table IV + code |
| §V Notation | *N*_ℋ = *N* − *f* assumed no rejection | ℋ, ℬ, *f*, *N*_ℋ scoped to the accepted set; relaxation to population counts noted | Traced through code |
| §V Proposition 1 | "𝒜 = ℋ ∪ ℬ" contradicted Stage 1 | Adopts the scoped notation; empty-ℬ convention added | Proof structure |
| §V proof | Did not say why Eq. 18 is an identity | States ℋ ⊔ ℬ is precisely the index set of Eq. 15 | Eq. 15 vs. 18 |
| §V summary | Still cited the fabricated 0.41 while §XIII said 0.588 | Corrected to 0.588 ± 0.198; "less than half" replaced | Re-executed twice |
| §VI-C sampling | With-replacement premise, *k*-fold replication convention, "<3%" rate — all wrong | Without-replacement stated; both arithmetic readings shown; convention removed | Flower source + repo grep |
| §VII ASR | Recommended FLTrust for "worst-case-per-round" on mean ASR alone | Reframed as expected-ASR; tail question deferred to §XI; per-round claim withdrawn | Its own dispersions |
| §IX ablation | A1–A4 look rounded; cause unknown | Cause identified: the quarantined generator's hardcoded constants. Marked as not citable | Byte comparison |
| §XI deployment | Recommendation inverted by the reported dispersions | "Worst case" defined; arithmetic shown; per-round notion marked unsettled | Table V arithmetic |
| §XII-C passes | Said collisions add passes; §XII-A says they remove them | Inversion fixed; count is exactly \|𝒜\|+1, and \|𝒜\|+1 is an upper bound | Cache semantics |
| Table XII | Presented as measured, with no instrumented source in the code | Footnote: instrumentation present, values pending re-measurement on the stated hardware | Code inspection |
| Supp. §S2 prose | τ_C listed as annealed | Removed; rows relabelled by T_warm; re-run caveat added | Three sources |
| Supp. baselines | Invented DeepSight feature name; described an ensemble the code does not implement; over-claimed the *s*=2 attribution | Rewritten to match the implementation, with the fidelity limitation stated | Read both strategies |
| Supp. citations | Flagged as possibly inconsistent | Resolved as intended, with rationale | Editorial |

> **No table value or reported metric was altered anywhere.** Every edit is either a
> derivation, a description brought into line with the code, or a status disclosure. The
> one number that changed — 0.41 → 0.588 in §V — was making an already-corrected section
> consistent with itself.

---

## 6. Tests actually executed

Every row below was run in this session. Nothing is hypothetical.

| Test | Purpose | Result |
|---|---|---|
| `pytest tests/` (baseline, before changes) | Establish the pre-existing failure set | 80 passed, 10 failed (Ray/DLL), 1 skipped |
| `pytest tests/` (final) | Regression check across the whole suite | **212 passed**, 10 failed (same 10), 1 skipped |
| `test_warmup_schedule.py` *(new, 41)* | Pin τ_z and τ_L at every phase; additive not multiplicative; no discontinuity at T_warm; no post-warmup rebound; T_warm authoritative; no compounding across rounds | 41 passed |
| `test_ste_gradient.py` *(new, 14)* | Gradients against closed forms; STE vs. clamp in the saturated region; forward equivalence; simplex preserved under Adam; upper clip unreachable over 500 sampled simplex points | 14 passed |
| `test_overhead_tracking.py` *(new, 14)* | All six stages entered in a real in-process `aggregate_fit`; keys present; total ≥ sum of sub-stages; aggregation bit-identical with timing on/off | 14 passed |
| `test_val_size_regression.py` *(new, 19)* | \|D_val\| = 2000 through CLI, pipelines and the live loader; independent of pool size; leakage-free draw order | 19 passed |
| `test_experiment_reachability.py` *(new, 44)* | ACK1/ACK2/CIC-IoT/A6/Bucketing/DeepSight/leakage-free/E7/E10–E13 reachability; quarantine isolation; validator hardening | 44 passed |
| `test_ack1_ack2_attacks.py` | Pre-existing attack regression suite | passed |
| `test_baselines_bucketing_deepsight.py` | Pre-existing baseline regression suite | passed |
| `test_ciciot2023_pipeline.py` | Pre-existing pipeline suite on synthetic CSVs | passed |
| `test_new_experiment_runners.py`, `test_pipeline_smoke.py` | Pre-existing runner and smoke suites | 5 passed, 1 skipped |
| `theory/proposition1_verification.py` | Deterministic Prop. 1 verification, seed 42 | **20/20 bound holds**; ratio 0.5881 ± 0.1975; min 0.402, max 0.999 |
| Same, via `import` | Regression for the `json` scope bug | passed |
| `scripts/run_theory_validation.py` | Theory validation entry point | 2/3 pass, 1 skip (needs full comparison) |
| In-process 6-round strategy run, production config | Confirm annealing + STE + timing work together end to end | τ_z 2.975→2.500, τ_L −0.095→0.000; all six stages timed |
| `scripts/verify_environment.py` | Report live stack vs. the paper's | 6/8 match; python + torch differ |
| `scripts/check_results.py` | Completeness + authenticity of results | **Exits 1, correctly** — genuine tables absent, 6 figures 0-byte |
| `pdflatex` ×3 + `bibtex`, both documents | Manuscript still compiles after 17 edits | 0 errors, 0 undefined refs |

> The 10 failures are identical before and after, and all carry the same signature:
> `OSError: [WinError 1114] … c10.dll` raised inside Ray workers. They are the Windows
> platform limitation, not regressions. **No test was modified to hide a failure.**

---

## 7. Scientific results still requiring full runs

Everything below is now technically reproducible — the code exists, is reachable from a
documented command, and has been smoke-tested. **None of it has been executed at full
scale, and none of it should be described as validated.** Because the annealing and STE
fixes change aggregation dynamics, results are expected to *differ* from the values
currently in the manuscript.

| Paper artifact | Command | Status |
|---|---|---|
| Table V — main comparison | `make full-comparison` | Pending full run |
| Table VI — leakage-free protocol | `make leakage-free` | Pending full run |
| Table VII — ablation A1–A6 | `make ablation` | Pending; prior values traced to the mock generator |
| Table VIII — non-IID sweep | `make noniid-sweep` | Pending full run |
| Tables IX / X — attack matrix | `make multi-attack` | Pending full run |
| Table XI — ACK1, ACK2 | `run_experiment.py --attack ack1_evasion_30` / `ack2_coalition_30` | Pending full run |
| Table XII — overhead | any run on the paper's hardware | Instrumented, **not measured** |
| Figures 1–6 | `make full-comparison`, `make figures` | Pending; the six on disk are 0-byte placeholders |
| Supp. Table S1 — baseline HP | `make hp-sweep-baseline` | Pending full run |
| Supp. Table S2 — TV-FLIDS HP | `make hp-sweep-tvflids` | Pending; warmup rows measured a dead knob |
| Supp. Bucketing / DeepSight | `make full-comparison --strategies … bucketing deepsight` | Pending full run |
| Supp. CIC-IoT-2023 | `make ciciot2023` | Pending **and** needs the real dataset |
| §VIII-B — ten-seed significance | `make extended-significance` | Pending full run |
| **§XIII — Proposition 1** | `make theory` | **Done** — deterministic, provenanced |

---

## 8. Remaining risks

**Requires human mathematical review.** The three derivations in §3 are self-contained and
checkable, but a co-author should confirm the Proposition 1 rescoping specifically, since
it touches a formal statement. The bound itself is unchanged; only its index set moved.
The empty-ℬ convention is new and worth a second reading.

**Requires full experimental execution.** Everything in §7 except Proposition 1. Two items
are blocked *only* by this and cannot be resolved editorially: the η_meta 0.33 pp
discrepancy (Table S2 prose vs. table — whichever is wrong, the run logs decide), and the
withdrawn "higher per-round variance" claim, which needs per-round logs and a figure with
error bands.

**Requires different hardware or OS.** The campaign cannot run on this machine. Ray
workers cannot import PyTorch on Windows, and there is no CUDA device present — local
torch is CPU-only. **Linux or WSL2 with the pinned stack is required**, and the paper's
timing table additionally needs its stated hardware (RTX 3080, 32 GB, Ryzen 9 5900X) if
Table XII is to be comparable.

**Requires the exact paper environment.** Python 3.10.12 and torch 2.1.0. The pins are
correct in `requirements.txt` and `environment.yml`; the local venv simply is not built
from them. Since torch 2.1 → 2.11 spans several releases, results from a 2.11 environment
should not be compared to the paper's without re-baselining.

**Requires the real CIC-IoT-2023 dataset.** The pipeline is tested only against synthetic
CSVs matching the expected schema. The real capture has not been downloaded or processed
here, so the cross-dataset table is blocked on data as well as on compute.

**Left for manual decision.** `git stash@{0}` is **not** dropped. It has been fully
recovered — its one genuinely missing artifact (`results/_QUARANTINED_MOCK/tables/
proposition1_real.json`, the original fabricated `{"bound_ratio": 0.41}` file the
quarantine README documents) is now restored, and every other file it holds is a
superseded version of what is in the working tree. It is safe to drop, but that is a
deliberate call for you to make, not a side effect of this pass. No destructive git
command was run. The six 0-byte figure PDFs are left in place, now correctly flagged by
`make check-results` rather than silently reported as present.

---

## 9. Final repository readiness

### READY FOR FINAL HUMAN REVIEW

Not "ready for the full experimental campaign" — and the gap is one decision, not one
defect.

**What supports readiness.** The implementation now matches the methodology on every point
checked: both warmup schedules reproduce the paper's closed forms exactly and are driven
by one authoritative parameter; the straight-through estimator is implemented and verified
against closed-form gradients; every stage the overhead table reports is instrumented on
the live path; |D_val| = 2000 is genuinely used downstream; all twelve named experiments
are reachable from documented commands, with the leakage-free gap closed. 212 tests pass
with zero regressions, the only failures being the documented Windows platform limitation.
Both documents compile. Nine of twelve review markers are closed with evidence, and the
three that remain are correctly blocked on data this pass was not permitted to generate.
No result was fabricated, no expected value hardcoded, no test weakened.

**Why not "ready for the campaign".** Two corrections in this pass — enabling the
annealing and implementing the STE — change what the algorithm computes. That is the right
outcome scientifically, but it means the manuscript currently describes an algorithm no
reported number was produced by. Before spending the compute, a human needs to confirm the
intended reading on three points:

1. that the annealing genuinely should be on by default (the paper says so; the code said
   otherwise for the entire history of these results);
2. that the STE is the intended optimizer behaviour rather than the paper's wording being
   aspirational;
3. that the Proposition 1 rescoping is the intended framing.

Each is a methodological commitment, and the campaign is expensive enough that getting them
wrong is costly.

Once those three are confirmed, the criterion is met: the implementation is aligned with
the methodology and experimental protocol the paper describes, and the only remaining work
is the full-scale runs needed to regenerate the tables and figures, plus the
environment-specific timing measurements.
