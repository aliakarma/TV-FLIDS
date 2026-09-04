# `colab/` — reproducible Colab execution of the TV-FLIDS campaign

Run the full experimental campaign on a Colab T4, resumably, auditably, and
without inventing a single number.

This directory is an **orchestration layer**. It reimplements no science: every
experiment is executed by the repository's own runners under `experiments/`,
and every table and figure is produced by the repository's own generators under
`scripts/`.

---

## Contents

| File | Role |
| --- | --- |
| `TV_FLIDS_full_campaign.ipynb` | the notebook — run it top to bottom |
| `campaign_inventory.py` | **the single source of truth**: every cell, its identity, its output path, its schema |
| `generate_expected_results.py` | emits the `expected_results/` contract tree |
| `parallel_runner.py` | persistent queue + bounded parallel execution |
| `run_campaign.py` | one entry point for every step |
| `finalize_campaign.py` | writes `campaign_results/final_manifest.json` |
| `validate_campaign.py` | expected vs. actual, integrity, provenance, statistics |
| `check_colab_environment.py` | environment report + hard compatibility gate |
| `benchmark_concurrency.py` | measures 1 vs 2 workers, recommends the stable one |
| `smoke_test.py` | tiny pre-campaign run + 11 assertions on the artifact |
| `setup_colab.sh` | installs the pinned stack |
| `config_colab.yaml` | orchestration settings (never scientific values) |

---

## 1. How to open the notebook

**From GitHub:**

```
https://colab.research.google.com/github/aliakarma/TV-FLIDS/blob/main/colab/TV_FLIDS_full_campaign.ipynb
```

**Or:** colab.research.google.com → File → Upload notebook →
`colab/TV_FLIDS_full_campaign.ipynb`.

Then **Runtime → Change runtime type → Hardware accelerator: T4 GPU**. Do this
before running anything; Section 1 refuses to proceed without a GPU when you
pass `--require-gpu`.

---

## 2. The minimal sequence

```bash
# once per session
bash colab/setup_colab.sh
python colab/check_colab_environment.py --require-gpu
python colab/run_campaign.py link

# once, before committing to the campaign
python colab/run_campaign.py smoke
python colab/run_campaign.py benchmark          # then set max_parallel_jobs

# the campaign — re-run any of these after a disconnect
python colab/run_campaign.py run --priority 1   # main paper
python colab/run_campaign.py run --priority 2   # supporting analysis
python colab/run_campaign.py run --priority 3   # supplementary

# after each phase family completes
python colab/run_campaign.py finalize
python colab/run_campaign.py validate
python colab/run_campaign.py artifacts
```

---

## 3. How to configure Drive

Colab sessions end without warning, so put the campaign on Drive.

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
python colab/run_campaign.py link
```

`link` creates `<drive_root>/campaign_results/` and makes `<repo>/results` a
**symlink** to it.

**Why a symlink.** The repository's runners and `scripts/generate_manuscript_tables.py`
both hardcode a `results/...` prefix. Rather than fork that path handling —
which would mean touching scientific code — `results` is redirected. The
runners write `results/logs/comparison/...` unchanged; the bytes land at
`campaign_results/logs/comparison/...`, which is exactly the relative path
`expected_results/` mirrors.

`link` **refuses** to replace a non-empty real `results/` directory. If you
have local results to keep, merge them first:

```bash
cp -a results/. /content/drive/MyDrive/TV-FLIDS-Campaign/campaign_results/
rm -rf results
python colab/run_campaign.py link
```

Configure the location in `config_colab.yaml`:

```yaml
persistence:
  mount_drive: true
  drive_root: /content/drive/MyDrive/TV-FLIDS-Campaign
  campaign_dir: campaign_results
  fallback_root: /content/campaign_results   # session-local; LOST on disconnect
```

Without Drive, `link` says loudly that storage is session-local. Use that only
for throwaway tests.

---

## 4. How to select concurrency

**Measure it. Do not assume it.**

```bash
python colab/run_campaign.py benchmark
```

This runs the same short TV-FLIDS jobs at concurrency 1 and 2 and records wall
time, throughput, peak GPU memory, GPU utilisation, host RAM and failures. It
recommends the highest-throughput mode that had **zero** failures — a mode with
any failure is never recommended, however fast it looked. The objective is
maximum *stable* completed cells per hour.

Then adopt it:

```yaml
execution:
  max_parallel_jobs: 2      # whatever the benchmark said
  sim_client_cpus: 1
  sim_client_gpus: 0.1      # Ray fraction per simulated client actor
```

or per-invocation: `python colab/run_campaign.py run --workers 2`.

**The default is 1.** It stays 1 until a measurement on the actual T4 says
otherwise. Two things the benchmark prints every time, and you should believe:

- **Short jobs flatter parallelism.** A benchmark job runs a handful of rounds,
  so Ray/Flower start-up is a large share of its wall time. A 100-round
  campaign cell amortises that start-up over ~50× more compute, so the real
  gain from concurrency is *smaller* than the benchmark table suggests.
- **A CPU-only benchmark measures CPU contention only.** If it reports 0 MiB of
  GPU memory, it has told you nothing about sharing one T4. Re-run it on the
  T4.

`sim_client_gpus` is the more useful dial on a single GPU: `0.1` lets ten client
actors share the T4 inside *one* worker, which parallelises the simulation
without paying for a second Ray cluster.

---

## 5. How to start and resume the campaign

Start:

```bash
python colab/run_campaign.py run --priority 1
```

**Resume: run exactly the same command again.** Nothing else is required.

- `TVFLIDS_RESUME=1` is exported for every job, so a cell whose own
  `experiment_log.json` is complete for the requested round count returns that
  run's stored summary instead of recomputing. It can only ever replay a
  genuine prior execution: it reads the artifact that run wrote, computes
  nothing, and refuses anything short of a complete log.
- The queue at `results/_campaign/colab/queue.json` is written atomically after
  every state change, so a disconnect leaves it readable.
- On restart, `complete` units are skipped, `running` units are re-queued
  (their process is gone), and `failed` units are retried up to
  `max_attempts`.
- **No completed result is ever overwritten.**

Useful flags:

```bash
python colab/run_campaign.py run --dry-run              # print the commands
python colab/run_campaign.py run --phase A_main_comparison
python colab/run_campaign.py run --limit 4              # a timed probe
python colab/run_campaign.py run --retry-failed
python colab/run_campaign.py run --skip-aggregation     # stage 1 only
python colab/run_campaign.py status                     # queue snapshot
```

### One campaign process per results root

**Do not run two campaign processes against the same results root.** The queue
serialises work *within* one process, but nothing stops a second process — a
second Colab session, or a forgotten background job — from picking up the same
pending cell and writing the same `log_dir`. Two runs interleaving in one cell
directory produce a corrupt artifact.

If you need a second session, either point it at a different results root or
restrict it to disjoint phases with `--phase`. `status` is always safe to run
concurrently: it only reads.

The scoping flags are honoured on a resumed queue as well, so
`run --phase X` will never quietly widen itself to everything the queue has
ever seen.

---

## 6. How results are stored

```
campaign_results/                       (on Drive; <repo>/results points here)
├── logs/
│   ├── comparison/<strategy>_<attack>_seed<S>/
│   │   ├── config.json                 written at logger init
│   │   ├── experiment_log.json         101 round entries + summary + provenance
│   │   └── final_predictions.npz
│   ├── comparison_nslkdd_leakage_free/…
│   ├── ablation_<arm>_<seed>/…
│   ├── noniid_sweep/<strategy>_noniid_alpha<A>_seed<S>/…
│   ├── multi_attack_matrix/<strategy>_<attack>_seed<S>/…
│   ├── extended_significance/<strategy>_<attack>_seed<S>/…
│   ├── hyperparam_sweep_baseline/<sweep>_<value>_seed<S>/…
│   ├── hyperparam_sweep_tvflids/<factor>_<value>_seed<S>/…
│   └── ratio_sweep/<method>_ratio<R>_seed<S>/…
├── tables/                             the aggregate artifact per phase
│   └── _extra_baselines/               Supp. Table S3
├── figures/
├── statistics/
├── final_manifest.json                 what genuinely exists
└── _campaign/
    ├── colab/queue.json                the persistent queue
    ├── colab/job_logs/                 one log per unit
    ├── colab/config_snapshots/         config bytes + SHA-256 per unit
    └── validation_report.json
```

A cell's **per-round log is not a separate file**: its 101 entries live in
`experiment_log.json` under `rounds[]` (round 0 is the pre-training evaluation
of the initial global model; rounds 1–100 are the FL rounds).

---

## 7. How seeds are mapped

Primary seeds: **42, 123, 456, 789, 1337**.
Extension seeds (Section VIII-B only): **+ 2024, 31415, 8080, 555, 999**.

Every cell has a stable identity:

```
<phase>/<strategy>/<attack>/<dataset>/<protocol>/seed<seed>[/<arm>]
```

and each seed gets its own directory, its own `config.json`, its own config
hash, its own provenance block and its own log. To see the whole map:

```bash
python colab/run_campaign.py report
```

```
Experiment                   Seed Status       Config Hash    Rounds Output
N_clean_baseline              123 complete     a310f1d4bcad      101 logs/multi_attack_matrix/tvflids_no_attack_seed123
N_clean_baseline             1337 incomplete   -                   - logs/multi_attack_matrix/tvflids_no_attack_seed1337
```

To go the other way — from a manuscript table to the cells behind it —
read `expected_results/by_paper_artifact.json`.

---

## 8. How validation works

```bash
python colab/run_campaign.py validate
# or directly, with options:
python colab/validate_campaign.py --results-root campaign_results \
       --priority 1 --table --json report.json
```

### The expected/actual mapping

`expected_results/` is path-for-path isomorphic with `campaign_results/`:

```
expected_results/logs/comparison/fedavg_label_flip_30_seed42/experiment_log.json   <- contract
campaign_results/logs/comparison/fedavg_label_flip_30_seed42/experiment_log.json   <- result
```

Same directory names, same filenames. The difference is the content: the
expected file declares required schema, required provenance fields, valid
metric ranges and completion criteria. It declares **no expected numerical
outcome**.

This isomorphism holds **by construction**: the queue, the contract tree and
the validator all derive from `campaign_inventory.py`, so there is no second
place where a path could be spelled differently. `campaign_inventory.py verify`
additionally imports the repository's real runner constants and asserts every
grid matches — a mismatch is a hard error.

### What it detects

| Failure | How |
| --- | --- |
| **missing seeds** | every condition is grouped by (phase, strategy, attack, protocol, arm, alpha) and each must carry every declared seed |
| **missing rounds** | `rounds[]` must have exactly 101 entries indexed 0..100, with no gap and no duplicate |
| **wrong configurations** | each cell's `config` must match its declared identity field for field; the seed must be in the declared seed set |
| **mock artifacts** | any `mockhash` fingerprint; a `_config_hash` that is empty, null or a placeholder; any `expected_results` contract file sitting at a result path |
| **quarantined output** | `results/_QUARANTINED_MOCK/` is skipped as the intended forensic archive, but no live cell or aggregate path may resolve inside it |
| **invalid statistics** | p-values outside `[0,1]`; negative standard deviations; an `n_seeds` that is not the number of seeds actually present; ASR compared in the wrong direction |
| **malformed figures** | a zero-byte PDF, or a PDF without a `%PDF-` header |
| **stale outputs** | a generated `.tex` that still carries provisional/placeholder content, or carries the generated banner *and* placeholder text |
| **partial aggregates** | a phase whose cells are all complete but whose aggregate artifact is missing; conversely, aggregation is *skipped* while any cell is incomplete, so a partial artifact is never written |
| **lying manifests** | any cell `final_manifest.json` calls `complete` is re-checked on disk |

Numerical validity is checked only against **mathematically valid bounds** — a
probability lies in `[0,1]`, a σ is non-negative, `α+β+γ = 1` — plus structural
counts fixed by the campaign definition. Never against an expected measurement.

### Historical values

`expected_results/historical_reference/` exists so that previously published
numbers have somewhere to live **outside** the validation path. Everything in
it is labelled **HISTORICAL — NON-TARGET — DO NOT USE FOR VALIDATION**, and the
validator refuses it as a comparison source. A new result that disagrees with a
historical value is not thereby wrong; the historical value is not evidence.

---

## 9. How to regenerate tables and figures

```bash
python colab/run_campaign.py artifacts
```

which runs:

```bash
python scripts/generate_manuscript_tables.py
python scripts/generate_manuscript_figures.py --results-root results
# then the coverage check (non-zero exit = something is still unbacked)
python scripts/generate_manuscript_tables.py  --check
python scripts/generate_manuscript_figures.py --results-root results --check
```

Both generators **skip** any table or figure whose backing artifact does not
exist, rather than inventing it, so running this early is safe — it simply
produces less.

**Do not edit the manuscript as individual cells finish.** The order is:

1. finish an experiment family;
2. validate every cell in it;
3. aggregate;
4. validate the statistics;
5. regenerate that family's tables and figures;
6. only then update the corresponding manuscript section.

Partial regeneration produces an inconsistent manuscript.

---

## 10. How to verify the final campaign

```bash
python colab/run_campaign.py finalize
python colab/validate_campaign.py --results-root campaign_results --strict \
       --json campaign_results/_campaign/validation_report.json
python scripts/check_results.py
python scripts/check_manuscript.py
```

`--strict` treats warnings as failures too (a dirty git tree behind a result, a
still-provisional table, an unbacked figure). Use it for the pre-submission
check.

---

## 11. The campaign

**580 executable cells**, 100 rounds each.

| Phase | Prio | Cells | Paper artifact | Aggregate artifact |
| --- | ---: | ---: | --- | --- |
| `A_main_comparison` | 1 | 40 | Table V; Figures 2–4 | `tables/full_comparison_results.json` |
| `C_ablation` | 1 | 35 | Table VII | `tables/ablation_results.json` |
| `D_noniid_sweep` | 1 | 30 | Table VIII | `tables/noniid_sweep_results.json` |
| `N_clean_baseline` | 1 | 5 | Table IX (no-attack row) | `tables/multi_attack_matrix_results_no_attack.json` |
| `E_multi_attack` | 1 | 120 | Tables IX–X | `tables/multi_attack_matrix_results.json` |
| `F_adaptive_attacks` | 1 | 30 | Table XI | `tables/multi_attack_matrix_results_ack1_evasion_30_ack2_coalition_30.json` |
| `R_ratio_sweep` | 1 | 140 | Figure 5 | `tables/ratio_sweep_results.json` |
| `B_leakage_free` | 2 | 40 | Table VI | `tables/full_comparison_results_nslkdd_leakage_free.json` |
| `G_extended_significance` | 2 | 20 | Section VIII-B | `tables/extended_significance_results.json` |
| `H_hp_sweep_baseline` | 3 | 40 | Supp. Table S1 | `tables/hyperparam_sweep_baseline_results.json` |
| `I_hp_sweep_tvflids` | 3 | 70 | Supp. Table S2 | `tables/hyperparam_sweep_tvflids_results.json` |
| `J_bucketing_deepsight` | 3 | 10 | Supp. Table S3 | `tables/_extra_baselines/full_comparison_results.json` |

**Derived** (no cells of their own): `L_overhead` (Table XII, from the
`compute_overhead_ms` / `comm_overhead_pct` / `model_params` fields of the
phase-A summaries) and the Wilcoxon / Cohen's *d* columns of Tables V–XI.

**Blocked:** `K_ciciot2023` (Supp. Table S4, Supp. Figure S1) — the
CIC-IoT-2023 dataset is not in the repository. The phase is enumerated and
reported as blocked, and produces no queue entries. It is never faked.

---

## 12. Two execution modes, and why

`parallel_runner.py` runs two stages.

**Stage 1 — cell production (parallel, bounded).**

- **`cell` phases** (A, B, D, E, F, G, J, N): a cell is a plain
  `experiments/run_experiment.py` invocation, so each cell is its own job.
  Maximum parallelism, and a crash loses at most one cell. The per-cell command
  is built by `scripts/run_campaign_cells.py::build_cmd` — *imported*, not
  copied, so the invocation can never drift from the repository's own.
- **`shard` phases** (C ablation, H/I hyperparameter grids, R ratio sweep):
  these runners write a temp YAML per cell to apply trust overrides, grid
  values or adversarial ratios. Those configs are **scientific content**, so
  this layer refuses to reproduce them. It shards the family runner instead,
  along axes the CLI already exposes (`--seeds`, `--methods`), each shard
  writing its aggregate to a throwaway directory.

**Stage 2 — aggregation (sequential, cheap).** Each phase's family runner is
invoked once over the **full** grid with `TVFLIDS_RESUME=1` and the real
`--output`. Every cell is already on disk, so the runner reuses each stored
summary and does only what it alone should do: aggregate, run the statistics,
write the phase artifact. A phase with any incomplete cell is **skipped**, so a
partial artifact is never written.

Nothing in either stage computes, adjusts, averages or synthesises a metric.

---

## 13. Estimating runtime

The honest estimate comes from *your* measurement, not from an assumption:

```bash
python colab/run_campaign.py run --priority 1 --limit 4 --skip-aggregation
python colab/run_campaign.py status
```

Then: `580 cells × (measured minutes per cell) ÷ (workers)`.

For reference, and **explicitly not a target or a prediction**: the previous
local CPU host (12 logical cores, no CUDA) measured **≈ 21.5 min per 100-round
cell**. At that rate 580 cells is ≈ 208 worker-hours. Whether a T4 beats it,
and by how much, is what the benchmark and the `--limit` probe are for. Colab
sessions are time-limited, so plan on **several sessions** and rely on
resumability rather than on one long run.

---

## 14. Scientific-integrity guarantees

- No expected accuracy, F1, ASR, p-value, improvement percentage, ablation
  delta or hyperparameter result exists anywhere in `expected_results/`.
- `expected_results/` is a contract for **structure and completeness**: paths,
  filenames, schemas, required provenance fields, seed coverage, output counts,
  valid metric ranges, consistency invariants.
- Historical manuscript values are **not** targets, live outside the validation
  path, and are labelled non-target.
- Every result carries provenance: git commit and dirty state, config hash,
  seed, dataset, package versions, hardware, timestamps.
- The config bytes that each unit saw are snapshotted with their SHA-256.
- Resume can only ever replay a genuine prior execution of the same cell.
- Aggregation is skipped rather than run partially.
- The workflow is enforced in one direction only:

```
Colab notebook -> real experiment -> seed-specific output -> real provenance
   -> validated result -> table/figure -> paper
```

never

```
paper number -> expected number -> modified code -> fake result
```
