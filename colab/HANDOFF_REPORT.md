# TV-FLIDS — CPU campaign stopped, Colab pipeline built

**2026-09-04** · commit `0fc75a3` · branch `main`

---

## 1. Current CPU campaign

Both lanes were `bash scripts/campaign_lane.sh` processes running **inside
WSL2** — invisible to Windows `Get-Process`, which is why they had to be found
with `wsl -e bash -c ps`.

| | Lane 1 | Lane 2 |
| --- | --- | --- |
| Shell / runner PID | 286584 / 286594 | 1473559 / 1473569 |
| Phase at stop | `A_main_comparison` | `N_clean_baseline` |
| Cell in flight | `fedavg / label_flip_30 / seed 1337`, round **13** of 100 | `tvflids / no_attack / seed 789`, round **84** of 100 |
| Disposition | in-flight cell discarded (13 % done) | **cell allowed to finish** |
| Stopped | SIGINT (absorbed by Ray) → SIGTERM shell → SIGTERM runner | watcher → SIGTERM shell → SIGTERM runner |
| Exit | clean | clean |

Lane 2 was 16 rounds from completing a genuine cell, so a watcher polled for a
complete 101-entry `experiment_log.json` and stopped the lane the instant it
appeared — preserving that cell rather than discarding 20 minutes of real work.
One orphaned Ray dashboard agent (PID 1485571) was terminated. No Ray, raylet,
GCS or actor process remains.

A separate long-running Python job on this machine belongs to a **different
repository** (`Desktop/Current Research/ICLR/Paper 1/Safe-Lie`). It was not
touched.

### Completed cells preserved — 8, all verified genuine

101 round entries each, a real per-cell config hash, a full provenance block,
no `mockhash`.

| Log directory | Seed | Config hash | Elapsed |
| --- | ---: | --- | ---: |
| `logs/comparison/fedavg_label_flip_30_seed42` | 42 | `a3d2d64a169b` | 20.4 min |
| `logs/comparison/fedavg_label_flip_30_seed123` | 123 | `2b74ec3272a4` | 14.5 min |
| `logs/comparison/fedavg_label_flip_30_seed456` | 456 | `61fdedb0c971` | 25.5 min |
| `logs/comparison/fedavg_label_flip_30_seed789` | 789 | `62687f7101b8` | 23.1 min |
| `logs/multi_attack_matrix/tvflids_no_attack_seed42` | 42 | `b9f0a9b02994` | 20.7 min |
| `logs/multi_attack_matrix/tvflids_no_attack_seed123` | 123 | `a310f1d4bcad` | 15.1 min |
| `logs/multi_attack_matrix/tvflids_no_attack_seed456` | 456 | `fe331d8f1718` | 27.8 min |
| `logs/multi_attack_matrix/tvflids_no_attack_seed789` | 789 | `5acb259042d8` | 24.5 min |

### Partial cells preserved — 2

Both in-flight cells had written their `config.json` stub before the logger
writes `experiment_log.json`, so each left a directory:
`comparison/fedavg_label_flip_30_seed1337` and
`multi_attack_matrix/tvflids_no_attack_seed1337`. Both are kept, and both are
correctly classified as incomplete by the new tooling. They are harmless to
resume: the resume guard requires a *complete* log, so each cell simply re-runs.

*(My first draft of the checkpoint report said the in-flight cells left nothing
behind. The finalizer found these two stubs; the report is corrected.)*

### Failed cells — none

Neither `lane1.status` nor `lane2.status` was ever written, i.e. no phase
reached its END marker, so no phase recorded a non-zero exit. Nothing failed;
the campaign was simply nowhere near finished. No aggregate table artifact was
produced.

**Nothing under `results/` was deleted, truncated or overwritten.**

**Checkpoint:** `results/campaign_checkpoint_before_colab.md`

---

## 2. Colab system

| Path | Role |
| --- | --- |
| `colab/TV_FLIDS_full_campaign.ipynb` | the notebook — 44 cells, 11 sections |
| `colab/campaign_inventory.py` | **single source of truth**: 580 cells, identity, path, schema |
| `colab/generate_expected_results.py` | emits the `expected_results/` contract tree |
| `colab/check_isomorphism.py` | proves expected ↔ actual mapping + no invented targets |
| `colab/parallel_runner.py` | persistent queue, bounded parallel execution |
| `colab/run_campaign.py` | one entry point for every step |
| `colab/finalize_campaign.py` | writes `campaign_results/final_manifest.json` |
| `colab/validate_campaign.py` | completeness, integrity, provenance, statistics, manuscript |
| `colab/check_colab_environment.py` | environment report + hard compatibility gate |
| `colab/benchmark_concurrency.py` | measures concurrency, recommends the stable one |
| `colab/smoke_test.py` | tiny run + 12 assertions |
| `colab/setup_colab.sh` | installs the pinned stack |
| `colab/config_colab.yaml` | orchestration settings only |
| `colab/README.md` | full operating manual |

**Installation:** `bash colab/setup_colab.sh` installs the exact
`requirements.txt` pins, taking torch from the CUDA 12.1 wheel index because
the `+cpu` build cannot touch a T4. Nothing is silently upgraded; the final
report prints what actually landed. `2.1.0+cu121` is correctly recognised as
torch 2.1.0 — the local suffix names the build variant, not the release.

**Persistence:** Drive is mounted and `<repo>/results` becomes a **symlink** to
`<drive>/TV-FLIDS-Campaign/campaign_results`. The repository's runners and its
table generator both hardcode a `results/...` prefix; redirecting the path
rather than forking that handling means zero changes to scientific code, and
the bytes land at exactly the relative paths `expected_results/` mirrors.
`link` **refuses** to replace a non-empty real `results/` directory — verified
against the 8 preserved cells.

---

## 3. GPU benchmark

Run on **this CPU host**, as a harness self-test. `--rounds 2`, TV-FLIDS under
label-flip (the heaviest strategy).

| Concurrency | Runtime | min/job | GPU memory | Throughput | Failures | Recommended? |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 2.19 min | 2.19 | 0 MiB (no CUDA) | 27.45 jobs/h | 0 | |
| 2 | 2.11 min | 1.05 | 0 MiB (no CUDA) | 56.89 jobs/h | 0 | yes |

Host RAM peaked at 3.43 GB (1 worker) and 6.16 GB (2 workers).

**This does not predict the T4 answer, and the tool says so on every run:**

- **Short jobs flatter parallelism.** A 2-round job is dominated by Ray/Flower
  start-up, which is largely serial. A 100-round cell amortises that over ~50×
  more compute, so the real gain is *smaller* than this table suggests.
- **0 MiB of GPU memory means this measured CPU contention only.** One T4
  shared by two workers contends completely differently (one GPU, few vCPUs).

`config_colab.yaml` therefore ships `max_parallel_jobs: 1`. **Re-run
`python colab/run_campaign.py benchmark` on the T4 and adopt what it measures.**
On a single GPU the more useful dial is `sim_client_gpus: 0.1`, which lets ten
client actors share the T4 inside *one* worker.

---

## 4. Expected-results structure

`expected_results/<path>` ↔ `campaign_results/<path>` — same directory names,
same filenames:

```
expected_results/logs/comparison/fedavg_label_flip_30_seed42/experiment_log.json   <- contract
campaign_results/logs/comparison/fedavg_label_flip_30_seed42/experiment_log.json   <- result
```

The contract declares required schema, sibling files, round count and index
range, per-round keys, summary/config/provenance keys, mathematically valid
metric ranges, and completion criteria. It declares **no numerical outcome**.
Each carries `"__expected_contract__": true` and a `"__not_a_result__"` note.

This holds **by construction**: the queue, the contract tree and the validator
all derive from `campaign_inventory.py`, so there is no second place a path
could be spelled differently.

```
expected_results/
├── README.md
├── manifest_template.json          580 cells, status "pending"
├── by_paper_artifact.json          paper table/figure -> backing cells
├── logs/**/experiment_log.json     580 per-cell contracts
├── tables/*.json                   12 aggregate contracts
├── tables/_extra_baselines/        Supp. Table S3
├── figures/EXPECTED_FIGURES.json
├── statistics/EXPECTED_STATISTICS.json
└── historical_reference/README.md  HISTORICAL - NON-TARGET, empty of targets
```

Per-round logs are not a separate tree: a cell's 101 entries live inside its own
`experiment_log.json` under `rounds[]`, and the contract states the required
count, index range and keys.

**Verified:**

```
inventory declares      592 result path(s)
expected_results holds  592 contract file(s)
declared without contract  0
orphan contracts           0
predicted outcome values   0   <- none, as required
results outside contract   0
```

The "predicted outcome values" check walks all 596 contract JSON files and
fails on any number under a key naming an experimental outcome, exempting
`metric_ranges` (where `accuracy ∈ [0,1]` is a definition, not a prediction).

---

## 5. Experiment coverage

| Experiment | Expected cells | Queue created | Output schema | Ready |
| --- | ---: | --- | --- | --- |
| `A_main_comparison` — Table V, Figs 2–4 | 40 | 40 cell units | `tables/full_comparison_results.json` | yes |
| `C_ablation` — Table VII | 35 | 5 shard units | `tables/ablation_results.json` | yes |
| `D_noniid_sweep` — Table VIII | 30 | 30 cell units | `tables/noniid_sweep_results.json` | yes |
| `N_clean_baseline` — Table IX row | 5 | 5 cell units | `tables/multi_attack_matrix_results_no_attack.json` | yes |
| `E_multi_attack` — Tables IX–X | 120 | 120 cell units | `tables/multi_attack_matrix_results.json` | yes |
| `F_adaptive_attacks` — Table XI | 30 | 30 cell units | `..._ack1_evasion_30_ack2_coalition_30.json` | yes |
| `R_ratio_sweep` — Figure 5 | 140 | 20 shard units | `tables/ratio_sweep_results.json` | yes |
| `B_leakage_free` — Table VI | 40 | 40 cell units | `tables/full_comparison_results_nslkdd_leakage_free.json` | yes |
| `G_extended_significance` — §VIII-B | 20 | 20 cell units | `tables/extended_significance_results.json` | yes |
| `H_hp_sweep_baseline` — Table S1 | 40 | 5 shard units | `tables/hyperparam_sweep_baseline_results.json` | yes |
| `I_hp_sweep_tvflids` — Table S2 | 70 | 5 shard units | `tables/hyperparam_sweep_tvflids_results.json` | yes |
| `J_bucketing_deepsight` — Table S3 | 10 | 10 cell units | `tables/_extra_baselines/full_comparison_results.json` | yes |
| **Total** | **580** | **330 units** | | |
| `L_overhead` — Table XII | derived | — | from phase-A summaries | yes |
| `K_ciciot2023` — Table S4, Fig S1 | 30 | **0** | `tables/dataset_comparison_results.json` | **blocked** |

`K_ciciot2023` is blocked: `data/raw/CICIoT2023_{train,test}.csv` are absent. It
is enumerated and reported as blocked, never faked.

**Why 330 units for 580 cells.** Phases whose runner writes a temp YAML per cell
(ablation trust overrides, hyperparameter grids, ratio overrides) hold
*scientific content* in those configs, so the orchestrator refuses to reproduce
them. It shards the family runner instead, along axes the CLI already exposes
(`--seeds`, `--methods`), each shard writing to a throwaway output. Everything
else runs one cell per unit.

---

## 6. Smoke tests actually executed

Every one of these was run, not merely written.

| Test | Result |
| --- | --- |
| `campaign_inventory.py summary` | 580 cells, 580 distinct log dirs, 0 collisions |
| `campaign_inventory.py verify` (imports the real runner modules) | **OK — every grid matches the repository's own constants** |
| `generate_expected_results.py` | 598 files written |
| `check_isomorphism.py` | 592 ↔ 592, 0 orphans, 0 predicted outcomes, 0 stray results |
| No-invented-numbers scan | 596 contract files, **0** predicted accuracy/F1/ASR/p-value/delta |
| `check_colab_environment.py` (WSL) | exit 0; correctly flagged the `+cpu` torch build and the absent CIC-IoT dataset |
| `benchmark_concurrency.py --rounds 2 --modes 1 2` | completed; 3 jobs, 0 failures; recommended 2 with both caveats printed |
| `smoke_test.py --rounds 3 --clients 6` | **12 assertions, 0 failures** (11 pass, GPU skipped on CPU) |
| `finalize_campaign.py` against real results | 580 cells: 8 complete, 2 failed, 570 pending — matched disk exactly |
| `validate_campaign.py` against real results | correctly found the missing seed 1337, the dirty tree, absent aggregates, provisional tables |
| `parallel_runner.py --dry-run` (all phases, and `--phase`-scoped) | 330 units; commands and both stages render correctly |
| `run_campaign.py link` against non-empty `results/` | **refused**, listing all 8 entries and the merge procedure |
| `py_compile` + `--help` on all 10 modules | all pass |

### Smoke-test assertions (all on the produced artifact)

```
[PASS] run completed
[PASS] tiny config took effect (num_clients honoured)
[PASS] round log has rounds+1 entries indexed 0..N
[PASS] model trained (parameters present, metrics not frozen)
[PASS] attack executed (malicious clients selected and recorded)
[PASS] verification gate executed (decisions recorded)
[PASS] trust EMA executed (per-round trust telemetry present, in [0,1])
[PASS] STE / meta-gradient executed (alpha+beta+gamma == 1 each round)
[PASS] provenance complete and config hash is real
[SKIP] GPU genuinely used            <- no CUDA here; binding on the T4
[PASS] artifact matches the expected_results schema and metric ranges
[PASS] no mock or quarantined artifact consumed
```

### Defects these tests found and fixed

1. **`run_ablation.py` wrote a colon into its log directory path**
   (`ablation_A1:_No_Verification_42`). Legal on Linux, **illegal on Windows**
   and unsafe in Google Drive — the contract tree could not even be generated.
   Fixed to strip the colon, matching the sanitisation the *same function*
   already applied to its temp config filename. Safe: no ablation cell existed
   yet, only one site builds that path, and nothing reads it by name. A
   cross-platform path check now guards every log dir in `verify`.
2. **`Sampler._stop` shadowed `threading.Thread._stop()`**, which `join()`
   calls internally — the benchmark crashed with `'Event' object is not
   callable`. Renamed.
3. **The smoke test's client override silently did nothing.** It looked for
   `num_clients` in the wrong sections, so it ran 20 clients while reporting 6.
   Now it searches the whole config tree, **fails loudly** if the key is not
   found, and asserts the recorded `num_clients` matches what was requested.
4. **`--phase` / `--priority` were ignored on a resumed queue** — a scoped run
   would quietly widen to every unit the queue had ever seen. Execution and
   `--retry-failed` now intersect with the current selection.
5. **The validator flagged `results/_QUARANTINED_MOCK/` as a violation.** That
   directory is the *intended* forensic archive and is *expected* to match
   every fabrication fingerprint. Now skipped (as `scripts/check_results.py`
   does), with a separate assertion that no live cell or aggregate path
   resolves inside it.
6. **The validator's manifest cross-check false-positived under `--phase`**,
   judging cells it had not examined. Now scoped to what it actually validated.
7. **The smoke test failed on its own quarantine check.** All 45 hits were in
   `provenance.git_dirty_paths` — provenance *correctly* recording an untracked
   directory. Narrowed to input/output paths (`log_dir`, `script`, `argv`,
   `dataset`); negative control confirms a genuine violation is still caught.
8. **The validator treated `*_provisional.tex` as a defect.** Those are the
   repository's deliberate `\IfFileExists` fallbacks. Now reported as "still
   unbacked", while a *generated* body containing provisional text remains a
   hard error.

---

## 7. Validation

`python colab/validate_campaign.py --results-root campaign_results [--strict]`

| Detects | How |
| --- | --- |
| **missing seeds** | conditions grouped by (phase, strategy, attack, protocol, arm, alpha); each must carry every declared seed |
| **missing rounds** | `rounds[]` must be exactly 101 entries indexed 0..100, no gap, no duplicate |
| **wrong configurations** | each cell's `config` must match its declared identity field for field; seed must be in the declared set |
| **mock artifacts** | any `mockhash`; a `_config_hash` that is empty/null/placeholder; any contract file sitting at a result path |
| **quarantined output** | the archive is skipped as intended, but no live cell or aggregate path may resolve inside it |
| **invalid statistics** | p-values outside [0,1]; negative σ; `n_seeds` that is not the number actually present; ASR direction |
| **malformed figures** | zero-byte PDFs; PDFs without a `%PDF-` header |
| **stale outputs** | a *generated* `.tex` containing provisional text is an error; a labelled `_provisional.tex` still in service is a warning naming the missing generated body |
| **partial aggregates** | a phase with all cells complete but no aggregate is an error; conversely aggregation is *skipped* while any cell is incomplete, so a partial artifact is never written |
| **lying manifests** | every cell `final_manifest.json` calls `complete` is re-checked on disk |

Numerical validity is checked only against **mathematically valid bounds** and
structural counts — never against an expected measurement.

Historical values live in `expected_results/historical_reference/`, are labelled
**HISTORICAL — NON-TARGET — DO NOT USE FOR VALIDATION**, and are refused as a
comparison source.

---

## 8. How to run in Colab

```
https://colab.research.google.com/github/aliakarma/TV-FLIDS/blob/main/colab/TV_FLIDS_full_campaign.ipynb
```

Runtime → Change runtime type → **T4 GPU**. Then:

```bash
bash colab/setup_colab.sh
python colab/check_colab_environment.py --require-gpu
python colab/run_campaign.py link

python colab/check_isomorphism.py --results-root results --scaffold
python colab/run_campaign.py smoke
python colab/run_campaign.py benchmark          # then set max_parallel_jobs

python colab/run_campaign.py run --priority 1   # main paper
python colab/run_campaign.py run --priority 2
python colab/run_campaign.py run --priority 3

python colab/run_campaign.py finalize
python colab/run_campaign.py validate
python colab/run_campaign.py artifacts
```

**After a disconnect: re-run the same `run` command.** Completed cells are
skipped, `running` units are re-queued, failed units are retried.

**Do not run two campaign processes against one results root** — the queue
serialises within a process, not across them.

---

## 9. Estimated campaign time

**No T4 measurement exists yet**, so no honest T4 estimate can be given. What is
measured:

- **Local CPU host, 12 cores:** ≈ **21.5 min per 100-round cell** (from the 8
  genuine cells). At that rate 580 cells ≈ **208 worker-hours**.
- **Local CPU concurrency probe:** 2 workers roughly doubled throughput on
  2-round jobs — but with 0 MiB GPU involvement and start-up-dominated jobs, so
  it does not transfer.

Get the real number in about 20 minutes on the T4:

```bash
python colab/run_campaign.py run --priority 1 --limit 4 --skip-aggregation
python colab/run_campaign.py status          # per-unit elapsed_seconds
```

Then: `580 × (measured min/cell) ÷ workers`. Colab sessions are time-limited, so
plan on **several sessions** and rely on resumability rather than one long run.

I am not going to put a T4 number here that I have not measured.

---

## 10. Remaining issues

1. **`K_ciciot2023` is blocked** — `data/raw/CICIoT2023_{train,test}.csv` are
   absent, so Supplementary Table S4 and Figure S1 cannot be produced. Supply
   the dataset or drop those artifacts; they will not be faked.
2. **Concurrency is unmeasured on a T4.** The shipped default is the
   conservative `max_parallel_jobs: 1`. Run the benchmark on Colab first.
3. **Python version.** The pins target 3.10; Colab's default is usually newer.
   The campaign still runs and records the actual versions in every result's
   provenance — but the environment must not then be described as matching the
   paper's stated stack.
4. **Working tree is dirty** (the new `colab/` and `expected_results/` trees).
   Commit before starting, or every result records a dirty provenance.
5. **580 cells is a multi-session campaign** under any plausible per-cell time.
   This is a scale fact, not a defect, and is why priority order and
   resumability exist.

None of these blocks starting the campaign.

---

## 11. Final status

## **READY TO START FULL COLAB CAMPAIGN**

The pipeline is built, every component has been executed at least once, and the
twelve-assertion smoke test passes end to end on a real run. Two things to do on
the T4 before committing to the full campaign — both are notebook steps, both
take minutes:

1. `python colab/run_campaign.py smoke` — the GPU assertion is skipped on CPU
   and becomes binding there.
2. `python colab/run_campaign.py benchmark` — then set `max_parallel_jobs` from
   what it measures.

---

### The workflow this enforces

```
Colab notebook -> real experiment -> seed-specific output -> real provenance
   -> validated result -> table/figure -> paper
```

and never

```
paper number -> expected number -> modified code -> fake result
```

`expected_results/` is a contract for structure and completeness. It contains no
numerical truth, and a mechanical check over all 596 contract files confirms it.
