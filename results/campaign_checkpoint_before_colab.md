# Campaign Checkpoint — CPU lanes stopped before migration to Colab

**Written:** 2026-09-04T12:33Z (15:33 local, UTC+3)
**Reason for stop:** the campaign is being moved to Google Colab (T4 GPU, ~52 GB
system RAM). This file is the hand-off record of the local CPU campaign.

**Repository state at stop**

| Field | Value |
| --- | --- |
| Commit | `0fc75a3817d2d72ddc5e937eeb341d74c4100821` |
| Branch | `main` |
| Working tree | clean (0 dirty paths) |
| Host | Windows 11 + WSL2 (Ubuntu), 12 logical CPUs, ~9 GB RAM visible to WSL |
| Python | `~/miniconda3/envs/tvflids_cpu/bin/python` (3.10) |
| Torch build | CPU-only (no CUDA on this machine's WSL2 path) |

---

## 1. Lane inventory at the moment of the stop

Both lanes were `bash scripts/campaign_lane.sh` processes running **inside WSL2**
(not visible to Windows `Get-Process`; found via `wsl -e bash -c ps`).

### Lane 1

| Field | Value |
| --- | --- |
| Lane shell PID / PGID | 286584 / 286584 |
| Runner PID | 286594 |
| Command | `experiments/run_full_comparison.py --strategies fedavg krum trimmed_mean fltrust foolsgold flame rfa tvflids --attack label_flip_30 --seeds 42 123 456 789 1337 --rounds 100` |
| Phase queue | `A_main_comparison C_ablation D_noniid_sweep G_extended_significance B_leakage_free` |
| Phase at stop | `A_main_comparison` (started 2026-09-04T10:59:01Z) |
| Uptime at stop | 1 h 28 m |
| Cell in flight | `fedavg / label_flip_30 / seed 1337`, **round 13 of 100** |
| Disposition | in-flight cell **discarded** (13 % complete); will be re-run |
| Stop method | `SIGINT` to PGID (absorbed by Ray/Flower) then `SIGTERM` to lane shell, then to runner. Exited cleanly. |

### Lane 2

| Field | Value |
| --- | --- |
| Lane shell PID / PGID | 1473559 / 1473559 |
| Runner PID | 1473569 |
| Command | `experiments/run_multi_attack_matrix.py --strategies tvflids --attacks no_attack --seeds 42 123 456 789 1337 --rounds 100` |
| Phase queue | `N_clean_baseline E_multi_attack F_adaptive_attacks R_ratio_sweep H_hp_sweep_baseline I_hp_sweep_tvflids J_bucketing_deepsight` |
| Phase at stop | `N_clean_baseline` (restarted 2026-09-04T12:06:25Z) |
| Cell in flight | `tvflids / no_attack / seed 789`, round 84 of 100 when the stop was requested |
| Disposition | **cell allowed to finish.** A watcher polled for a complete 101-entry `experiment_log.json` and stopped the lane the instant it appeared. The seed-789 cell is preserved as a genuine result. In the seconds between that write and the signal landing, the runner did open seed 1337 and write its `config.json` stub (see §2). |
| Stop method | watcher then `SIGTERM` lane shell (so no next phase starts) then `SIGTERM` runner. Exited cleanly. |

One orphaned Ray dashboard agent (PID 1485571) survived its dead driver and was
terminated. It produces no scientific output. No Ray, raylet, GCS or actor
process remains.

**A separate long-running Python training job on this machine belongs to a
different repository** (`Desktop/Current Research/ICLR/Paper 1/Safe-Lie`,
`scripts/train.py`). It was **not** touched.

---

## 2. Completed cells preserved (8 of 8 verified genuine)

Every cell below carries 101 round entries (round 0 baseline plus rounds 1-100),
a real per-cell configuration hash, and a full provenance block. No `mockhash`.

| Log directory | Rounds | Strategy | Attack | Seed | Config hash | final_accuracy | Elapsed |
| --- | ---: | --- | --- | ---: | --- | ---: | ---: |
| `results/logs/comparison/fedavg_label_flip_30_seed42` | 101 | fedavg | label_flip_30 | 42 | `a3d2d64a169b` | 0.430758 | 20.4 min |
| `results/logs/comparison/fedavg_label_flip_30_seed123` | 101 | fedavg | label_flip_30 | 123 | `2b74ec3272a4` | 0.430758 | 14.5 min |
| `results/logs/comparison/fedavg_label_flip_30_seed456` | 101 | fedavg | label_flip_30 | 456 | `61fdedb0c971` | 0.430758 | 25.5 min |
| `results/logs/comparison/fedavg_label_flip_30_seed789` | 101 | fedavg | label_flip_30 | 789 | `62687f7101b8` | 0.430758 | 23.1 min |
| `results/logs/multi_attack_matrix/tvflids_no_attack_seed42` | 101 | tvflids | no_attack | 42 | `b9f0a9b02994` | 0.633162 | 20.7 min |
| `results/logs/multi_attack_matrix/tvflids_no_attack_seed123` | 101 | tvflids | no_attack | 123 | `a310f1d4bcad` | 0.682754 | 15.1 min |
| `results/logs/multi_attack_matrix/tvflids_no_attack_seed456` | 101 | tvflids | no_attack | 456 | `fe331d8f1718` | 0.697614 | 27.8 min |
| `results/logs/multi_attack_matrix/tvflids_no_attack_seed789` | 101 | tvflids | no_attack | 789 | `5acb259042d8` | 0.822436 | 24.5 min |

Each directory holds `config.json`, `experiment_log.json`, `final_predictions.npz`.

### Partial cells preserved (2)

Neither in-flight cell wrote an `experiment_log.json` — the logger writes that
at end of run — but each had already written its `config.json` stub, so each
left a directory behind:

| Log directory | Contents | Interpretation |
| --- | --- | --- |
| `results/logs/comparison/fedavg_label_flip_30_seed1337` | `config.json` (344 B) only | lane 1's in-flight cell, stopped at round 13 |
| `results/logs/multi_attack_matrix/tvflids_no_attack_seed1337` | `config.json` (341 B) only | lane 2 opened this cell at 12:31Z, immediately after seed 789 completed and moments before the stop signal landed |

Both are **preserved, not deleted**, and both are correctly classified as
incomplete by `colab/finalize_campaign.py` (status `failed`, reason
"interrupted: config.json written, no experiment_log.json") and by
`colab/validate_campaign.py`. They are harmless to the resume path: the resume
guard requires a *complete* `experiment_log.json`, so each cell simply re-runs.

**Corrupt or zero-round `experiment_log.json` files on disk: none.**

> The `final_accuracy` values above are recorded for provenance only. They are a
> record of what these 8 CPU cells produced, **not** targets for the Colab run,
> and they are not written into `expected_results/`.

---

## 3. Incomplete / not-yet-started work

| Phase | Paper artifact | State at stop |
| --- | --- | --- |
| `A_main_comparison` | Table V, Figures 2-4 | 4 of 40 cells done (fedavg only, seeds 42/123/456/789); seed 1337 is a `config.json`-only stub |
| `N_clean_baseline` | Table IX "no attack" row | 4 of 5 cells done (seeds 42/123/456/789); seed 1337 is a `config.json`-only stub |
| `B_leakage_free` | Table VI | not started |
| `C_ablation` | Table VII | not started |
| `D_noniid_sweep` | Table VIII | not started |
| `E_multi_attack` | Tables IX-X | not started |
| `F_adaptive_attacks` | Table XI | not started |
| `G_extended_significance` | Section VIII-B (10 seeds) | not started |
| `H_hp_sweep_baseline` | Supp. Table S1 | not started |
| `I_hp_sweep_tvflids` | Supp. Table S2 | not started |
| `J_bucketing_deepsight` | Supp. Table S3 | not started |
| `K_ciciot2023` | Supp. Table S4, Figure S1 | not started (needs the real dataset) |
| `R_ratio_sweep` | Figure 5 | not started |
| `L_overhead` | Table XII | derived from run summaries; needs the runs above |

**Failed cells: none.** Neither `lane1.status` nor `lane2.status` was ever
written, i.e. no phase reached its END marker, so no phase recorded a non-zero
exit code. Nothing failed — the campaign was simply nowhere near finished.

No aggregate table artifact was produced: `results/tables/` contains only
`proposition1_real.json` (deterministic theory output, predates this campaign).

---

## 4. Files preserved (nothing deleted)

| Path | Kept |
| --- | --- |
| `results/logs/comparison/**` | 4 complete cells + 1 `config.json` stub |
| `results/logs/multi_attack_matrix/**` | 4 complete cells + 1 `config.json` stub |
| `results/_campaign/manifest.json` | campaign `init` manifest (14 KB) |
| `results/_campaign_logs/lane1.log` | 6.34 MB full lane-1 transcript |
| `results/_campaign_logs/lane2.log` | 6.44 MB full lane-2 transcript |
| `results/_campaign_logs/lane1.boot`, `lane2.boot` | lane launch banners |
| `results/_campaign_logs/stop_lane2.watch` | this stop's watcher trace |
| `results/_campaign_logs/bench*/`, `det*/`, `smoke_fedavg/` | earlier probe and benchmark output |
| `results/_campaign_logs/*.sh`, `*.log`, `*.out` | every probe, install and benchmark script/log |
| `results/figures/*.pdf` | 9 figures from earlier cells |
| `results/tables/proposition1_real.json` | deterministic theory artifact |
| `results/_QUARANTINED_MOCK/**` | forensic evidence, untouched |

No file under `results/` was deleted, truncated or overwritten during this stop.

---

## 5. Measured CPU throughput (the baseline the Colab run must beat)

8 complete cells consumed **171.6 min of in-cell wall clock** (2.86 h), i.e.
**about 21.5 min per 100-round cell** at the local configuration of 2 lanes with
6 Ray actors each over 12 logical CPUs (`TVFLIDS_SIM_CLIENT_CPUS=2`).

Wall-clock elapsed for the two lanes was 10:59Z to 12:33Z, about 4.6 h, during
which the machine also ran an unrelated training job, so effective campaign
throughput was materially below the per-cell figure.

This is a **measured historical CPU reference**, not a target.

---

## 6. Resumability guarantee

`scripts/campaign_lane.sh` exports `TVFLIDS_RESUME=1`, and
`experiments/run_experiment.py::_load_completed_run` returns a stored summary
for any cell whose own `experiment_log.json` is complete for the requested round
count. The 8 preserved cells are therefore **skipped, not recomputed**, by any
future run at the same round count and log path — on this machine or on Colab,
provided the same `results/logs/...` relative layout is used.
