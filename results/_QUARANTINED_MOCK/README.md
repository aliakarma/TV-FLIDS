# QUARANTINED — NOT GENUINE EXPERIMENTAL OUTPUT

Everything under this directory was moved here during the forensic-audit
remediation pass (2026-09-03) because it is confirmed or flagged to be
**fabricated, mock, or unverifiable** output — not the result of a real
executed experiment. Nothing here may be cited as evidence for any paper
table, figure, or claim. Treat this directory as archival/forensic evidence
of the prior fabrication, not as a data source.

## What's here and why

| File / directory | Status | Evidence |
|---|---|---|
| `mock_phase7_results.py.DO_NOT_RUN` | **Confirmed mock generator.** Writes random noise around hardcoded target numbers directly to the paths real results occupy. Renamed with a `.DO_NOT_RUN` extension so it cannot be executed by path/import and cannot be mistaken for a results source. | Self-declared placeholder script (see its own docstring/body: `generate_mock_results()`). |
| `tables/ablation_results.json` | **Confirmed fabricated.** Byte-identical to `mock_phase7_results.py`'s hardcoded A1=0.82 … A5=0.88 with zero seed variance (impossible for a genuine 2-5 seed stochastic FL run). | Byte comparison, audit §2/§13. |
| `tables/proposition1_real.json` | **Confirmed fabricated.** Byte-identical to the mock generator's hardcoded `{bound_ratio: 0.41, observed_deviation: 0.05, theoretical_bound: 0.12}`. This is the direct source of the paper's headline "0.41" claim — it was never produced by `theory/proposition1_verification.py`. Running that script for real (deterministic, seed=42) gives ≈0.588. See `theory/proposition1_verification.py` and the remediation report for the corrected number. | Byte comparison + live re-execution, audit §2/§4/§13. |
| `tables/ratio_sweep_results.json` | **Confirmed fabricated (empty stub).** `{"tvflids": {}, "fedavg": {}}` — no data at all, despite backing Figure 5 of the paper. | Byte comparison, audit §4 N7. |
| `logs_comparison/*/experiment_log.json` (40 dirs) | **Confirmed fabricated.** Every file carries the literal `"_config_hash": "mockhash"` (a real run's logger computes a genuine sha256-based hash — see `utils/logger.py::log_config`) and constant per-round accuracy across all 100 rounds, which is not possible for a real stochastic FL simulation. | `grep "_config_hash": "mockhash"`, audit §2/§13. |
| `tables/table1.tex.DO_NOT_CITE` | **Flagged; quarantined during the final closure pass (2026-09-04).** A ready-to-`\input` LaTeX table sitting at `results/tables/table1.tex`, carrying the same 24 cells as `full_comparison_results.json` and as the paper's Table V. It is derived from a flagged artifact, is in a form that invites being pasted straight into a manuscript, and occupied a generated-output path. Renamed with `.DO_NOT_CITE` so it cannot be `\input` by its old path. | Cell-by-cell match against the quarantined `full_comparison_results.json`. |
| `tables/full_comparison_results.json` | **Flagged, not confirmed fabricated.** Not a byte-match to the mock generator (has plausible per-seed spread), but its 5-seed means agree with the paper's Table V to all four decimal places across all 24 (strategy × metric) cells simultaneously — a level of agreement inconsistent with genuine independent stochastic measurement. Quarantined out of caution per the "do not let a mock/stale artifact sit where a measured one should be" requirement; not deleted, since the audit could not directly prove fabrication (no generating script found for this specific file). | Audit §4 N6 ("cannot verify as genuine measured output"). |

## What replaces this

- `results/tables/proposition1_real.json` (regenerated at the top-level `results/tables/` path, **not** here) now holds the actual output of a real, deterministic execution of `theory/proposition1_verification.py`, with real provenance (script path, git commit, timestamp, parameters) instead of `"mockhash"`.
- All other tables/logs above require a genuine execution of the corresponding runner (`run_full_comparison.py`, `run_ablation.py`, `run_ratio_sweep.py`) to be regenerated. These runners are real, working infrastructure — the fabrication was in the *output files*, not in the code that (when actually run) produces them. Per this remediation's scope restriction, the full-scale campaigns were **not** rerun; only small smoke tests were executed to confirm the pipelines work. See the remediation report for what was smoke-tested and what remains pending a full run.

## Commit history note

These files were originally introduced in commit `ce03b31` ("Deterministic
partitioning, mock results, docs"). They were never subsequently overwritten
by genuine executions for four of the five files `mock_phase7_results.py`
touches, and sat at the paths a real result would occupy — meaning any
downstream table/figure generation reading from `results/tables/` or
`results/logs/comparison/` prior to this remediation was consuming fabricated
data without any marker distinguishing it from real output.
