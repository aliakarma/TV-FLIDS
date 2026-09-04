# TV-FLIDS — Final Pre-Campaign Closure Pass

**Date:** 2026-09-04
**Scope:** source inspection, mathematical derivation, targeted/unit/regression tests,
compilation, PDF text inspection, configuration checks.
**Explicitly out of scope and not performed:** any full-scale experimental run
(100 rounds, multi-seed, attack matrices, ablations, hyperparameter sweeps,
CIC-IoT-2023, GPU campaigns, Table V–XII reproduction).

---

## 1. Final Closure Summary

Ten defects carried forward from the methodology gate were resolved, plus four
found during this pass. Nothing scientific was generated; no value was
fabricated; no test was weakened.

The pass corrected two mathematical statements (the validation forward-pass
count and Proposition 1's missing hypothesis), removed a genuine symbol
collision from the manuscript, repaired an invisible LaTeX corruption and built
tooling that detects its whole class, closed a configuration reproducibility
trap, made ablation A1's semantics explicit, gave every data-driven figure a
reproducible generation path, and rewrote the Abstract, Conclusion, results
sections and every affected caption so that no unsupported value reads as a
measurement.

Four defects were found during this pass, not inherited:

1. The malformed cross-reference was not a typo. The backslash of `\ref` had
   been replaced by a **raw carriage-return byte**. LaTeX treats a bare CR as
   whitespace, so it compiled with no error and no undefined-reference warning,
   and typeset as `Section efsec:adaptive` — with the braces eaten as grouping,
   so grepping the PDF text for `ef{` also missed it.
2. `theory/proposition1_verification.py` raised a numpy `ValueError` on exactly
   the `ℬ_𝒜 = ∅` case the proposition documents as well defined.
3. `figure4_ablation_bars` and `figure5_adaptive_weights` had no call site
   anywhere, while `scripts/check_results.py` listed their outputs as required —
   so `make check-results` could never pass even after a genuine campaign.
4. `evaluation/visualization.py::_save` printed a U+2192 arrow, which raises
   `UnicodeEncodeError` on a cp1252 Windows console *after* writing the file,
   aborting the remaining figures in the same call.

---

## 2. Defect → Resolution Matrix

| # | Defect | Root cause | Resolution | Files | Test / verification | Status |
|---|---|---|---|---|---|---|
| 1 | §XII-A/§XII-C gave the validation forward-pass count as \|𝒜\|+1 and called it an upper bound | Check 1 evaluates a client *before* acceptance is known, so rejected clients consume a pass too; the count runs over the cohort 𝒫, not 𝒜 | Exact count restated as \|𝒫\|+1 (new Eq. 21); \|𝒜\|+1 identified as a **lower** bound; naive count corrected to \|𝒫\|+\|𝒜\|+1; gate complexity restated as O(\|𝒫\|·P·B_val); §XII-B changed to "per participating client" | `Paper/TV-FLIDS.tex` §XII-A/B/C | `tests/test_forward_pass_count.py` — 10 tests instrumenting the real `aggregate_fit`; asserts \|𝒫\|+1 with 0 and with 3 of 6 rejected | **Fixed** |
| 2 | ℋ, ℬ, f, N_H used for both population and accepted-set quantities (N_H = 14 vs. accepted honest count, with only 10 clients sampled) | §V rescoped to the accepted set without renaming | §III pinned as population-level with an explicit scope paragraph; §V uses ℋ_𝒜, ℬ_𝒜, f_𝒜, N_{H,𝒜}, τ̄_{H,𝒜}, τ^max_{ℬ,𝒜} via new macros; every occurrence in §I, §V, §X, §XIII, Fig. 3 caption traced and reclassified; Lemma 1 explicitly marked population-level | `Paper/TV-FLIDS.tex` preamble, §III-C, §V-A, Prop. 1, proof, Cor. 1, Remarks, §X, §XIII, Fig. 3 caption | PDF text inspection; `scripts/check_manuscript.py` | **Fixed** |
| 3 | Proposition 1 had no hypothesis excluding ℋ_𝒜 = ∅, which the accepted-set formulation made reachable | Rescoping introduced a case where **w\*** and τ̄ are undefined and the denominator is 0 | Added hypothesis N_{H,𝒜} ≥ 1 (Eq. 17) inside the proposition; added a remark proving it is both necessary and sufficient (the old τ̄ > 0 assumption follows from it via the floor clip); proof now shows W_H ≥ N_{H,𝒜}τ_min > 0 and handles ℬ_𝒜 = ∅ | `Paper/TV-FLIDS.tex` §V; `theory/proposition1_verification.py` | `tests/test_proposition1_domain.py` — 12 tests: valid case, empty-honest rejected as out-of-domain, empty-Byzantine equality convention | **Fixed** |
| 3b | Verification script **crashed** on the documented ℬ_𝒜 = ∅ case | `np.vstack` on `flatten_params([])`, shape `(0,)` not `(0,D)` | Guarded both the aggregate and the max-deviation computation; empty-Byzantine now returns 0 = 0 as documented | `theory/proposition1_verification.py` | same file; `make theory` still reproduces 0.588 ± 0.198 exactly | **Fixed** |
| 4 | `Section~\ref{sec:adaptive}` malformed; compiled clean | Backslash replaced by a **raw CR byte**; invisible to LaTeX and to PDF-text greps | Repaired byte-level; the forward-pointer's unsupported claim ("where the two dispersions are compared directly" — §XI does no such comparison) also corrected | `Paper/TV-FLIDS.tex` §VII-A | `scripts/check_manuscript.py` (3-level detector) + `tests/test_manuscript_integrity.py` — 18 tests incl. no-false-positive cases | **Fixed** |
| 5 | Abstract presented pending/withdrawn results as established | Not updated after the audit | Rewritten: keeps the verified bound + its numerical check (0.588 ± 0.198) and the two derived cost properties; states the evaluation protocol as released machinery; states plainly that the reported values are withdrawn or pending | `Paper/TV-FLIDS.tex` abstract | PDF text inspection | **Fixed** |
| 5b | Conclusion likewise | same | Restructured into *established analytically* / *established numerically* / *implemented but not yet established*; explicitly withdraws the ablation deltas, the p ≈ 0.014 claim, and the 15.1% overhead figure | `Paper/TV-FLIDS.tex` §XVI | PDF text inspection | **Fixed** |
| 6 | A1–A4 mock-derived values displayed as measurements; provenance hidden in LaTeX comments | Disclosure was in `%%` comments only | Table VII retitled **"Historical, non-citable"**, given a **Provenance column** naming each row's traced origin, bold emphasis removed, footnote added; three visible paragraphs above it state the provenance and that no cell may be cited; §IX prose rewritten to describe mechanisms, with the 6.01/4.01/3.01/2.01 deltas withdrawn | `Paper/TV-FLIDS.tex` §IX | PDF text inspection (renders unmistakably) | **Fixed** |
| 7 | A1 sets τ_z = +∞, which also disables the anomaly **signal** O_i, not just Check 3 | τ_z is shared between Check 3 and Eq. (9) | Determined **Option B** — implementation is faithful to the stated definition. Manuscript now states the consequence explicitly, treats the A1 gap as an *upper bound* on the gate's own contribution, notes τ_L/τ_C are gate-local, and names the seventh arm that would disentangle them | `Paper/TV-FLIDS.tex` §IX | `tests/test_a1_ablation_semantics.py` — 7 tests; O_i < 1e-6 under A1's τ_z, and A1 trust ≡ trust with O_i ≡ 0 | **Fixed (implementation unchanged)** |
| 8 | Figures hand-authored; regenerating results did not update the manuscript | No pipeline from artifacts to TikZ | **Option B** — generated TikZ. `scripts/generate_manuscript_figures.py` emits `Paper/figures/fig_*.tex`; each axis wraps its bodies in `\IfFileExists`, so a generated file supersedes the provisional coordinates and the document compiles either way. Generator **refuses** to emit for a missing artifact. `utils/logger.py` gained `log_extra`, and `run_experiment.py` now persists trust history + strategy round logs, without which Figures 3–4 had no disk source | `scripts/generate_manuscript_figures.py`, `Paper/figures/README.md`, `utils/logger.py`, `experiments/run_experiment.py`, `Makefile`, both `.tex` | `tests/test_manuscript_figure_generation.py` — 26 tests: refusal, partial artifacts, fidelity, idempotence | **Fixed** |
| 8b | `figure4_ablation_bars` / `figure5_adaptive_weights` had no call site, yet `check_results.py` required their output | Orphaned generators | Wired into `run_ablation.py` and `run_experiment.py` | `experiments/run_ablation.py`, `experiments/run_experiment.py` | Verified both produce valid PDFs from the exact in-repo data shapes | **Fixed** |
| 8c | `_save` crashed on cp1252 consoles after writing the file | U+2192 in a `print` | ASCII arrow | `evaluation/visualization.py` | Reproduced, then verified fixed | **Fixed** |
| 9 | `--strategy tvflids_fixed` produced α=0.4, β=0.4, γ=0.2 from the shipped config; only `run_ablation.py`'s override made A6 correct | Base config disagreed with paper Table IV | Base config set to 1/3 each, matching Table IV; in-code fallbacks aligned. Adaptive strategy provably unaffected (it never reads these keys — softmax(0) = 1/3) | `config/fl_config.yaml`, `fl/strategy.py`, `trust/trust_scorer.py` | `tests/test_config_defaults.py` — 17 tests: CLI == A6, runner agrees, other arms untouched, adaptive unchanged | **Fixed** |
| 10 | `utils/ste.py` docstring quoted the superseded "at boundary points" wording | Not updated with the paper | Quote replaced with the current text; added a note that the estimator covers the whole saturated region, and that "boundary points" is measure-zero and effectively vacuous in floating point. Functionality unchanged | `utils/ste.py` | `tests/test_ste_gradient.py` (existing, unmodified) still passes | **Fixed** |
| 11 | Prose could be read as τ_z = 3.0 / τ_L = −0.1 being used in a live round | t = 0 is the schedule endpoint; Flower's first round is t = 1 | Added a "Round indexing" paragraph giving the first applied values, τ_z(1) = 2.975 and τ_L(1) = −0.095; Table IV footnote updated; "initialized at 3.0 during warmup" reworded. Schedule unchanged | `Paper/TV-FLIDS.tex` §IV-A, Table IV | `tests/test_warmup_schedule.py` (existing) already pins (1, 2.975) | **Fixed** |
| 12 | Assertive language unsupported by current evidence | Not updated after the audit | Visible status banner at the head of §VII covering Tables V–XII and Figures 2–5; 11 table/figure captions marked; 20 individual claims rewritten across §I, §VI-E, §VII, §VIII, §X, §XI, §XV and the supplementary; the ten-seed narrative rewritten with its outcome explicitly withdrawn | both `.tex` | PDF text inspection | **Fixed** |
| 13 | `results/tables/table1.tex` — a ready-to-`\input` LaTeX table of the unverifiable Table V values, sitting at a generated-output path | Left behind by an earlier pass | Moved to `results/_QUARANTINED_MOCK/tables/table1.tex.DO_NOT_CITE` with an entry in the quarantine README | `results/`, quarantine README | `make check-results` | **Fixed** |
| 13b | Six 0-byte placeholder PDFs at the paper's figure paths | Left behind | Deleted; `check_results` now reports MISSING (accurate) rather than EMPTY | `results/figures/` | `make check-results` | **Fixed** |
| 14 | Equation numbers cited in code drifted (eq:prop1 moved 17 → 18, eq:agg 14 → 15, eq:signal_range quoted as 12) | Two equations inserted this pass | All citations resynced against the compiled `.aux` | `utils/ste.py`, `theory/proposition1_verification.py`, tests | `tests/test_equation_crossrefs.py` — 17 tests resolving each label against `Paper/TV-FLIDS.aux` | **Fixed** |

---

## 3. Mathematical Corrections

### 3.1 Validation forward-pass count

`trust/verification.py::verify_all` computes `loss_after` for a client and only
*then* `continue`s on rejection. Acceptance is therefore not knowable until the
pass has been spent, so **every** submitted client consumes one. Stage 2 adds
none: `A_i` (Eq. 8) is defined on the two quantities Check 1 already cached, and
`_eval_model`'s parameter-hash cache returns them for bit-identical vectors.

    forward_passes = |𝒫| + 1        (one baseline + one Check-1 pass per submitted client)

Since 𝒜 ⊆ 𝒫, `|𝒜| + 1 ≤ |𝒫| + 1`: the old expression is a **lower** bound, short
by exactly the number of Stage-1 rejections — it under-counts precisely when the
gate is working hardest. A cache collision can only *reduce* the count, so
`|𝒫| + 1` is exact in the collision-free case and an upper bound in general.
Both directions are asserted in `tests/test_forward_pass_count.py`, which counts
real sweeps of `D_val` through the released `aggregate_fit`.

### 3.2 Notation

| Symbol | Level | Where |
|---|---|---|
| ℋ, ℬ, f, N_H | population (N = 20, f = 6, N_H = 14) | §III-C, §II (Krum's f), §X (f/N), Fig. 3 legend, §XV |
| 𝒫 | round cohort, \|𝒫\| = ρ_fit·N = 10 | §III-A, Alg. 1, §XII |
| 𝒜 ⊆ 𝒫 | Stage-1 accepted set | §IV, §V |
| ℋ_𝒜, ℬ_𝒜, f_𝒜, N_{H,𝒜}, τ̄_{H,𝒜}, τ^max_{ℬ,𝒜} | accepted-set | §V, §I contribution bullet, §X, §XIII |

Defined as ℋ_𝒜 := 𝒜 ∩ ℋ and ℬ_𝒜 := 𝒜 ∩ ℬ, so the relation to the population sets
is explicit rather than asserted, giving f_𝒜 ≤ f and N_{H,𝒜} ≤ N_H directly.
Lemma 1 is a per-client statement across rounds and is therefore stated on the
population sets, now said so in the text. §XIII notes that its harness admits all
20 clients, so 𝒜 is the full set there and f_𝒜 = f.

### 3.3 Proposition 1 hypothesis

`N_{H,𝒜} ≥ 1` is added as Eq. (17), inside the proposition's assumptions. It is
**necessary**: with ℋ_𝒜 = ∅, **w\*** (Eq. 16) and τ̄_{H,𝒜} are undefined and the
denominator is 0. It is **sufficient**: the floor clip gives T_i ≥ τ_min > 0 for
every accepted client, so N_{H,𝒜} ≥ 1 already forces τ̄_{H,𝒜} ≥ τ_min > 0. The
earlier `τ̄ > 0` assumption is implied by it and is now stated as a consequence.
The theorem is neither weakened nor strengthened — only made well posed.

### 3.4 A1 semantics

τ_z appears in Check 3 *and* in O_i = 1 − exp(−z_i/τ_z) (Eq. 9). A1's τ_z = +∞
therefore sends O_i → 0 and removes −γO_i from Eq. (6). Measured: max O_i <
1e-6 under A1's τ_z = 1e9, and A1 trust scores equal those computed with O_i ≡ 0
to 1e-9. This is faithful to A1 as specified, so the implementation is unchanged
and the manuscript now states the consequence and its interpretive cost: A1
bounds the gate's contribution from above rather than isolating it.

---

## 4. Methodology Confirmation

- **Adaptive threshold annealing — ENABLED by default.** `verification.adaptive_thresholds: true`; `use_adaptive_thresholds=None` at the production call site resolves to the config. Pinned by `tests/test_config_defaults.py`. Round indexing clarified (§XII of this report, item 11) without touching the schedule.
- **Straight-through estimator — part of the proposed method.** `utils/ste.py::clip_ste` is on the live meta-gradient path in `fl/strategy.py`; forward bit-identical to `torch.clamp`, backward identity. Docstring now matches the corrected manuscript wording.
- **Proposition 1 accepted-set formulation — RETAINED, and now complete.** It is the object Eq. (15) aggregates, the proof sums over, and the implementation records. Its two boundary cases are handled as specified, and it carries the N_{H,𝒜} ≥ 1 hypothesis the rescoping required.

---

## 5. Manuscript Corrections

**Main paper** — preamble (notation macros); Abstract; §I (contribution bullets, "our approach"); §III-C (population-scope paragraph); §IV-A (round indexing, τ_z wording); §IV-C (unchanged, referenced); §V-A (Notation), Proposition 1 + new Remark, proof, Corollary 1, Remarks 1–4, Lemma 1; §VI-C (Table IV footnote); §VI-D (baseline defaults); §VI-E (both sweeps); §VII (status banner, Table V caption, §VII-A ASR paragraph incl. the repaired reference, FedAvg paragraph, figure-provenance note, Figures 2–4 captions and `\IfFileExists` wiring); §VIII-A, §VIII-B (ten-seed narrative), §VIII-C; §IX (A1 definition, three provenance paragraphs, Table VII rebuilt with a Provenance column, all six arm discussions); §X (Figure 5 caption + wiring, f/N paragraph); §XI (ACK1 paragraph); §XII-A/B/C (rewritten, new Eq. 21); §XIII (accepted-set scope, new boundary-case remark); §XV (six items + new "empirical campaign" item listing the regeneration command for every table and figure); §XVI (Conclusion restructured).

**Supplementary** — new standalone status section; Table S1 and S2 prose; Bucketing/DeepSight discussion; CIC-IoT-2023 discussion; all four table captions and the Figure S1 caption; Figure S1 `\IfFileExists` wiring.

**Repository documentation** — `README.md` (third-pass section, manuscript-figure pipeline, corrected figure status row); `Paper/figures/README.md` (new); `results/_QUARANTINED_MOCK/README.md` (table1.tex entry); `Makefile` (four new targets + help).

---

## 6. Figure Reproducibility Architecture

Figure 1 is an analytical schematic with no experimental content and stays
hand-authored. Figures 2–5 and S1 each contain, inside their own `axis`:

```latex
\IfFileExists{figures/fig_<name>.tex}{%
    \input{figures/fig_<name>.tex}%
}{%
    <provisional coordinates, marked pending in the caption>
}
```

`scripts/generate_manuscript_figures.py` reads campaign artifacts and writes
exactly those files. **It never fabricates**: a figure whose artifact is missing
is skipped with a message and no file is written, so the document falls back to
coordinates its caption already marks as provisional.

| Generated file | Figure | Input artifact | Command |
|---|---|---|---|
| `fig_convergence.tex` | 2 | `results/logs/comparison/<s>_label_flip_30_seed<S>/experiment_log.json` → `rounds[].accuracy` | `make full-comparison` |
| `fig_trust.tex` | 3 | same → `extra.trust_history`, `extra.malicious_ids` | `make full-comparison` |
| `fig_weights.tex` | 4 | same → `extra.strategy_round_logs[].adaptive_{alpha,beta,gamma}` | `make full-comparison` |
| `fig_robustness.tex` | 5 | `results/tables/ratio_sweep_results.json` | `make figures` |
| `fig_ciciot.tex` | S1 | `results/tables/dataset_comparison_results.json` → `ciciot2023` | `make ciciot2023` |

`extra.trust_history` and `extra.strategy_round_logs` are new: they existed only
in memory on the live strategy object, so Figures 3 and 4 had no disk source at
all. `utils/logger.py::log_extra` persists them.

**Future command sequence:**

```bash
make full-comparison && make figures && make ciciot2023
make manuscript-figures          # results/ -> Paper/figures/fig_*.tex
make check-manuscript-figures    # exits non-zero if any figure is still unbacked
make paper                       # regenerates figure data, then builds both PDFs
make check-manuscript            # source + PDF + log integrity
```

---

## 7. Tests Executed

New (75 tests, all passing):

| File | Tests | Covers |
|---|---|---|
| `tests/test_forward_pass_count.py` | 10 | \|𝒫\|+1 against the real `aggregate_fit`; rejected clients still cost a pass; \|𝒜\|+1 is a lower bound; cache only reduces; derivation |
| `tests/test_proposition1_domain.py` | 12 | valid N_{H,𝒜} ≥ 1; empty-honest out-of-domain; empty-Byzantine equality convention |
| `tests/test_config_defaults.py` | 17 | `tvflids_fixed` == A6 from the CLI; runner agrees; other arms untouched; adaptive unaffected; annealing/warmup/val-size defaults |
| `tests/test_a1_ablation_semantics.py` | 7 | A1's τ_z collapses O_i; γO_i drops out; τ_L/τ_C are gate-local |
| `tests/test_manuscript_figure_generation.py` | 26 | refusal without artifacts; partial artifacts; fidelity; idempotence; downsampling |
| `tests/test_manuscript_integrity.py` | 18 | stray CR, `ef{` residue, missing backslash, TAB-mangled macros; no false positives |
| `tests/test_equation_crossrefs.py` | 17 | every equation number cited in code resolves against the compiled `.aux` |

Existing suite: `pytest tests/ --ignore=tests/test_integration.py` →
**319 passed, 1 skipped, 7 failed.** The 7 failures are the pre-existing
Ray-simulation tests (`TestDeterminism`, `TestBaselineConvergence`). They are
environmental, not a regression: a bare `import ray; import torch` in this
environment fails with `OSError: [WinError 1114] ... c10.dll`, with no project
code involved. No test was modified to hide a failure.

Also run: `make theory` (0.588 ± 0.198, min 0.402, max 0.999 — unchanged after
the empty-Byzantine fix); `scripts/check_results.py` (correctly exits non-zero:
tables missing, authenticity clean); `scripts/check_manuscript.py` (source, PDF
and log — all pass); clean `latexmk` rebuild of both documents (**zero LaTeX
warnings, zero BibTeX warnings, zero undefined references**).

---

## 8. Remaining Full-Campaign Requirements

Every item below needs a real run and **cannot** be closed by inspection.

| Artifact | Command |
|---|---|
| Table V, Figures 2–4 | `make full-comparison` |
| Table VI (leakage-free) | `make leakage-free` |
| Table VII (ablation A1–A6) | `make ablation` |
| Table VIII (non-IID sweep) | `make noniid-sweep` |
| Tables IX / X (attack matrix) | `make multi-attack` |
| Table XI (ACK1, ACK2) | `make multi-attack` |
| Table XII (overhead) | any run on the hardware of §VI-F |
| Figure 5 | `make figures` |
| §VIII-B ten-seed significance | `make extended-significance` |
| Supp. Tables S1–S2 | `make hp-sweep-baseline`, `make hp-sweep-tvflids` |
| Supp. Bucketing / DeepSight | `make full-comparison --strategies ... bucketing deepsight` |
| Supp. CIC-IoT-2023 (+ the real dataset) | `make ciciot2023` |

Then `make manuscript-figures`, `make check-manuscript-figures`,
`make check-results`, `make paper`, `make check-manuscript`.

---

## 9. Remaining Human Review

1. **Ray/PyTorch DLL failure** blocks every simulation-based test and therefore
   the campaign itself on this machine. Must be resolved before any run.
   Reproduce with `python -c "import ray, torch"`.
2. **`[HUMAN REVIEW REQUIRED]` markers already in the sources** (unchanged this
   pass): the η_meta = 0.1 accuracy in Supp. Table S2 versus the prose, and the
   removed "higher per-round variance" clause in §VII-B. Both become moot once
   the tables are regenerated.
3. **Two stale figure PDFs** — `results/figures/fig1_tvflids_label_flip_30_seed42.pdf`
   and `fig2_trust_tvflids_seed42.pdf` (2026-05-06). Genuine output of a
   pre-correction run, referenced nowhere, gitignored. Deleting real output is
   the authors' call, so they were left in place.
4. **Whether to retain Table VII at all.** It is now unmistakably marked as
   historical and non-citable, which preserves the record of what was withdrawn.
   Removing it entirely is defensible; that is an editorial judgment.
5. **Venue disclosure.** Whether the Section I provenance note and the withdrawal
   language suffice for TDSC, or whether an erratum / withdrawal-and-resubmission
   is warranted, is an editorial and ethical decision for the authors.

---

## 10. Final Go/No-Go

**GO — READY FOR FULL EXPERIMENTAL CAMPAIGN**

Qualified by item 1 of Section 9: the campaign cannot physically run until the
Ray/PyTorch DLL problem is resolved. That is an environment defect, not a
repository defect, and no amount of source work closes it.

Subject to that, the mathematics, implementation, configuration, manuscript,
provenance and reproducibility path now describe the same method. Every claim in
the manuscript is either supported by current evidence, derived from the
algorithm, or visibly marked as withdrawn or pending; every pending item has a
documented command; and `check-results`, `check-manuscript` and
`check-manuscript-figures` will each fail loudly if a fabricated, malformed or
unbacked artifact re-enters.
