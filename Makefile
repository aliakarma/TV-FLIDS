.PHONY: install install-pip data verify-env check-results smoke test full-comparison ablation figures dataset-comparison multi-attack noniid-sweep extended-significance hp-sweep-baseline hp-sweep-tvflids leakage-free ciciot2023 theory manuscript-figures check-manuscript-figures check-manuscript paper reproduce clean help

# ── Environment ──────────────────────────────────────────────────────────────
install:
	conda env create -f environment.yml
	@echo "Activate with: conda activate tvflids"

install-pip:
	pip install -r requirements.txt

# ── Data ─────────────────────────────────────────────────────────────────────
data:
	bash scripts/download_nslkdd.sh

# ── Environment / integrity checks ──────────────────────────
# Reports the ACTUAL stack against the paper's stated one (Section VI-D).
verify-env:
	python scripts/verify_environment.py

# Completeness + authenticity of generated results (rejects mock/0-byte artifacts).
check-results:
	python scripts/check_results.py

# Deterministic Proposition 1 verification (no full run needed).
theory:
	python theory/proposition1_verification.py

# ── Verification ─────────────────────────────────────────────────────────────
smoke:
	python experiments/run_experiment.py \
	    --strategy tvflids --attack label_flip_30 \
	    --rounds 3 --seed 42
	python tests/test_integration.py

test:
	python tests/test_all.py
	python tests/test_integration.py

# ── Core Experiments ─────────────────────────────────────────────────────────
full-comparison:
	python experiments/run_full_comparison.py \
	    --strategies fedavg krum trimmed_mean fltrust foolsgold flame rfa tvflids \
	    --attack label_flip_30 \
	    --seeds 42 123 456 789 1337 \
	    --rounds 100

ablation:
	python experiments/run_ablation.py \
	    --attack label_flip_30 \
	    --rounds 100 \
	    --seeds 42 123 456 789 1337

figures:
	python experiments/run_ratio_sweep.py \
	    --methods fedavg krum fltrust tvflids \
	    --ratios 0.0 0.1 0.2 0.3 0.4 0.5 0.6 \
	    --seeds 42 123 456 789 1337 \
	    --rounds 100

dataset-comparison:
	python experiments/run_dataset_comparison.py \
	    --datasets nslkdd unswnb15 \
	    --strategies fedavg fltrust tvflids \
	    --seeds 42 123 456 789 1337 \
	    --rounds 100

# ── Experiments added during forensic-audit remediation ─────────────
# NONE of the targets below has been executed at full scale. Each is the
# documented full-run command for a paper table whose numbers are still
# pending a genuine execution (see README "Results pending regeneration").

# E13 - Table IX / X: every strategy x attack cell.
multi-attack:
	python experiments/run_multi_attack_matrix.py \
	    --seeds 42 123 456 789 1337 --rounds 100

# E12 - Table VIII: non-IID concentration sweep.
noniid-sweep:
	python experiments/run_noniid_sweep.py \
	    --alphas 0.1 0.5 1.0 --seeds 42 123 456 789 1337 --rounds 100

# E7 - Section VIII-B: extended ten-seed paired significance test.
extended-significance:
	python experiments/run_extended_significance.py \
	    --strategies tvflids fltrust --rounds 100

# E10 - Supp. Table S1: baseline hyperparameter sensitivity.
hp-sweep-baseline:
	python experiments/run_hyperparameter_sweep.py \
	    --target baseline --seeds 42 123 456 789 1337 --rounds 100

# E11 - Supp. Table S2: TV-FLIDS's own hyperparameter sensitivity.
hp-sweep-tvflids:
	python experiments/run_hyperparameter_sweep.py \
	    --target tvflids --seeds 42 123 456 789 1337 --rounds 100

# Table VI - leakage-free protocol (D_val drawn pre-SMOTE, SMOTE per client).
leakage-free:
	python experiments/run_full_comparison.py \
	    --strategies fedavg krum trimmed_mean fltrust foolsgold flame rfa tvflids \
	    --attack label_flip_30 --seeds 42 123 456 789 1337 --rounds 100 \
	    --protocol leakage_free

# Supp. - CIC-IoT-2023 cross-dataset evaluation (needs the real dataset).
ciciot2023:
	python experiments/run_dataset_comparison.py \
	    --datasets nslkdd ciciot2023 --strategies fedavg fltrust tvflids \
	    --seeds 42 123 456 789 1337 --rounds 100

# ── Manuscript figures ───────────────────────────────────────────────
# Converts result artifacts into the pgfplots bodies the manuscript inputs
# (Paper/figures/fig_*.tex). Figures whose backing artifact does not exist are
# SKIPPED, never invented: the manuscript then keeps its provisional,
# explicitly-labelled coordinates. See Paper/figures/README.md.
manuscript-figures:
	python scripts/generate_manuscript_figures.py

# Report-only; exits non-zero if any data-driven figure is still unbacked.
check-manuscript-figures:
	python scripts/generate_manuscript_figures.py --check

# Source- and PDF-level manuscript integrity checks: stray control bytes,
# reference commands missing their backslash, undefined references, and
# forbidden text in the compiled PDFs. Catches the class of defect that
# compiles cleanly and is therefore invisible to latexmk's exit status.
check-manuscript:
	python scripts/check_manuscript.py

# Regenerate figure data, then rebuild both PDFs.
paper: manuscript-figures
	latexmk -pdf -cd Paper/TV-FLIDS.tex
	latexmk -pdf -cd Paper/TV-FLIDS_supplementary.tex

# ── Full Reproduction ─────────────────────────────────────────────────────────
reproduce:
	bash scripts/run_all_experiments.sh

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	rm -rf results/logs/* results/figures/* results/tables/*
	find /tmp -name "ablation_config_*.yaml" -delete 2>/dev/null; true
	find /tmp -name "ratio_sweep_*.yaml" -delete 2>/dev/null; true

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo "TV-FLIDS Experiment Runner"
	@echo ""
	@echo "  make install          Create conda environment"
	@echo "  make install-pip      Install via pip only"
	@echo "  make data             Download NSL-KDD dataset"
	@echo "  make smoke            2-round sanity check"
	@echo "  make test             Full unit + integration test suite"
	@echo "  make full-comparison  Table 1 (5 seeds, all strategies)"
	@echo "  make ablation         Table 2 (ablation A1-A6)"
	@echo "  make figures          Figure 3 (robustness curve)"
	@echo "  make reproduce        Full paper reproduction"
	@echo "  make verify-env       Report actual vs. paper environment"
	@echo "  make check-results    Completeness + authenticity of results"
	@echo "  make theory           Proposition 1 numerical verification"
	@echo ""
	@echo "  Pending full runs (added during audit remediation):"
	@echo "    make multi-attack           Table IX/X"
	@echo "    make noniid-sweep           Table VIII"
	@echo "    make extended-significance  Section VIII-B (10 seeds)"
	@echo "    make hp-sweep-baseline      Supp. Table S1"
	@echo "    make hp-sweep-tvflids       Supp. Table S2"
	@echo "    make leakage-free           Table VI"
	@echo "    make ciciot2023             Supp. cross-dataset"
	@echo ""
	@echo "  Manuscript figures (results -> Paper/figures/*.tex):"
	@echo "    make manuscript-figures        Regenerate Figures 2-5 + S1 from results"
	@echo "    make check-manuscript-figures  Report which figures are backed by real data"
	@echo "    make paper                     Regenerate figure data, then build both PDFs"
	@echo "    make check-manuscript          Integrity-check the .tex sources and built PDFs"
	@echo ""
	@echo "  make clean            Remove all generated outputs"
