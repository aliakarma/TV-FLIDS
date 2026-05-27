.PHONY: install data smoke test full-comparison ablation figures reproduce clean help

# ── Environment ──────────────────────────────────────────────────────────────
install:
	conda env create -f environment.yml
	@echo "Activate with: conda activate tvflids"

install-pip:
	pip install -r requirements.txt

# ── Data ─────────────────────────────────────────────────────────────────────
data:
	bash scripts/download_nslkdd.sh

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
	    --rounds 50 \
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
	@echo "  make ablation         Table 2 (ablation A1-A5)"
	@echo "  make figures          Figure 3 (robustness curve)"
	@echo "  make reproduce        Full paper reproduction"
	@echo "  make clean            Remove all generated outputs"
