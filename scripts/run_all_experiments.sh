#!/usr/bin/env bash
# scripts/run_all_experiments.sh
# Reproduces all paper results from scratch.
# Runtime estimate: ~2-4 hours on CPU, ~30 min on GPU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
if command -v nproc >/dev/null 2>&1; then
    HOST_CPUS="$(nproc)"
else
    HOST_CPUS="4"
fi
export TVFLIDS_SIM_CLIENT_CPUS="${TVFLIDS_SIM_CLIENT_CPUS:-$HOST_CPUS}"
echo "Simulation client CPU slots: $TVFLIDS_SIM_CLIENT_CPUS"

PYTHON=""
is_usable_python() {
    local cmd="$1"
    "$cmd" -c "import numpy" >/dev/null 2>&1
}

for CANDIDATE in python3 python; do
    if command -v "$CANDIDATE" >/dev/null 2>&1 && is_usable_python "$CANDIDATE"; then
        PYTHON="$CANDIDATE"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    shopt -s nullglob
    PYENV_CANDIDATES=(/mnt/c/Users/*/.pyenv/pyenv-win/versions/*/python.exe)
    shopt -u nullglob
    if [ "${#PYENV_CANDIDATES[@]}" -gt 0 ]; then
        for CANDIDATE in "${PYENV_CANDIDATES[@]}"; do
            if is_usable_python "$CANDIDATE"; then
                PYTHON="$CANDIDATE"
                break
            fi
        done
    fi
fi

if [ -z "$PYTHON" ]; then
    echo "Error: no working python executable found in PATH."
    echo "Tried: python3, python, and pyenv-win version interpreters."
    exit 1
fi

echo "Using Python: $PYTHON"

CONFIG="config/fl_config.yaml"
ROUNDS=100

echo "================================================================"
echo " TV-FLIDS: Full Experiment Reproduction Pipeline"
echo "================================================================"

# Step 1: Download data
echo ""
echo "[Step 1] Downloading NSL-KDD dataset..."
bash "$SCRIPT_DIR/download_nslkdd.sh"

# Step 2: Stage 1 - Baseline comparison (all 6 strategies)
echo ""
echo "[Step 2] Stage 1: Full strategy comparison (30% label flip)..."
for STRATEGY in fedavg krum trimmed_mean fltrust foolsgold tvflids; do
    echo "  Running $STRATEGY..."
    for SEED in 42 123 456; do
        "$PYTHON" experiments/run_experiment.py \
            --strategy "$STRATEGY" \
            --attack label_flip_30 \
            --rounds "$ROUNDS" \
            --seed "$SEED" \
            --config "$CONFIG" \
            --quiet
    done
done

# Step 3: Stage 2 - Attack variety (TV-FLIDS)
echo ""
echo "[Step 3] Stage 2: Attack variety experiments..."
for ATTACK in gradient_scale_30 noise_30 backdoor_20 min_max_30; do
    echo "  TV-FLIDS vs $ATTACK..."
    "$PYTHON" experiments/run_experiment.py \
        --strategy tvflids \
        --attack "$ATTACK" \
        --rounds "$ROUNDS" \
        --seed 42 \
        --config "$CONFIG" \
        --quiet
done

# Proposition 1 verification (requires log_client_params=true)
echo ""
echo "[Proposition 1] Verification on real experimental outputs..."
echo "  Set 'log_client_params: true' in config/fl_config.yaml"
echo "  Output: results/tables/proposition1_real.json"

# Step 4: Stage 3 - Adversarial ratio sweep
echo ""
echo "[Step 4] Stage 3: Adversarial ratio sweep (0-40%)..."
"$PYTHON" experiments/run_ratio_sweep.py \
    --methods fedavg fltrust tvflids \
    --ratios 0.0 0.1 0.2 0.3 0.4 \
    --seeds 42 123 \
    --rounds "$ROUNDS" \
    --output results/tables

# Step 5: Ablation study
echo ""
echo "[Step 5] Ablation studies (A1-A5)..."
"$PYTHON" experiments/run_ablation.py \
    --attack label_flip_30 \
    --rounds 50 \
    --seeds 42 123 \
    --output results/tables

# Step 6: Statistical validation
echo ""
echo "[Step 6] Full multi-seed comparison (5 seeds)..."
"$PYTHON" experiments/run_full_comparison.py \
    --strategies fedavg krum fltrust tvflids \
    --attack label_flip_30 \
    --seeds 42 123 456 789 1337 \
    --rounds "$ROUNDS" \
    --output results/tables

echo ""
echo "================================================================"
echo " All experiments complete."
echo " Results: results/tables/"
echo " Figures: results/figures/"
echo " Logs:    results/logs/"
echo "================================================================"
