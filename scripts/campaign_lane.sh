#!/usr/bin/env bash
# scripts/campaign_lane.sh - execute one ordered lane of the campaign.
#
# Usage:  bash scripts/campaign_lane.sh <lane-name> <phase> [<phase> ...]
#
# Each <phase> is a key in the case block below and maps to exactly the
# documented command for that paper artifact. A lane runs its phases in order,
# appending to results/_campaign_logs/<lane>.log and recording one line per
# phase in results/_campaign_logs/<lane>.status (phase, start, end, exit code).
#
# The lane is resumable: TVFLIDS_RESUME=1 makes every (strategy, attack, seed)
# cell whose own complete experiment_log.json is already on disk return that
# run's stored summary instead of recomputing it. Re-running a lane after a
# crash therefore continues where it stopped rather than starting over, and
# never overwrites a genuine completed cell.
#
# Nothing in this file computes, adjusts or synthesises a metric. It is an
# ordering, logging and restart wrapper around the repository's own runners.

set -u

REPO="/mnt/c/Users/Ali Akarma/Documents/GitHub/TV-FLIDS"
cd "$REPO"

PY="$HOME/miniconda3/envs/tvflids_cpu/bin/python"

export PYTHONUNBUFFERED=1
# WSL2's CUDA driver shim aborts (double free in cuInit) in any process whose
# CUDA devices are masked off, which is what Ray does to every worker actor.
# The campaign runs on the CPU build of the same torch release, so no CUDA is
# initialised anywhere; the flag is kept as a belt-and-braces guard.
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
# 6 concurrent Ray client actors per lane (12 logical CPUs / 2 lanes / 1 CPU
# per actor). Benchmarked as the fastest setting on this machine.
export TVFLIDS_SIM_CLIENT_CPUS="${TVFLIDS_SIM_CLIENT_CPUS:-2}"
export TVFLIDS_SIM_CLIENT_GPUS=0.0
# TensorBoard event files are a convenience view, not the scientific record;
# their per-round flush to the 9p-mounted repo dominated the round time.
export TVFLIDS_TENSORBOARD=0
export TVFLIDS_RESUME=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

LANE="$1"; shift
LOG="results/_campaign_logs/${LANE}.log"
STATUS="results/_campaign_logs/${LANE}.status"
mkdir -p results/_campaign_logs

SEEDS="42 123 456 789 1337"
STRATS8="fedavg krum trimmed_mean fltrust foolsgold flame rfa tvflids"

note () { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

run_phase () {
    local phase="$1"; shift
    local start end rc
    start=$(date -u +%FT%TZ)
    note "=== PHASE $phase START ==="
    "$@" >> "$LOG" 2>&1
    rc=$?
    end=$(date -u +%FT%TZ)
    note "=== PHASE $phase END rc=$rc ==="
    printf '%s\t%s\t%s\t%s\t%s\n' "$LANE" "$phase" "$start" "$end" "$rc" >> "$STATUS"
    return $rc
}

phase_cmd () {
    case "$1" in
      A_main_comparison)
        run_phase "$1" $PY experiments/run_full_comparison.py \
            --strategies $STRATS8 --attack label_flip_30 \
            --seeds $SEEDS --rounds 100 ;;
      B_leakage_free)
        run_phase "$1" $PY experiments/run_full_comparison.py \
            --strategies $STRATS8 --attack label_flip_30 \
            --seeds $SEEDS --rounds 100 --protocol leakage_free ;;
      C_ablation)
        run_phase "$1" $PY experiments/run_ablation.py \
            --attack label_flip_30 --rounds 100 --seeds $SEEDS ;;
      D_noniid_sweep)
        run_phase "$1" $PY experiments/run_noniid_sweep.py \
            --alphas 0.1 0.5 1.0 --seeds $SEEDS --rounds 100 ;;
      E_multi_attack)
        run_phase "$1" $PY experiments/run_multi_attack_matrix.py \
            --seeds $SEEDS --rounds 100 ;;
      F_adaptive_attacks)
        # Table XI: ACK1 over {FedAvg, FLTrust, TV-FLIDS}, ACK2 over the same
        # three (the manuscript's table lists FLTrust/TV-FLIDS for ACK2; FedAvg
        # is included so both adaptive attacks have the undefended reference).
        run_phase "$1" $PY experiments/run_multi_attack_matrix.py \
            --strategies fedavg fltrust tvflids \
            --attacks ack1_evasion_30 ack2_coalition_30 \
            --seeds $SEEDS --rounds 100 \
            --output results/tables ;;
      G_extended_significance)
        run_phase "$1" $PY experiments/run_extended_significance.py \
            --strategies tvflids fltrust --rounds 100 ;;
      H_hp_sweep_baseline)
        run_phase "$1" $PY experiments/run_hyperparameter_sweep.py \
            --target baseline --seeds $SEEDS --rounds 100 ;;
      I_hp_sweep_tvflids)
        run_phase "$1" $PY experiments/run_hyperparameter_sweep.py \
            --target tvflids --seeds $SEEDS --rounds 100 ;;
      J_bucketing_deepsight)
        run_phase "$1" $PY experiments/run_full_comparison.py \
            --strategies bucketing deepsight --attack label_flip_30 \
            --seeds $SEEDS --rounds 100 ;;
      R_ratio_sweep)
        run_phase "$1" $PY experiments/run_ratio_sweep.py \
            --methods fedavg krum fltrust tvflids \
            --ratios 0.0 0.1 0.2 0.3 0.4 0.5 0.6 \
            --seeds $SEEDS --rounds 100 ;;
      *)
        note "UNKNOWN PHASE: $1"; return 2 ;;
    esac
}

note "lane=$LANE phases=$* python=$PY cpus=$TVFLIDS_SIM_CLIENT_CPUS"
for p in "$@"; do
    phase_cmd "$p" || note "phase $p returned non-zero; continuing to next phase"
done
note "lane=$LANE COMPLETE"
