#!/usr/bin/env bash
# colab/setup_colab.sh — install the pinned stack on a Colab T4 runtime.
#
#   bash colab/setup_colab.sh                 # install, then report versions
#   bash colab/setup_colab.sh --report-only   # report, install nothing
#
# Principles
#   * Install the versions requirements.txt pins. Nothing is silently upgraded:
#     every pip call carries explicit ==versions, and the final report prints
#     what actually landed so a difference is visible rather than assumed.
#   * torch is installed from the CUDA 12.1 wheel index, because the +cpu build
#     cannot touch the T4 however healthy nvidia-smi looks. 2.1.0+cu121 IS
#     torch 2.1.0 — the local suffix names the build variant, not the release.
#   * This script never claims the environment matches the paper's stated
#     stack. Run scripts/verify_environment.py for that comparison; it reports.
#
# Colab preinstalls a newer numpy/torch than this project pins, and Colab's
# Python is normally 3.11+ while these wheels target 3.10. The notebook checks
# both and says so; where the Python minor differs, the pins cannot all be
# honoured and the campaign must record that difference in its provenance.

set -uo pipefail

REPORT_ONLY=0
[[ "${1:-}" == "--report-only" ]] && REPORT_ONLY=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

say () { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "runtime"
python --version
python -c "import sys; print('executable:', sys.executable)"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version \
               --format=csv,noheader
else
    echo "nvidia-smi: not present (no GPU attached to this runtime)"
fi

PYMM="$(python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
if [[ "$PYMM" != "3.10" ]]; then
    cat >&2 <<EOF

  WARNING: this runtime is Python $PYMM, but requirements.txt pins wheels for
  3.10 (see .python-version). Some pins may be unsatisfiable. The campaign will
  still run and will record the ACTUAL versions in every result's provenance —
  but do not describe the environment as matching the paper's stack.

  To get 3.10 on Colab, select a runtime that provides it, or accept the
  difference and let provenance record it.

EOF
fi

if [[ "$REPORT_ONLY" -eq 1 ]]; then
    say "installed versions (report only)"
    python colab/check_colab_environment.py
    exit $?
fi

say "torch 2.1.0 + torchvision 0.16.0 (CUDA 12.1 build)"
# --index-url points only this call at the CUDA wheel index.
pip install --quiet \
    torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121 \
  || { echo "cu121 wheels unavailable for Python $PYMM; falling back to the" \
            "default index (this may install a CPU build)" >&2
       pip install --quiet torch==2.1.0 torchvision==0.16.0; }

say "flower + ray"
# ray is pinned explicitly although flwr[simulation] pulls it transitively:
# every virtual client executes inside a Ray worker, so its version is part of
# the reproducible surface.
pip install --quiet "flwr[simulation]==1.6.0" "ray==2.6.3"

say "scientific stack"
pip install --quiet \
    scikit-learn==1.3.2 \
    pandas==2.1.3 \
    numpy==1.26.2 \
    scipy==1.11.4 \
    imbalanced-learn==0.11.0 \
    hdbscan==0.8.33 \
    matplotlib==3.8.2 \
    seaborn==0.13.0 \
    pyyaml==6.0.1 \
    tqdm==4.66.1 \
    statsmodels==0.14.1

say "tensorboard (imported by utils/logger.py even when event files are off)"
pip install --quiet tensorboard==2.15.1

say "dataset"
if [[ -f data/raw/KDDTrain+.txt && -f data/raw/KDDTest+.txt ]]; then
    echo "NSL-KDD already present:"
    ls -l data/raw/KDDTrain+.txt data/raw/KDDTest+.txt
else
    bash scripts/download_nslkdd.sh
fi

say "what actually landed"
python colab/check_colab_environment.py
ENV_RC=$?

say "the paper's stated stack, for the record"
python scripts/verify_environment.py || true

say "campaign inventory"
python colab/campaign_inventory.py summary

if [[ "$ENV_RC" -ne 0 ]]; then
    echo
    echo "  Environment check reported errors above. Fix them before starting"
    echo "  the campaign; do not proceed on the assumption they are cosmetic."
fi
exit "$ENV_RC"
