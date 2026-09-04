"""
audit_hyperparams.py
Thin backward-compatible CLI shim (repo root, historical location).

Previously this script only printed the currently-active values from
config/fl_config.yaml and did not sweep anything, despite its name (audit
finding, IDs E10/E11). The real sweep logic now lives in
experiments/run_hyperparameter_sweep.py::run_baseline_hyperparameter_sweep()
and run_tvflids_hyperparameter_sweep(). This shim just forwards to it so
`python audit_hyperparams.py` keeps working from the repo root.

Usage:
    python audit_hyperparams.py --target baseline
    python audit_hyperparams.py --target tvflids
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.run_hyperparameter_sweep import main as _sweep_main

if __name__ == "__main__":
    _sweep_main()
