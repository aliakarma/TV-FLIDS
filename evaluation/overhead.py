"""
evaluation/overhead.py
Computational and communication overhead analysis.
Reference: Guide §16
"""

import time
import numpy as np
from typing import Dict, List, Optional


class OverheadTracker:
    """
    Times each phase of the aggregation pipeline per round.
    Computes overhead of TV-FLIDS relative to FedAvg.
    """

    def __init__(self):
        self.timings: Dict[str, List[float]] = {
            "verification":   [],
            "trust_scoring":  [],
            "aggregation":    [],
            "total":          [],
            "fedavg_total":   [],
        }

    def time_phase(self, phase: str):
        """Context manager for timing a code block."""
        return _Timer(self.timings, phase)

    def get_summary(self) -> Dict[str, float]:
        """Return mean timing stats across all rounds."""
        summary = {}
        for phase, times in self.timings.items():
            if times:
                summary[f"{phase}_mean_ms"] = float(np.mean(times) * 1000)
                summary[f"{phase}_std_ms"]  = float(np.std(times) * 1000)

        # Compute overhead percentage vs FedAvg
        if self.timings["fedavg_total"] and self.timings["total"]:
            fedavg_mean = np.mean(self.timings["fedavg_total"])
            tvflids_mean = np.mean(self.timings["total"])
            if fedavg_mean > 0:
                summary["overhead_pct"] = float(
                    (tvflids_mean - fedavg_mean) / fedavg_mean * 100
                )
        return summary

    def print_report(self) -> None:
        summary = self.get_summary()
        print("\n[Overhead Report]")
        print(f"  Verification gate:  {summary.get('verification_mean_ms', 0):.1f} ms/round")
        print(f"  Trust scoring:      {summary.get('trust_scoring_mean_ms', 0):.1f} ms/round")
        print(f"  Aggregation:        {summary.get('aggregation_mean_ms', 0):.1f} ms/round")
        print(f"  TV-FLIDS total:     {summary.get('total_mean_ms', 0):.1f} ms/round")
        print(f"  FedAvg total:       {summary.get('fedavg_total_mean_ms', 0):.1f} ms/round")
        if "overhead_pct" in summary:
            print(f"  Overhead vs FedAvg: {summary['overhead_pct']:.1f}%")


class _Timer:
    """Context manager for phase timing."""

    def __init__(self, store: Dict[str, List[float]], phase: str):
        self.store = store
        self.phase = phase
        self.start = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        if self.phase not in self.store:
            self.store[self.phase] = []
        self.store[self.phase].append(elapsed)


def estimate_communication_cost(
    model_params: int,
    num_active_clients: int,
    dtype_bytes: int = 4,
) -> Dict[str, float]:
    """
    Estimate per-round communication cost in MB.

    TV-FLIDS adds only 4 bytes (val_loss scalar) per client vs FedAvg.
    Reference: Guide §16.2

    Args:
        model_params:       Number of model parameters.
        num_active_clients: Clients participating per round.
        dtype_bytes:        Bytes per parameter (4 for float32).
    """
    param_bytes = model_params * dtype_bytes
    param_mb = param_bytes / (1024 ** 2)

    fedavg_mb = 2 * num_active_clients * param_mb
    tvflids_mb = fedavg_mb + num_active_clients * 4 / (1024 ** 2)

    return {
        "model_params":         model_params,
        "param_size_mb":        param_mb,
        "fedavg_total_mb":      fedavg_mb,
        "tvflids_total_mb":     tvflids_mb,
        "overhead_bytes":       num_active_clients * 4,
        "overhead_pct":         (tvflids_mb - fedavg_mb) / fedavg_mb * 100,
    }
