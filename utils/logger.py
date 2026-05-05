"""
utils/logger.py — Structured experiment logger.
Writes per-round metrics to JSON and optionally to TensorBoard.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


class ExperimentLogger:
    """
    Logs experiment metadata and per-round metrics.

    Usage:
        logger = ExperimentLogger(log_dir="results/logs/exp1")
        logger.log_config(config)
        logger.log_round(round=1, metrics={"accuracy": 0.91, "f1_macro": 0.88})
        logger.save()
    """

    def __init__(self, log_dir: str, experiment_name: str = "experiment",
                 use_tensorboard: bool = True):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.start_time = time.time()
        os.makedirs(log_dir, exist_ok=True)

        self.config: Dict[str, Any] = {}
        self.round_logs: List[Dict[str, Any]] = []
        self.summary: Dict[str, Any] = {}

        self.writer = None
        if use_tensorboard and _TB_AVAILABLE:
            tb_dir = os.path.join(log_dir, "tensorboard")
            self.writer = SummaryWriter(log_dir=tb_dir)

        print(f"[Logger] Initialized: {log_dir}")

    def log_config(self, config: Dict[str, Any]) -> None:
        """Save experiment configuration."""
        self.config = config
        config_path = os.path.join(self.log_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

    def log_round(self, round_num: int, metrics: Dict[str, Any]) -> None:
        """Log metrics for a single FL round."""
        entry = {"round": round_num, "timestamp": time.time(), **metrics}
        self.round_logs.append(entry)

        if self.writer is not None:
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    self.writer.add_scalar(f"metrics/{key}", val, round_num)

        # Print to console for real-time monitoring
        metric_str = " | ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
            if not isinstance(v, (list, dict))
        )
        print(f"[Round {round_num:03d}] {metric_str}")

    def log_summary(self, summary: Dict[str, Any]) -> None:
        """Log final experiment summary."""
        self.summary = summary
        summary["elapsed_seconds"] = time.time() - self.start_time

    def save(self) -> str:
        """Save all logs to disk. Returns path to log file."""
        output = {
            "experiment_name": self.experiment_name,
            "config": self.config,
            "rounds": self.round_logs,
            "summary": self.summary,
            "elapsed_seconds": time.time() - self.start_time,
        }
        log_path = os.path.join(self.log_dir, "experiment_log.json")
        with open(log_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        if self.writer is not None:
            self.writer.close()

        print(f"[Logger] Saved to {log_path}")
        return log_path

    def get_metric_series(self, metric: str) -> List[float]:
        """Extract a metric across all rounds for plotting."""
        return [r[metric] for r in self.round_logs if metric in r]
