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
                 use_tensorboard: Optional[bool] = None):
        # TensorBoard event files are a convenience view, never the scientific
        # record -- experiment_log.json is. On a campaign whose log_dirs live on
        # a slow (9p / network) filesystem the per-round flush dominates the
        # round, so TVFLIDS_TENSORBOARD=0 turns it off for the whole campaign
        # without changing a single logged value.
        if use_tensorboard is None:
            use_tensorboard = os.getenv("TVFLIDS_TENSORBOARD", "1") != "0"
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.start_time = time.time()
        import datetime as _dt
        self._start_utc = (_dt.datetime.now(_dt.timezone.utc)
                           .isoformat().replace("+00:00", "Z"))
        os.makedirs(log_dir, exist_ok=True)

        self.config: Dict[str, Any] = {}
        self.round_logs: List[Dict[str, Any]] = []
        self.summary: Dict[str, Any] = {}
        # Free-form per-run artifacts that are neither a per-round metric nor a
        # scalar summary field -- currently the per-client trust trajectories
        # and the strategy's own round logs, which back manuscript Figures 3
        # and 4. Without these on disk, those two figures had no regeneration
        # path: the data existed only in memory on the live strategy object.
        self.extra: Dict[str, Any] = {}

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
        import hashlib, json as _json
        config_str = _json.dumps(config, sort_keys=True, default=str)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]
        self.config["_config_hash"] = config_hash
        print(f"[Logger] Config hash: {config_hash}")

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

    def log_extra(self, key: str, value: Any) -> None:
        """Attach a named per-run artifact to the saved log.

        Used for data a figure needs but that is not a per-round scalar
        metric, e.g. ``trust_history`` (Figure 3) and ``strategy_round_logs``
        (Figure 4). Written under the ``extra`` key of experiment_log.json and
        read back by scripts/generate_manuscript_figures.py.
        """
        self.extra[key] = value

    def save(self) -> str:
        """Save all logs to disk. Returns path to log file.

        Every saved run carries a provenance block (git commit + dirty-tree
        state, package versions, hardware, interpreter, timestamps and the
        simulation-parallelism knob). Without it a result on disk cannot be
        traced back to the code and environment that produced it, which is
        exactly the gap the forensic audit found in the quarantined artifacts.
        """
        from utils.provenance import run_provenance
        output = {
            "experiment_name": self.experiment_name,
            "config": self.config,
            "rounds": self.round_logs,
            "summary": self.summary,
            "extra": self.extra,
            "elapsed_seconds": time.time() - self.start_time,
            "provenance": run_provenance(
                extra={
                    "started_utc": self._start_utc,
                    "config_hash": self.config.get("_config_hash"),
                    "log_dir": os.path.abspath(self.log_dir),
                }
            ),
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
