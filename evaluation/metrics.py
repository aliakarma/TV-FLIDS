"""
evaluation/metrics.py
Comprehensive metrics computation for TV-FLIDS experiments.

Tracks per-round and final metrics:
  - Accuracy, F1-Macro, F1-Weighted, per-class F1
  - Attack Success Rate (key IDS metric)
  - False Negative Rate
  - Trust score statistics

Reference: Guide §10.2
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)


class ExperimentMetrics:
    """
    Tracks and computes all metrics for a TV-FLIDS experiment run.
    """

    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or ["Normal", "DoS", "Probe", "R2L", "U2R"]
        self.round_metrics: List[Dict] = []

    def compute_round_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        round_num: int,
        trust_scores: Optional[np.ndarray] = None,
    ) -> Dict:
        """Compute all metrics for a single FL round's global model."""
        accuracy = float(accuracy_score(y_true, y_pred))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

        # Attack Success Rate: fraction of TRUE attacks classified as Normal
        attack_mask = y_true != 0
        if attack_mask.sum() > 0:
            attack_success_rate = float((y_pred[attack_mask] == 0).mean())
        else:
            attack_success_rate = 0.0

        # False Negative Rate: attacks predicted as Normal
        fn_idx = np.where(y_true != 0)[0]
        fn_rate = float((y_pred[fn_idx] == 0).mean()) if len(fn_idx) > 0 else 0.0

        metrics = {
            "round":               round_num,
            "accuracy":            accuracy,
            "f1_macro":            f1_macro,
            "f1_weighted":         f1_weighted,
            "f1_per_class":        f1_per_class,
            "attack_success_rate": attack_success_rate,
            "false_negative_rate": fn_rate,
            "trust_mean":          float(np.mean(trust_scores)) if trust_scores is not None else None,
            "trust_min":           float(np.min(trust_scores)) if trust_scores is not None else None,
            "trust_max":           float(np.max(trust_scores)) if trust_scores is not None else None,
        }
        self.round_metrics.append(metrics)
        return metrics

    def compute_final_report(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[str, np.ndarray]:
        """Generate final classification report and confusion matrix."""
        n_classes = len(self.class_names)
        labels_present = sorted(set(y_true) | set(y_pred))
        names = [self.class_names[i] for i in labels_present
                 if i < len(self.class_names)]

        report = classification_report(
            y_true, y_pred,
            labels=labels_present,
            target_names=names,
            zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
        return report, cm

    def get_metric_series(self, metric: str) -> List[float]:
        """Extract a metric time-series across all rounds."""
        return [r[metric] for r in self.round_metrics if metric in r and r[metric] is not None]

    def get_convergence_round(self, target_accuracy: float = 0.90) -> Optional[int]:
        """Round at which model first reaches target accuracy."""
        for m in self.round_metrics:
            if m["accuracy"] >= target_accuracy:
                return m["round"]
        return None

    def get_final_summary(self) -> Dict:
        """Aggregate final statistics across all rounds."""
        if not self.round_metrics:
            return {}
        last = self.round_metrics[-1]
        acc_series = self.get_metric_series("accuracy")
        return {
            "final_accuracy":            last["accuracy"],
            "final_f1_macro":            last["f1_macro"],
            "final_attack_success_rate": last["attack_success_rate"],
            "final_false_negative_rate": last["false_negative_rate"],
            "peak_accuracy":             max(acc_series) if acc_series else 0.0,
            "num_rounds":                len(self.round_metrics),
        }

    def reset(self) -> None:
        self.round_metrics = []
