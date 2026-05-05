"""
evaluation/visualization.py
Publication-grade figure generation for TV-FLIDS paper.

Six figures:
  Figure 1 — Convergence curves (accuracy + F1-macro vs round)
  Figure 2 — Trust score evolution per client
  Figure 3 — Robustness curve (accuracy vs adversarial ratio)
  Figure 4 — Ablation bar chart
  Figure 5 — Adaptive weight trajectories (α, β, γ)
  Figure 6 — Side-by-side confusion matrices

All figures: 300 DPI, PDF, Times New Roman, colorblind-safe palette.
Reference: Guide §10.4
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive — safe for server/headless use
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Dict, List, Optional, Tuple

try:
    import seaborn as sns
    _SNS = True
except ImportError:
    _SNS = False

try:
    from sklearn.metrics import confusion_matrix
    _SKL = True
except ImportError:
    _SKL = False

# ── Publication Style ─────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":          11,
    "axes.labelsize":     12,
    "axes.titlesize":     12,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    9,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.format":     "pdf",
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "lines.linewidth":    1.8,
})

# Colorblind-safe palette (IEEE-compatible)
PALETTE = {
    "FedAvg":        "#d62728",
    "Krum":          "#ff7f0e",
    "TrimmedMean":   "#9467bd",
    "FLTrust":       "#2ca02c",
    "FoolsGold":     "#8c564b",
    "TV-FLIDS":      "#1f77b4",
    "TV-FLIDS-Adp":  "#17becf",
}
LINESTYLES = {
    "FedAvg":       "--",
    "Krum":         ":",
    "TrimmedMean":  "-.",
    "FLTrust":      (0, (5, 1)),
    "FoolsGold":    (0, (3, 1, 1, 1)),
    "TV-FLIDS":     "-",
    "TV-FLIDS-Adp": "-",
}

def _save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure] Saved → {path}")
    return path


# ── Figure 1: Convergence Curves ─────────────────────────────────────────────

def figure1_convergence_curves(
    round_metrics_dict: Dict[str, List[Dict]],
    save_path: str = "results/figures/fig1_convergence.pdf",
) -> str:
    """
    Accuracy and F1-Macro vs FL Round for all methods.

    Args:
        round_metrics_dict: {method: [{"round": t, "accuracy": v, "f1_macro": v}]}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    for method, metrics in round_metrics_dict.items():
        rounds = [m["round"] for m in metrics]
        accs   = [m.get("accuracy", 0.0) for m in metrics]
        f1s    = [m.get("f1_macro", 0.0) for m in metrics]

        color = PALETTE.get(method, "gray")
        ls    = LINESTYLES.get(method, "-")
        lw    = 2.5 if "TV-FLIDS" in method else 1.5
        zorder = 3 if "TV-FLIDS" in method else 2

        ax1.plot(rounds, accs, color=color, linestyle=ls, linewidth=lw,
                 label=method, zorder=zorder)
        ax2.plot(rounds, f1s, color=color, linestyle=ls, linewidth=lw,
                 label=method, zorder=zorder)

    for ax, ylabel, title in [
        (ax1, "Test Accuracy",  "(a) Accuracy vs. Communication Round"),
        (ax2, "F1-Macro",       "(b) F1-Macro vs. Communication Round"),
    ]:
        ax.set_xlabel("Communication Round")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="lower right", framealpha=0.9, fontsize=8)
        ax.set_xlim(left=1)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    plt.suptitle(
        "Convergence under 30% Label Flip Attack (NSL-KDD, Non-IID α=0.5)",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    return _save(fig, save_path)


# ── Figure 2: Trust Score Evolution ──────────────────────────────────────────

def figure2_trust_evolution(
    trust_history_dict: Dict[int, List[float]],
    malicious_ids: List[int],
    save_path: str = "results/figures/fig2_trust_evolution.pdf",
) -> str:
    """
    Per-client trust score T_i(t) over rounds.
    Malicious clients colored red; honest clients blue (faded).
    """
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(7, 4))

    n_honest = len(trust_history_dict) - len(malicious_ids)

    for client_id, history in trust_history_dict.items():
        rounds = list(range(1, len(history) + 1))
        if client_id in malicious_ids:
            ax.plot(rounds, history, color="#d62728", linewidth=1.5,
                    alpha=0.85, zorder=3)
        else:
            ax.plot(rounds, history, color="#1f77b4", linewidth=0.9,
                    alpha=0.35, zorder=2)

    legend_elements = [
        Line2D([0], [0], color="#1f77b4", lw=1.5, alpha=0.6,
               label=f"Honest Clients (n={n_honest})"),
        Line2D([0], [0], color="#d62728", lw=1.5,
               label=f"Malicious Clients (n={len(malicious_ids)})"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Trust Score $T_i(t)$")
    ax.set_title("Trust Score Evolution under 30% Label Flip Attack")
    if trust_history_dict:
        ax.set_xlim(1, max(len(h) for h in trust_history_dict.values()))
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    return _save(fig, save_path)


# ── Figure 3: Robustness Curve ────────────────────────────────────────────────

def figure3_robustness_curve(
    ratio_results_dict: Dict[str, Dict[float, Dict]],
    save_path: str = "results/figures/fig3_robustness_curve.pdf",
) -> str:
    """
    Accuracy vs Adversarial Ratio for all methods.

    Args:
        ratio_results_dict: {method: {ratio: {"accuracy_mean": v, "accuracy_std": v}}}
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    for method, ratio_data in ratio_results_dict.items():
        ratios = sorted(ratio_data.keys())
        means  = [ratio_data[r].get("accuracy_mean", 0.0) for r in ratios]
        stds   = [ratio_data[r].get("accuracy_std", 0.0) for r in ratios]

        color  = PALETTE.get(method, "gray")
        ls     = LINESTYLES.get(method, "-")
        lw     = 2.5 if "TV-FLIDS" in method else 1.5

        x_vals = [r * 100 for r in ratios]
        ax.plot(x_vals, means, color=color, linestyle=ls, linewidth=lw,
                marker="o", markersize=4, label=method,
                zorder=3 if "TV-FLIDS" in method else 2)
        ax.fill_between(
            x_vals,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            color=color, alpha=0.12,
        )

    ax.axvline(x=30, color="gray", linestyle=":", linewidth=1.0, alpha=0.8,
               label="Primary eval (30%)")
    ax.set_xlabel("Fraction of Adversarial Clients (%)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Robustness vs. Adversarial Participation Rate\n(Label Flip, NSL-KDD)")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax.set_xlim(0, 65)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    plt.tight_layout()
    return _save(fig, save_path)


# ── Figure 4: Ablation Bar Chart ──────────────────────────────────────────────

def figure4_ablation_bars(
    ablation_results: Dict[str, Dict[str, Tuple[float, float]]],
    save_path: str = "results/figures/fig4_ablation.pdf",
) -> str:
    """
    Ablation study bar chart: F1-Macro and Attack Success Rate per variant.

    Args:
        ablation_results: {
            "TV-FLIDS (Full)": {"f1_macro": (mean, std), "attack_success_rate": (mean, std)},
            "A1: No Verification": {...},
            ...
        }
    """
    labels    = list(ablation_results.keys())
    f1_means  = [ablation_results[l]["f1_macro"][0] for l in labels]
    f1_stds   = [ablation_results[l]["f1_macro"][1] for l in labels]
    asr_means = [ablation_results[l]["attack_success_rate"][0] for l in labels]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))

    bars1 = ax.bar(x - w / 2, f1_means, w, yerr=f1_stds, capsize=3,
                   color="#1f77b4", alpha=0.85, label="F1-Macro")
    bars2 = ax.bar(x + w / 2, asr_means, w,
                   color="#d62728", alpha=0.85, label="Attack Success Rate")

    # Highlight full method (first bar)
    if bars1:
        bars1[0].set_edgecolor("black")
        bars1[0].set_linewidth(2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study — Component Contribution (30% Label Flip, NSL-KDD)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    return _save(fig, save_path)


# ── Figure 5: Adaptive Weight Trajectories ────────────────────────────────────

def figure5_adaptive_weights(
    weight_history_per_attack: Dict[str, List[Dict[str, float]]],
    save_path: str = "results/figures/fig5_adaptive_weights.pdf",
) -> str:
    """
    Learned α, β, γ trajectories per attack type.

    Args:
        weight_history_per_attack: {
            "Label Flip": [{"alpha": v, "beta": v, "gamma": v}, ...],
            "Gradient Scale": [...],
        }
    """
    n_attacks = len(weight_history_per_attack)
    if n_attacks == 0:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return _save(fig, save_path)

    fig, axes = plt.subplots(1, n_attacks,
                              figsize=(4.5 * n_attacks, 3.5), sharey=True)
    if n_attacks == 1:
        axes = [axes]

    weight_colors = {"alpha": "#1f77b4", "beta": "#2ca02c", "gamma": "#d62728"}
    weight_labels = {
        "alpha": "α (Similarity)",
        "beta":  "β (Accuracy)",
        "gamma": "γ (Anomaly)",
    }

    for ax, (attack_name, history) in zip(axes, weight_history_per_attack.items()):
        rounds = list(range(1, len(history) + 1))
        for key in ["alpha", "beta", "gamma"]:
            vals = [h.get(key, 1.0 / 3.0) for h in history]
            ax.plot(rounds, vals, color=weight_colors[key], linewidth=1.8,
                    label=weight_labels[key])
        ax.set_title(attack_name, fontsize=10)
        ax.set_xlabel("Round")
        ax.set_ylim(0, 1)
        ax.legend(loc="center right", fontsize=7)

    axes[0].set_ylabel("Weight Value")
    plt.suptitle("Adaptive Trust Weights (α, β, γ) Evolution per Attack Type", y=1.02)
    plt.tight_layout()
    return _save(fig, save_path)


# ── Figure 6: Confusion Matrices ─────────────────────────────────────────────

def figure6_confusion_matrices(
    y_true: np.ndarray,
    y_pred_fedavg: np.ndarray,
    y_pred_tvflids: np.ndarray,
    class_names: List[str],
    save_path: str = "results/figures/fig6_confusion.pdf",
) -> str:
    """Side-by-side normalized confusion matrices: FedAvg vs TV-FLIDS."""
    if not _SKL:
        print("[Warning] sklearn not available for confusion matrix.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "sklearn required", ha="center")
        return _save(fig, save_path)

    n_classes = len(class_names)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, y_pred, title in [
        (ax1, y_pred_fedavg,  "FedAvg (30% Label Flip)"),
        (ax2, y_pred_tvflids, "TV-FLIDS (30% Label Flip)"),
    ]:
        cm = confusion_matrix(y_true, y_pred,
                               labels=list(range(n_classes)), normalize="true")
        if _SNS:
            sns.heatmap(
                cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, vmin=0, vmax=1,
                linewidths=0.3, linecolor="gray",
                annot_kws={"size": 9},
            )
        else:
            im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(n_classes))
            ax.set_yticks(range(n_classes))
            ax.set_xticklabels(class_names, rotation=45, ha="right")
            ax.set_yticklabels(class_names)
            for i in range(n_classes):
                for j in range(n_classes):
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                            fontsize=8, color="black")
            plt.colorbar(im, ax=ax)

        ax.set_title(title)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.tight_layout()
    return _save(fig, save_path)
