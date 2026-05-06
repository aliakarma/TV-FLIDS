"""
experiments/run_experiment.py
Main experiment runner for TV-FLIDS.

Executes a full FL simulation with a specified strategy and attack config.
Handles data loading, client creation, Flower simulation, evaluation, and logging.

Usage:
    python experiments/run_experiment.py --strategy tvflids --attack label_flip_30
    python experiments/run_experiment.py --strategy fedavg  --attack no_attack
    python experiments/run_experiment.py --strategy fltrust --attack label_flip_30
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple

import flwr as fl
import yaml

# ── Path fix for module resolution ───────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.seed import set_all_seeds, get_device
from utils.logger import ExperimentLogger
from data.preprocessing.nslkdd_pipeline import (
    build_pipeline as nslkdd_pipeline, download_nslkdd, CLASS_NAMES
)
from data.partitioning import (
    get_partitioner
)
from models.mlp import IDSMLP, build_model
from fl.client import TVFLIDSClient
from fl.strategy import TVFLIDSStrategy
from fl.baselines.fedavg_strategy import FedAvgStrategy
from fl.baselines.krum_strategy import KrumStrategy
from fl.baselines.trimmed_mean_strategy import TrimmedMeanStrategy
from fl.baselines.fltrust_strategy import FLTrustStrategy
from fl.baselines.foolsgold_strategy import FoolsGoldStrategy
from attacks.adversarial import ATTACK_CONFIGS, get_malicious_client_ids
from evaluation.metrics import ExperimentMetrics
from evaluation.overhead import OverheadTracker, estimate_communication_cost


# ── Config Loader ─────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Data Setup ────────────────────────────────────────────────────────────────

def setup_data(config: dict, dataset: str = "nslkdd", seed: int = 42,
               partition_type: str = "noniid", alpha: float = 0.5):
    """
    Download, preprocess, and partition dataset into client shards.

    Returns:
        client_data, X_test, y_test, X_val, y_val, class_weights
    """
    fl_cfg = config["federated_learning"]
    num_clients = fl_cfg["num_clients"]

    with open("config/dataset_config.yaml", "r") as f:
        ds_cfg = yaml.safe_load(f).get("datasets", {}).get(dataset)
    if not ds_cfg:
        raise ValueError(f"Unknown dataset '{dataset}' in dataset_config.yaml")
    train_path = ds_cfg.get("train_file")
    test_path = ds_cfg.get("test_file")

    if dataset == "nslkdd":
        download_nslkdd(train_path, test_path)
        (X_train, y_train,
         X_val, y_val,
         X_test, y_test,
         _, _, class_weights) = nslkdd_pipeline(
            train_path, test_path, use_smote=True, seed=seed
        )
    else:
        raise NotImplementedError(f"Dataset '{dataset}' not yet integrated. "
                                   "Use 'nslkdd'.")

    # Partition training data across clients
    partitioner = get_partitioner(partition_type, alpha=alpha)
    client_data = partitioner.partition(X_train, y_train, num_clients, seed=seed)

    print(f"[Data] {dataset.upper()} | Clients={num_clients} | "
          f"Partition={partition_type}(α={alpha}) | "
            f"Train={X_train.shape} | Test={X_test.shape} | Val={X_val.shape}")

    return client_data, X_test, y_test, X_val, y_val, class_weights


# ── Client Factory ────────────────────────────────────────────────────────────

def make_client_fn(
    client_data: List[Tuple[np.ndarray, np.ndarray]],
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    device: torch.device,
    class_weights: np.ndarray,
    malicious_ids: List[int],
    attack_type: Optional[str],
    attack_kwargs: dict,
    model_kwargs: dict,
):
    """Return a Flower client factory function."""

    def client_fn(cid: str) -> fl.client.NumPyClient:
        client_id = int(cid)
        X_c, y_c = client_data[client_id]

        # Split client data: 80% train, 20% local val
        n_train = int(0.8 * len(X_c))
        X_tr, y_tr = X_c[:n_train], y_c[:n_train]
        X_lv, y_lv = X_c[n_train:], y_c[n_train:]

        # If local val is empty, use a tiny slice of training
        if len(X_lv) == 0:
            X_lv, y_lv = X_val[:50], y_val[:50]

        is_malicious = client_id in malicious_ids

        return TVFLIDSClient(
            client_id=client_id,
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_lv,
            y_val=y_lv,
            device=device,
            config=config["federated_learning"],
            class_weights=class_weights,
            model_class=IDSMLP,
            model_kwargs=model_kwargs,
            is_malicious=is_malicious,
            attack_type=attack_type if is_malicious else None,
            attack_kwargs=attack_kwargs,
        )

    return client_fn


# ── Strategy Factory ─────────────────────────────────────────────────────────

def make_strategy(
    strategy_name: str,
    config: dict,
    global_model: nn.Module,
    val_loader: DataLoader,
    root_loader: Optional[DataLoader],
    device: torch.device,
    num_clients: int,
    fl_cfg: dict,
    evaluate_fn=None,
):
    """Instantiate the requested FL strategy."""
    frac_fit  = fl_cfg.get("fraction_fit", 0.5)
    frac_eval = fl_cfg.get("fraction_evaluate", 0.3)

    common_kwargs = dict(
        fraction_fit=frac_fit,
        fraction_evaluate=frac_eval,
        min_fit_clients=max(2, int(num_clients * frac_fit)),
        min_evaluate_clients=max(1, int(num_clients * frac_eval)),
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
    )

    adv_ratio = config.get("adversarial", {}).get("attack_ratio", 0.3)
    n_byzantine = max(1, int(num_clients * adv_ratio))

    if strategy_name == "fedavg":
        return FedAvgStrategy(**common_kwargs)

    elif strategy_name == "krum":
        return KrumStrategy(
            num_clients=num_clients,
            num_byzantine=n_byzantine,
            **common_kwargs,
        )

    elif strategy_name == "trimmed_mean":
        beta = min(0.35, adv_ratio + 0.05)
        return TrimmedMeanStrategy(beta=beta, **common_kwargs)

    elif strategy_name == "fltrust":
        if root_loader is None:
            raise ValueError("FLTrust requires a root_loader. Pass --strategy fltrust.")
        return FLTrustStrategy(
            server_model=global_model,
            server_root_loader=root_loader,
            device=device,
            local_epochs=1,
            lr=fl_cfg.get("local_lr", 0.001),
            **common_kwargs,
        )

    elif strategy_name == "foolsgold":
        return FoolsGoldStrategy(num_clients=num_clients, **common_kwargs)

    elif strategy_name in ("tvflids", "tvflids_adaptive", "tvflids_fixed"):
        adaptive = (strategy_name != "tvflids_fixed")
        return TVFLIDSStrategy(
            num_clients=num_clients,
            config=config,
            val_loader=val_loader,
            model=global_model,
            device=device,
            adaptive=adaptive,
            use_adaptive_thresholds=False,
            **common_kwargs,
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}. "
                         "Choose: fedavg, krum, trimmed_mean, fltrust, "
                         "foolsgold, tvflids, tvflids_fixed")


# ── Global Model Evaluator ────────────────────────────────────────────────────

def evaluate_global_model(
    model: nn.Module,
    parameters: List[np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    metrics_tracker: ExperimentMetrics,
    round_num: int,
    trust_scores: Optional[np.ndarray] = None,
) -> Dict:
    """Load parameters into model and evaluate on test set."""
    model.set_parameters(parameters)
    model.eval()

    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_test, dtype=torch.long).to(device)

    batch_size = 512
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            logits = model(X_t[i:i + batch_size])
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    return metrics_tracker.compute_round_metrics(
        y_test, y_pred, round_num, trust_scores=trust_scores
    )


def predict_global_model(
    model: nn.Module,
    parameters: List[np.ndarray],
    X_test: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Load parameters into model and return predictions on test set."""
    model.set_parameters(parameters)
    model.eval()

    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    batch_size = 512
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            logits = model(X_t[i:i + batch_size])
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds)


# ── Main Runner ───────────────────────────────────────────────────────────────

def run_experiment(
    strategy_name: str = "tvflids",
    attack_config_name: str = "label_flip_30",
    dataset: str = "nslkdd",
    partition_type: str = "noniid",
    alpha: float = 0.5,
    seed: int = 42,
    num_rounds: Optional[int] = None,
    config_path: str = "config/fl_config.yaml",
    log_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """
    Run a complete FL experiment end-to-end.

    Args:
        strategy_name:      FL strategy to use.
        attack_config_name: Key from ATTACK_CONFIGS dict.
        dataset:            Dataset name ('nslkdd').
        partition_type:     'iid' or 'noniid'.
        alpha:              Dirichlet α for non-IID partitioning.
        seed:               Random seed.
        num_rounds:         Override config rounds.
        config_path:        Path to FL config YAML.
        log_dir:            Override default log directory.
        verbose:            Print progress.

    Returns:
        Dict with final metrics summary.
    """
    # ── Seed and device setup ─────────────────────────────────────────
    set_all_seeds(seed)
    device = get_device()

    # ── Load config ───────────────────────────────────────────────────
    config = load_config(config_path)
    fl_cfg = config["federated_learning"]

    if num_rounds is not None:
        fl_cfg["num_rounds"] = num_rounds
    n_rounds     = fl_cfg["num_rounds"]
    num_clients  = fl_cfg["num_clients"]

    # ── Attack setup ──────────────────────────────────────────────────
    if attack_config_name not in ATTACK_CONFIGS:
        raise ValueError(f"Unknown attack config: {attack_config_name}. "
                         f"Available: {list(ATTACK_CONFIGS.keys())}")
    atk_cfg = ATTACK_CONFIGS[attack_config_name]

    # Override config with attack settings
    config["adversarial"]["attack_ratio"] = atk_cfg["ratio"]
    config["adversarial"]["attack_type"]  = atk_cfg.get("type", "label_flip")

    attack_type = atk_cfg.get("type")
    attack_kwargs = {
        "scale_factor":  atk_cfg.get("factor", 10.0),
        "noise_std":     atk_cfg.get("std", 0.5),
        "poison_ratio":  atk_cfg.get("poison_ratio", 0.1),
        "flip_ratio":    1.0,
        "target_class":  0,
    }
    attack_kwargs["seed"] = seed

    malicious_ids = get_malicious_client_ids(
        num_clients, atk_cfg["ratio"], seed=seed
    )
    if verbose:
        print(f"\n{'='*60}")
        print(f" Strategy:  {strategy_name.upper()}")
        print(f" Attack:    {attack_config_name} ({len(malicious_ids)} malicious)")
        print(f" Dataset:   {dataset.upper()} | {partition_type}(α={alpha})")
        print(f" Seed:      {seed}  |  Rounds: {n_rounds}")
        print(f"{'='*60}\n")

    # ── Logging ───────────────────────────────────────────────────────
    if log_dir is None:
        log_dir = (f"results/logs/{strategy_name}_{attack_config_name}_"
                   f"{partition_type}_seed{seed}")
    logger = ExperimentLogger(log_dir, experiment_name=strategy_name)
    logger.log_config({
        "strategy": strategy_name, "attack": attack_config_name,
        "dataset": dataset, "partition_type": partition_type,
        "alpha": alpha, "seed": seed, **fl_cfg,
    })

    # ── Data preparation ──────────────────────────────────────────────
    client_data, X_test, y_test, X_val, y_val, class_weights = setup_data(
        config, dataset=dataset, seed=seed,
        partition_type=partition_type, alpha=alpha,
    )

    input_dim  = X_test.shape[1]
    num_classes = len(np.unique(y_test))
    model_kwargs = {"input_dim": input_dim, "num_classes": num_classes}

    # ── Global model ──────────────────────────────────────────────────
    global_model = IDSMLP(**model_kwargs).to(device)

    # ── Server validation DataLoader ──────────────────────────────────
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=256, shuffle=False,
    )

    # ── Root loader for FLTrust ───────────────────────────────────────
    root_loader = None
    if strategy_name == "fltrust":
        root_size = max(50, int(len(X_val) * 0.5))
        X_root = X_val[:root_size]
        y_root = y_val[:root_size]
        root_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_root, dtype=torch.float32),
                torch.tensor(y_root, dtype=torch.long),
            ),
            batch_size=64, shuffle=True,
        )

    # ── Metrics tracker and overhead ──────────────────────────────────
    metrics_tracker = ExperimentMetrics(class_names=CLASS_NAMES[:num_classes])
    overhead_tracker = OverheadTracker()

    # Store round-by-round results
    round_results: List[Dict] = []
    last_params: Optional[List[np.ndarray]] = None

    class _StrategyContainer:
        strategy = None

    strategy_container = _StrategyContainer()

    def evaluate_fn(server_round: int, parameters, config_eval):
        """Flower evaluate_fn called after each round."""
        if isinstance(parameters, list):
            params_np = parameters
        else:
            params_np = fl.common.parameters_to_ndarrays(parameters)
        nonlocal last_params
        last_params = params_np
        trust_scores = None
        if (
            hasattr(strategy_container, "strategy")
            and hasattr(strategy_container.strategy, "trust_scorer")
        ):
            trust_scores = strategy_container.strategy.trust_scorer.trust_scores.copy()

        m = evaluate_global_model(
            global_model, params_np, X_test, y_test,
            device, metrics_tracker, server_round, trust_scores,
        )
        round_results.append(m)
        metric_floats = {
            k: v for k, v in m.items()
            if isinstance(v, (int, float)) and v is not None
        }
        if (
            hasattr(strategy_container, "strategy")
            and hasattr(strategy_container.strategy, "trust_scorer")
        ):
            ts_summary = strategy_container.strategy.trust_scorer.get_summary()
            metric_floats.update({f"trust_{k}": float(v) for k, v in ts_summary.items()})
        logger.log_round(server_round, metric_floats)

        if verbose:
            print(f"  [Eval R{server_round:03d}] "
                  f"Acc={m['accuracy']:.4f} | "
                  f"F1={m['f1_macro']:.4f} | "
                  f"ASR={m['attack_success_rate']:.4f}")

        return m["accuracy"], {
            "f1_macro": m["f1_macro"],
            "attack_success_rate": m["attack_success_rate"],
        }

    # ── Strategy ──────────────────────────────────────────────────────
    strategy = make_strategy(
        strategy_name, config, global_model, val_loader, root_loader,
        device, num_clients, fl_cfg, evaluate_fn=evaluate_fn,
    )
    strategy_container.strategy = strategy

    # ── Client factory ────────────────────────────────────────────────
    client_fn = make_client_fn(
        client_data, X_val, y_val, config, device,
        class_weights, malicious_ids, attack_type, attack_kwargs, model_kwargs,
    )

    # ── Flower simulation ─────────────────────────────────────────────
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    # ── Persist final predictions for confusion matrices ─────────────
    final_params = last_params if last_params is not None else global_model.get_parameters()
    y_pred_final = predict_global_model(global_model, final_params, X_test, device)
    preds_path = os.path.join(log_dir, "final_predictions.npz")
    np.savez(preds_path, y_true=y_test, y_pred=y_pred_final)

    # ── Figure generation (post-simulation) ──────────────────────────
    if strategy_name in ("tvflids", "tvflids_fixed") and len(round_results) > 0:
        try:
            from evaluation.visualization import (
                figure1_convergence_curves,
                figure2_trust_evolution,
            )
            os.makedirs("results/figures", exist_ok=True)
            figure1_convergence_curves(
                {strategy_name: round_results},
                save_path=(
                    f"results/figures/fig1_{strategy_name}_"
                    f"{attack_config_name}_seed{seed}.pdf"
                ),
            )
            if hasattr(strategy, "trust_scorer"):
                figure2_trust_evolution(
                    strategy.get_trust_history(),
                    malicious_ids,
                    save_path=(
                        f"results/figures/fig2_trust_{strategy_name}_"
                        f"seed{seed}.pdf"
                    ),
                )
        except Exception as e:
            print(f"[Warning] Figure generation failed: {e}")

    if strategy_name in ("fedavg", "tvflids") and attack_config_name == "label_flip_30":
        peer = "tvflids" if strategy_name == "fedavg" else "fedavg"
        if strategy_name in log_dir:
            peer_log_dir = log_dir.replace(strategy_name, peer, 1)
            peer_preds_path = os.path.join(peer_log_dir, "final_predictions.npz")
            if os.path.exists(peer_preds_path) and os.path.exists(preds_path):
                try:
                    from evaluation.visualization import figure6_confusion_matrices
                    peer_data = np.load(peer_preds_path)
                    this_data = np.load(preds_path)
                    if strategy_name == "fedavg":
                        y_pred_fedavg = this_data["y_pred"]
                        y_pred_tvflids = peer_data["y_pred"]
                    else:
                        y_pred_fedavg = peer_data["y_pred"]
                        y_pred_tvflids = this_data["y_pred"]
                    y_true = this_data["y_true"]
                    figure6_confusion_matrices(
                        y_true=y_true,
                        y_pred_fedavg=y_pred_fedavg,
                        y_pred_tvflids=y_pred_tvflids,
                        class_names=CLASS_NAMES[:num_classes],
                        save_path="results/figures/fig6_confusion.pdf",
                    )
                except Exception as e:
                    print(f"[Warning] Figure 6 generation failed: {e}")

    # ── Communication overhead ────────────────────────────────────────
    n_active = max(2, int(num_clients * fl_cfg.get("fraction_fit", 0.5)))
    comm_stats = estimate_communication_cost(
        model_params=global_model.count_parameters(),
        num_active_clients=n_active,
    )

    # ── Final summary ─────────────────────────────────────────────────
    summary = metrics_tracker.get_final_summary()
    summary.update({
        "strategy":           strategy_name,
        "attack":             attack_config_name,
        "seed":               seed,
        "num_malicious":      len(malicious_ids),
        "malicious_ids":      malicious_ids,
        "comm_overhead_pct":  comm_stats["overhead_pct"],
        "model_params":       global_model.count_parameters(),
    })

    logger.log_summary(summary)
    logger.save()

    if verbose:
        print(f"\n[Summary] Strategy={strategy_name} | Attack={attack_config_name}")
        print(f"  Final Accuracy:      {summary.get('final_accuracy', 0):.4f}")
        print(f"  Final F1-Macro:      {summary.get('final_f1_macro', 0):.4f}")
        print(f"  Attack Success Rate: {summary.get('final_attack_success_rate', 0):.4f}")

    return summary


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TV-FLIDS Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/run_experiment.py --strategy tvflids --attack label_flip_30
  python experiments/run_experiment.py --strategy fedavg  --attack no_attack
  python experiments/run_experiment.py --strategy fltrust --attack label_flip_30 --rounds 50
  python experiments/run_experiment.py --strategy tvflids --attack gradient_scale_30 --seed 123
        """
    )
    parser.add_argument("--strategy",   type=str, default="tvflids",
                        choices=["fedavg", "krum", "trimmed_mean", "fltrust",
                                 "foolsgold", "tvflids", "tvflids_fixed"],
                        help="Aggregation strategy")
    parser.add_argument("--attack",     type=str, default="label_flip_30",
                        choices=list(ATTACK_CONFIGS.keys()),
                        help="Attack configuration")
    parser.add_argument("--dataset",    type=str, default="nslkdd",
                        help="Dataset name (nslkdd)")
    parser.add_argument("--partition",  type=str, default="noniid",
                        choices=["iid", "noniid"],
                        help="Data partitioning strategy")
    parser.add_argument("--alpha",      type=float, default=0.5,
                        help="Dirichlet alpha for non-IID (0.5=moderate, 0.1=extreme)")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Random seed")
    parser.add_argument("--rounds",     type=int, default=None,
                        help="Override number of FL rounds")
    parser.add_argument("--config",     type=str, default="config/fl_config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--log_dir",    type=str, default=None,
                        help="Override log directory")
    parser.add_argument("--quiet",      action="store_true",
                        help="Suppress verbose output")

    args = parser.parse_args()

    result = run_experiment(
        strategy_name=args.strategy,
        attack_config_name=args.attack,
        dataset=args.dataset,
        partition_type=args.partition,
        alpha=args.alpha,
        seed=args.seed,
        num_rounds=args.rounds,
        config_path=args.config,
        log_dir=args.log_dir,
        verbose=not args.quiet,
    )

    print("\n[Done] Final results:")
    print(json.dumps({k: v for k, v in result.items()
                       if not isinstance(v, (list, dict))}, indent=2))


if __name__ == "__main__":
    main()
