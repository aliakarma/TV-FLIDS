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
    build_pipeline as nslkdd_pipeline, download_nslkdd, CLASS_NAMES as NSLKDD_CLASS_NAMES
)
from data.preprocessing.unswnb15_pipeline import (
    build_pipeline as unswnb15_pipeline, CLASS_NAMES as UNSW_CLASS_NAMES
)
from data.partitioning import (
    get_partitioner
)
from models.mlp import IDSMLP, build_model
from fl.client import TVFLIDSClient
from fl.strategy import TVFLIDSStrategy
from fl.baselines.fedavg_strategy import FedAvgStrategy
from fl.baselines.krum_strategy import KrumStrategy
from fl.baselines.multikrum_strategy import MultiKrumStrategy
from fl.baselines.trimmed_mean_strategy import TrimmedMeanStrategy
from fl.baselines.norm_clipping_strategy import NormClippingStrategy
from fl.baselines.rfa_strategy import RFAStrategy
from fl.baselines.bucketing_strategy import BucketingStrategy
from fl.baselines.foolsgold_strategy import FoolsGoldStrategy
from fl.baselines.flame_strategy import FLAMEStrategy
from fl.baselines.deepsight_strategy import DeepSightStrategy
from fl.baselines.fldetector_strategy import FLDetectorStrategy
from fl.baselines.zeno_strategy import ZenoStrategy
from fl.baselines.fltrust_strategy import FLTrustStrategy
from fl.baselines.baffle_strategy import BaFFLeStrategy
from attacks.adversarial import ATTACK_CONFIGS, get_malicious_client_ids
from evaluation.metrics import ExperimentMetrics
from evaluation.overhead import OverheadTracker, estimate_communication_cost


# ── Config Loader ─────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Data Setup ────────────────────────────────────────────────────────────────

def setup_data(config: dict, dataset: str = "nslkdd", seed: int = 42,
               partition_type: str = "noniid", alpha: float = 0.5,
               val_size: int = 2000, protocol: str = "main"):
    """
    Download, preprocess, and partition dataset into client shards.

    Args:
        val_size: Absolute size of the stratified server validation set
            D_val (paper §IV-A / Table IV: 2,000). Previously this pipeline
            silently used an internal 5% fraction (~6,299 samples on
            NSL-KDD) that never matched the paper — see the audit finding
            N1. `create_server_validation_set(val_size=...)` is now the
            single source of truth for this split.
        protocol: "main" (paper's main-table protocol: global SMOTE, then
            D_val drawn from the balanced pool) or "leakage_free" (paper
            §VIII-A / Table VI: D_val drawn pre-SMOTE, SMOTE applied
            per-client after Dirichlet partitioning so no synthetic sample
            can leak information from a validation example into training).

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
        # protocol="leakage_free" defers SMOTE to the per-client stage below,
        # so the global pipeline applies SMOTE only for the "main" protocol.
        (X_train, y_train,
         X_val, y_val,
         X_test, y_test,
         _, _, class_weights) = nslkdd_pipeline(
            train_path, test_path, use_smote=(protocol == "main"),
            seed=seed, val_size=val_size, protocol=protocol,
        )
    elif dataset == "unswnb15":
        (X_train, y_train,
         X_val, y_val,
         X_test, y_test,
         _, _, class_weights) = unswnb15_pipeline(
            train_path, test_path, use_smote=True, seed=seed
        )
        if protocol == "leakage_free":
            print("[Warning] protocol='leakage_free' is not yet implemented for "
                  "dataset='unswnb15' (an orphaned, paper-unreferenced pipeline); "
                  "falling back to its existing single protocol.")
    elif dataset == "ciciot2023":
        from data.preprocessing.ciciot2023_pipeline import build_pipeline as ciciot2023_pipeline
        (X_train, y_train,
         X_val, y_val,
         X_test, y_test,
         _, _, class_weights) = ciciot2023_pipeline(
            train_path, test_path, use_smote=(protocol == "main"),
            seed=seed, val_size=val_size, protocol=protocol,
        )
    else:
        raise NotImplementedError(f"Dataset '{dataset}' not yet integrated. "
                                   "Use 'nslkdd', 'unswnb15', or 'ciciot2023'.")

    # Partition training data across clients
    partitioner = get_partitioner(partition_type, alpha=alpha)
    client_data = partitioner.partition(X_train, y_train, num_clients, seed=seed)

    if protocol == "leakage_free" and dataset in ("nslkdd", "ciciot2023"):
        # Leakage-free protocol (paper Section VIII-A / Table VI): D_val was
        # drawn pre-SMOTE by the pipeline above, so SMOTE is applied here,
        # per client, after Dirichlet partitioning. Both NSL-KDD and
        # CIC-IoT-2023 support this; UNSW-NB15 does not (warned above).
        if dataset == "nslkdd":
            from data.preprocessing.nslkdd_pipeline import apply_smote
        else:
            from data.preprocessing.ciciot2023_pipeline import apply_smote
        client_data = [
            apply_smote(Xc, yc, random_state=seed + cid)
            for cid, (Xc, yc) in enumerate(client_data)
        ]

    print(f"[Data] {dataset.upper()} | Clients={num_clients} | "
          f"Partition={partition_type}(alpha={alpha}) | Protocol={protocol} | "
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
    proxy_val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
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
            import warnings
            warnings.warn(
                f"[Client {client_id}] Local val set is empty after 80/20 split "
                f"(client has {len(X_c)} samples total). "
                "Falling back to 50 samples from server validation set. "
                "This client's local val loss signal may be slightly inflated.",
                RuntimeWarning,
                stacklevel=2,
            )
            X_lv, y_lv = X_val[:50], y_val[:50]

        is_malicious = client_id in malicious_ids

        # Belt-and-braces reproducibility: the client's own model init happens
        # inside a Ray worker with an unseeded RNG. Its values are overwritten
        # by set_parameters on every fit/evaluate call, so they do not affect
        # the algorithm -- but seeding them keeps any future code path that
        # reads a client's pre-broadcast weights a function of the run seed.
        torch.manual_seed(int(attack_kwargs.get("seed", 42)) * 7919 + client_id)

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
            proxy_val_data=proxy_val_data if is_malicious else None,
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
    attack_type: Optional[str] = None,
    attack_kwargs: Optional[dict] = None,
    malicious_ids: Optional[List[int]] = None,
    seed: int = 42,
    evaluate_fn=None,
    strategy_kwargs_override: Optional[dict] = None,
):
    """Instantiate the requested FL strategy.

    Args:
        strategy_kwargs_override: Optional dict of constructor kwargs that
            override the computed defaults for the selected strategy (e.g.
            {"num_byzantine": 4} for krum, {"beta": 0.1} for trimmed_mean).
            Used by experiments/run_hyperparameter_sweep.py (audit IDs
            E10/E11) to sweep baseline hyperparameters without disturbing
            the normal adv_ratio-derived defaults. None/omitted preserves
            prior behavior exactly.
    """
    override = strategy_kwargs_override or {}
    frac_fit  = fl_cfg.get("fraction_fit", 0.5)
    frac_eval = fl_cfg.get("fraction_evaluate", 0.3)

    common_kwargs = dict(
        fraction_fit=frac_fit,
        fraction_evaluate=frac_eval,
        min_fit_clients=max(2, int(num_clients * frac_fit)),
        min_evaluate_clients=max(1, int(num_clients * frac_eval)),
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
        # Reproducibility: the client's local shuffle must be a function of
        # (run seed, client id, round), not of whichever Ray worker process
        # happened to execute the client. Flower sends no round number to
        # fit() unless a strategy supplies one, so without this the same seed
        # produced different accuracies run to run
        # (tests/test_all.py::TestDeterminism). See fl/client.py::fit.
        on_fit_config_fn=lambda server_round: {
            "server_round": int(server_round), "run_seed": int(seed)},
        on_evaluate_config_fn=lambda server_round: {
            "server_round": int(server_round), "run_seed": int(seed)},
        # Reproducibility: without an explicit initial model, Flower asks one
        # RANDOMLY CHOSEN virtual client for its parameters
        # (server.py::_get_initial_parameters), and that client builds its
        # model inside a Ray worker whose RNG this code never seeded. Every
        # run therefore started from a different global model -- visible as a
        # different round-0 accuracy for the same seed. `global_model` is
        # constructed in this process after set_all_seeds(seed), so handing it
        # over makes round 0 (and everything downstream of it) a function of
        # the seed alone.
        initial_parameters=fl.common.ndarrays_to_parameters(
            global_model.get_parameters()),
    )

    adv_ratio = config.get("adversarial", {}).get("attack_ratio", 0.3)
    n_byzantine = max(1, int(num_clients * adv_ratio))

    attack_kwargs = attack_kwargs or {}

    if strategy_name == "fedavg":
        return FedAvgStrategy(
            global_model=global_model,
            attack_type=attack_type,
            attack_kwargs=attack_kwargs,
            malicious_ids=malicious_ids,
            # Supplies enable_overhead_tracking so FedAvg's own server-side
            # aggregation time is measured under the same flag as TV-FLIDS's;
            # Table XII's relative figure needs a measured denominator.
            config=config,
            **common_kwargs,
        )

    elif strategy_name == "krum":
        return KrumStrategy(
            num_clients=num_clients,
            num_byzantine=override.get("num_byzantine", n_byzantine),
            m=override.get("m", 1),
            global_model=global_model,
            attack_type=attack_type,
            attack_kwargs=attack_kwargs,
            malicious_ids=malicious_ids,
            **common_kwargs,
        )

    elif strategy_name in ("multikrum", "multi_krum"):
        return MultiKrumStrategy(
            num_clients=num_clients,
            num_byzantine=override.get("num_byzantine", n_byzantine),
            m=override.get("m", None),
            global_model=global_model,
            attack_type=attack_type,
            attack_kwargs=attack_kwargs,
            malicious_ids=malicious_ids,
            **common_kwargs,
        )

    elif strategy_name == "trimmed_mean":
        beta = override.get("beta", min(0.35, adv_ratio + 0.05))
        return TrimmedMeanStrategy(
            beta=beta,
            global_model=global_model,
            attack_type=attack_type,
            attack_kwargs=attack_kwargs,
            malicious_ids=malicious_ids,
            **common_kwargs,
        )

    elif strategy_name == "norm_clipping":
        return NormClippingStrategy(
            clip_factor=override.get("clip_factor", 1.0),
            clip_radius=override.get("clip_radius", None),
            weighted=override.get("weighted", False),
            global_model=global_model,
            attack_type=attack_type,
            attack_kwargs=attack_kwargs,
            malicious_ids=malicious_ids,
            **common_kwargs,
        )

    elif strategy_name == "fltrust":
        if root_loader is None:
            raise ValueError("FLTrust requires a root_loader. Pass --strategy fltrust.")
        return FLTrustStrategy(
            server_model=global_model,
            server_root_loader=root_loader,
            device=device,
            local_epochs=override.get("local_epochs", 1),
            lr=override.get("lr", fl_cfg.get("local_lr", 0.001)),
            attack_type=attack_type,
            attack_kwargs=attack_kwargs,
            malicious_ids=malicious_ids,
            **common_kwargs,
        )

    elif strategy_name == "foolsgold":
        return FoolsGoldStrategy(
            num_clients=num_clients,
            global_model=global_model,
            attack_type=attack_type,
            attack_kwargs=attack_kwargs,
            malicious_ids=malicious_ids,
            **common_kwargs,
        )

    elif strategy_name == "flame":
        return FLAMEStrategy(
            num_clients=num_clients,
            global_model=global_model,
            attack_type=attack_type,
            attack_kwargs=attack_kwargs,
            malicious_ids=malicious_ids,
            seed=seed,
            **common_kwargs,
        )

    elif strategy_name == "rfa":
        return RFAStrategy(
            global_model=global_model,
            attack_type=attack_type,
            attack_kwargs=attack_kwargs,
            malicious_ids=malicious_ids,
            **common_kwargs,
        )

    elif strategy_name == "bucketing":
        return BucketingStrategy(
            bucket_size=override.get("bucket_size", 2),
            beta=override.get("beta", min(0.35, adv_ratio + 0.05)),
            global_model=global_model,
            attack_type=attack_type,
            attack_kwargs=attack_kwargs,
            malicious_ids=malicious_ids,
            seed=seed,
            **common_kwargs,
        )

    elif strategy_name == "deepsight":
        return DeepSightStrategy(
            global_model=global_model,
            attack_type=attack_type,
            attack_kwargs=attack_kwargs,
            malicious_ids=malicious_ids,
            seed=seed,
            **common_kwargs,
        )

    elif strategy_name == "fldetector":
        return FLDetectorStrategy(
            window=override.get("window", 10),
            start_round=override.get("start_round", 10),
            beta=override.get("beta", 0.1),
            num_clients=num_clients,
            global_model=global_model,
            attack_type=attack_type,
            attack_kwargs=attack_kwargs,
            malicious_ids=malicious_ids,
            **common_kwargs,
        )

    elif strategy_name == "zeno":
        return ZenoStrategy(
            server_model=global_model,
            val_loader=val_loader,
            device=device,
            rho=override.get("rho", 0.001),
            b=override.get("b", 3),
            eta=override.get("eta", 1.0),
            global_model=global_model,
            attack_type=attack_type,
            attack_kwargs=attack_kwargs,
            malicious_ids=malicious_ids,
            **common_kwargs,
        )

    elif strategy_name == "baffle":
        return BaFFLeStrategy(
            num_clients=num_clients,
            num_validators=override.get("num_validators", 5),
            quorum=override.get("quorum", 0.3),
            lookback=override.get("lookback", 10),
            error_threshold=override.get("error_threshold", 0.05),
            val_loader=val_loader,
            device=device,
            global_model=global_model,
            attack_type=attack_type,
            attack_kwargs=attack_kwargs,
            malicious_ids=malicious_ids,
            seed=seed,
            **common_kwargs,
        )

    elif strategy_name in ("tvflids", "tvflids_adaptive", "tvflids_fixed"):
        adaptive = (strategy_name != "tvflids_fixed")
        return TVFLIDSStrategy(
            num_clients=num_clients,
            config=config,
            val_loader=val_loader,
            model=global_model,
            device=device,
            adaptive=adaptive,
            # None => read `verification.adaptive_thresholds` from the config
            # (default true, matching paper Section IV-A / Table IV footnote).
            # This was previously hardcoded False, which silently disabled the
            # tau_L / tau_z warmup annealing in every reported experiment.
            use_adaptive_thresholds=None,
            known_malicious=malicious_ids,
            attack_type=attack_type,
            attack_kwargs=attack_kwargs,
            seed=seed,
            **common_kwargs,
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}. "
                         "Choose: fedavg, krum, multikrum, trimmed_mean, "
                         "norm_clipping, fltrust, foolsgold, flame, rfa, "
                         "bucketing, deepsight, fldetector, zeno, baffle, "
                         "tvflids, tvflids_fixed")


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

def _load_completed_run(log_dir: str, n_rounds: int):
    """Return a completed run's own stored summary, or None.

    Guards, all of which must pass:
      * experiment_log.json exists and parses;
      * it logged exactly n_rounds rounds (a truncated run is not complete);
      * it carries a non-empty summary with a final_accuracy.

    Returns the summary dict exactly as run_experiment originally returned it.
    Nothing is synthesised: if any guard fails the caller re-runs the cell.
    """
    path = os.path.join(log_dir, "experiment_log.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    rounds = blob.get("rounds") or []
    summary = blob.get("summary") or {}
    # An n-round run logs n+1 evaluation points: round 0 is the pre-training
    # centralized evaluation of the initial global model, rounds 1..n are the
    # FL rounds. Anything short of that is a truncated run, not a complete one.
    if len(rounds) != n_rounds + 1:
        return None
    if max((r.get("round", -1) for r in rounds), default=-1) != n_rounds:
        return None
    if "final_accuracy" not in summary:
        return None
    return summary


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
    model_type: str = "mlp",
    val_size: int = 2000,
    protocol: str = "main",
    strategy_kwargs_override: Optional[dict] = None,
    attack_variant: Optional[str] = None,
    knowledge_tier: Optional[str] = None,
    on_off_k: Optional[int] = None,
) -> Dict:
    """
    Run a complete FL experiment end-to-end.

    Args:
        strategy_name:      FL strategy to use.
        attack_config_name: Key from ATTACK_CONFIGS dict.
        dataset:            Dataset name ('nslkdd', 'unswnb15', or 'ciciot2023').
        partition_type:     'iid' or 'noniid'.
        alpha:              Dirichlet α for non-IID partitioning.
        seed:               Random seed.
        num_rounds:         Override config rounds.
        config_path:        Path to FL config YAML.
        log_dir:            Override default log directory.
        verbose:            Print progress.
        val_size:           Server validation set D_val size (paper: 2,000).
        protocol:           'main' or 'leakage_free' data-preparation protocol
                             (see setup_data() / nslkdd_pipeline.build_pipeline()).
        strategy_kwargs_override: Optional dict of constructor-kwarg overrides
            passed through to make_strategy() (e.g. {"num_byzantine": 4} for
            krum, {"beta": 0.1} for trimmed_mean). Used by
            experiments/run_hyperparameter_sweep.py (audit IDs E10/E11) to
            sweep baseline hyperparameters. None preserves prior behavior.
        attack_variant:     Override attack variant (e.g. 'partial', 'full').
        knowledge_tier:     Override attack knowledge tier (e.g. 'K1', 'K2').
        on_off_k:           Override on-off attack round frequency / parameter k.

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
        "scale_factor":    atk_cfg.get("factor", 10.0),
        "noise_std":       atk_cfg.get("std", 0.5),
        "poison_ratio":    atk_cfg.get("poison_ratio", 0.1),
        "flip_ratio":      atk_cfg.get("flip_ratio", 1.0),
        "target_class":    atk_cfg.get("target_class", 0),
        "gamma":           atk_cfg.get("gamma", 2.0),
        "variant":         attack_variant or atk_cfg.get("variant", "partial"),
        "knowledge_tier":  knowledge_tier or atk_cfg.get("knowledge_tier", "K1"),
        "k":               on_off_k or atk_cfg.get("k", 30),
        "psi":             atk_cfg.get("psi", 0.85),
        "rho_a":           atk_cfg.get("rho_a", 1.0),
        "m":               atk_cfg.get("m", 0.0),
        "proxy_val_ratio": atk_cfg.get("proxy_val_ratio", 0.15),
        "aux_loss_weight": atk_cfg.get("aux_loss_weight", 0.5),
        "poison_strength": atk_cfg.get("poison_strength", 1.0),
        "shift_scale":     atk_cfg.get("shift_scale", 1.0),
    }
    attack_kwargs["seed"] = seed

    malicious_ids = get_malicious_client_ids(
        num_clients, atk_cfg["ratio"], seed=seed
    )
    if verbose:
        print(f"\n{'='*60}")
        print(f" Strategy:  {strategy_name.upper()}")
        print(f" Attack:    {attack_config_name} ({len(malicious_ids)} malicious)")
        print(f" Dataset:   {dataset.upper()} | {partition_type}(alpha={alpha})")
        print(f" Seed:      {seed}  |  Rounds: {n_rounds}")
        print(f"{'='*60}\n")

    # ── Logging ───────────────────────────────────────────────────────
    if log_dir is None:
        log_dir = (f"results/logs/{strategy_name}_{attack_config_name}_"
                   f"{partition_type}_seed{seed}")

    # ── Idempotent resume (campaign infrastructure, off by default) ────
    # A full campaign is hundreds of independent (strategy, attack, seed)
    # cells run over many hours; a crash or a reboot part-way through must
    # not force the completed cells to be recomputed. With TVFLIDS_RESUME=1,
    # a cell whose own experiment_log.json is already on disk AND complete
    # (all n_rounds logged, a summary present) returns that run's own stored
    # summary verbatim.
    #
    # This can only ever replay a genuine prior execution of this same cell:
    # it reads the artifact that run wrote, computes nothing, and refuses
    # anything short of a complete log. It is deliberately opt-in so that a
    # default invocation always executes.
    if os.getenv("TVFLIDS_RESUME", "0") == "1":
        cached = _load_completed_run(log_dir, n_rounds)
        if cached is not None:
            if verbose:
                print(f"[Resume] {log_dir}: complete ({n_rounds} rounds) - "
                      f"reusing stored summary, not re-running.")
            return cached

    logger = ExperimentLogger(log_dir, experiment_name=strategy_name)
    logger.log_config({
        "strategy": strategy_name, "attack": attack_config_name,
        "dataset": dataset, "partition_type": partition_type,
        "alpha": alpha, "seed": seed, "val_size": val_size,
        "protocol": protocol, **fl_cfg,
    })

    # ── Data preparation ──────────────────────────────────────────────
    client_data, X_test, y_test, X_val, y_val, class_weights = setup_data(
        config, dataset=dataset, seed=seed,
        partition_type=partition_type, alpha=alpha,
        val_size=val_size, protocol=protocol,
    )

    input_dim  = X_test.shape[1]
    num_classes = len(np.unique(y_test))
    model_kwargs = {"input_dim": input_dim, "num_classes": num_classes}

    # ── Global model ──────────────────────────────────────────────────
    from models.mlp import build_model
    global_model = build_model(
        model_type=model_type,
        input_dim=input_dim,
        num_classes=num_classes,
    ).to(device)

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
    if dataset == "nslkdd":
        class_names = NSLKDD_CLASS_NAMES
    elif dataset == "unswnb15":
        class_names = UNSW_CLASS_NAMES
    else:
        class_names = None

    metrics_tracker = ExperimentMetrics(
        class_names=class_names[:num_classes] if class_names else None
    )
    # NOTE: per-stage compute timing lives on the strategy itself
    # (TVFLIDSStrategy.overhead_tracker, instrumented inside aggregate_fit) and
    # is read back into `summary["compute_overhead_ms"]` at the end of this
    # function. A second, never-populated tracker used to be constructed here,
    # which is why Table XII had no instrumented source in the code path.

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
        device, num_clients, fl_cfg,
        attack_type=attack_type,
        attack_kwargs=attack_kwargs,
        malicious_ids=malicious_ids,
        seed=seed,
        evaluate_fn=evaluate_fn,
        strategy_kwargs_override=strategy_kwargs_override,
    )
    strategy_container.strategy = strategy

    # ── Proxy validation data for adaptive attacks ────────────────────
    proxy_val_data = None
    if attack_type in ("ack1", "ack1_evasion", "ack2", "ack3", "ack3_evasion", "ack4"):
        from attacks.knowledge import ValidationEstimateProvider
        ktier = attack_kwargs.get("knowledge_tier", "K1")
        try:
            if ktier == "K0":
                coalition_data = [client_data[cid] for cid in malicious_ids]
                proxy_val_data = ValidationEstimateProvider.get_validation_estimate(
                    tier="K0", coalition_data=coalition_data, seed=seed
                )
            elif ktier == "K1":
                X_bg = np.concatenate([X for X, _ in client_data], axis=0)
                y_bg = np.concatenate([y for _, y in client_data], axis=0)
                proxy_val_data = ValidationEstimateProvider.get_validation_estimate(
                    tier="K1", background_data=(X_bg, y_bg), seed=seed
                )
            elif ktier == "K2":
                proxy_val_data = ValidationEstimateProvider.get_validation_estimate(
                    tier="K2", server_val_data=(X_val, y_val), seed=seed
                )
        except Exception as e:
            print(f"[Warning] Failed to construct knowledge-tier {ktier} validation estimate: {e}")

    # ── Client factory ────────────────────────────────────────────────
    client_fn = make_client_fn(
        client_data, X_val, y_val, config, device,
        class_weights, malicious_ids, attack_type, attack_kwargs, model_kwargs,
        proxy_val_data=proxy_val_data,
    )

    # ── Flower simulation ─────────────────────────────────────────────
    sim_num_cpus = int(os.getenv("TVFLIDS_SIM_CLIENT_CPUS", "1"))
    sim_num_gpus = float(os.getenv("TVFLIDS_SIM_CLIENT_GPUS", "0.0"))

    ray_init_args = None
    local_mode_env = os.getenv("TVFLIDS_SIM_LOCAL_MODE", "")
    if local_mode_env == "1" or (local_mode_env != "0" and sys.platform == "win32"):
        ray_init_args = {
            "local_mode": True,
            "include_dashboard": False,
            "ignore_reinit_error": True,
        }

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
        client_resources={"num_cpus": sim_num_cpus, "num_gpus": sim_num_gpus},
        ray_init_args=ray_init_args,
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
            # Figure 5 (results/figures/fig5_adaptive_weights.pdf). Like
            # Figure 4, this was listed as required by scripts/check_results.py
            # while figure5_adaptive_weights had no call site anywhere, so the
            # file could never be produced. Only the adaptive strategy has a
            # weight history to plot; tvflids_fixed holds the weights constant
            # by definition.
            weight_history = getattr(strategy.trust_scorer, "weight_history", None)
            if weight_history:
                from evaluation.visualization import figure5_adaptive_weights
                figure5_adaptive_weights(
                    {attack_config_name: weight_history},
                    save_path="results/figures/fig5_adaptive_weights.pdf",
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
                        class_names=class_names[:num_classes] if class_names else None,
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

    # Per-stage computational overhead (paper Table XII). Only TVFLIDSStrategy
    # instruments its aggregate_fit stages; baselines expose no such breakdown,
    # so this key is absent for them rather than being filled with zeros.
    _ohead_summary = {}
    if hasattr(strategy, "get_overhead_summary"):
        try:
            _ohead_summary = strategy.get_overhead_summary()
        except Exception as e:
            print(f"[Warning] overhead summary unavailable: {e}")
    if _ohead_summary:
        summary["compute_overhead_ms"] = _ohead_summary
        try:
            OverheadTracker.print_report(strategy.overhead_tracker)
        except Exception:
            pass

    # ── Figure-backing artifacts (manuscript Figures 3 and 4) ─────────
    # These live only on the strategy object during the run. Persisting them
    # is what gives scripts/generate_manuscript_figures.py a disk artifact to
    # read, closing the gap where Figures 3 and 4 had no regeneration path.
    logger.log_extra("malicious_ids", list(malicious_ids))
    if hasattr(strategy, "get_trust_history"):
        try:
            logger.log_extra(
                "trust_history",
                {str(cid): [float(v) for v in hist]
                 for cid, hist in strategy.get_trust_history().items()},
            )
        except Exception as e:
            print(f"[Warning] trust history not persisted: {e}")
    if getattr(strategy, "round_logs", None):
        # Carries adaptive_alpha / adaptive_beta / adaptive_gamma per round
        # (Figure 4) alongside the realized tau_z / tau_L and gate counts.
        logger.log_extra("strategy_round_logs", strategy.round_logs)

    logger.log_summary(summary)
    logger.save()

    if config.get("log_client_params", False) and getattr(strategy, "_last_round_data", None):
        try:
            from theory.proposition1_verification import verify_from_experiment_log
            prop_result = verify_from_experiment_log(
                os.path.join(log_dir, "experiment_log.json"),
                strategy,
            )
            os.makedirs("results/tables", exist_ok=True)
            out_path = "results/tables/proposition1_real.json"
            with open(out_path, "w") as f:
                json.dump(prop_result, f, indent=2, default=str)
            if verbose:
                print(f"[Proposition 1] Saved verification to {out_path}")
        except Exception as e:
            print(f"[Warning] Proposition 1 verification failed: {e}")

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
                        choices=["fedavg", "krum", "multikrum", "multi_krum", "trimmed_mean",
                                 "norm_clipping", "fltrust", "foolsgold", "flame", "rfa",
                                 "bucketing", "deepsight", "fldetector", "zeno", "baffle",
                                 "tvflids", "tvflids_adaptive", "tvflids_fixed"],
                        help="Aggregation strategy")
    parser.add_argument("--attack",     type=str, default="label_flip_30",
                        choices=list(ATTACK_CONFIGS.keys()),
                        help="Attack configuration")
    parser.add_argument("--dataset",    type=str, default="nslkdd",
                        choices=["nslkdd", "unswnb15", "ciciot2023"],
                        help="Dataset name")
    parser.add_argument("--partition",  type=str, default="noniid",
                        choices=["iid", "noniid"],
                        help="Data partitioning strategy")
    parser.add_argument("--alpha",      type=float, default=0.5,
                        help="Dirichlet alpha for non-IID (0.5=moderate, 0.1=extreme)")
    parser.add_argument("--val-size",   type=int, default=2000,
                        help="Server validation set (D_val) size (paper: 2000)")
    parser.add_argument("--protocol",   type=str, default="main",
                        choices=["main", "leakage_free"],
                        help="Data-preparation protocol: 'main' (global SMOTE then "
                             "draw D_val from the balanced pool, paper's main-table "
                             "protocol) or 'leakage_free' (D_val drawn pre-SMOTE, "
                             "SMOTE applied per-client post-partition, paper Table VI)")
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
    parser.add_argument("--model",      type=str, default="mlp",
                        choices=["mlp", "bilstm"],
                        help="Model architecture (default: mlp)")
    parser.add_argument("--knowledge-tier", type=str, default=None,
                        choices=["K0", "K1", "K2"],
                        help="Adversary knowledge tier for adaptive attacks")
    parser.add_argument("--attack-variant", type=str, default=None,
                        choices=["partial", "omniscient"],
                        help="Attack variant for Min-Max / Min-Sum")
    parser.add_argument("--on-off-k",   type=int, default=None,
                        choices=[10, 20, 30, 50],
                        help="Honest phase length for on-off attacks")

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
        model_type=args.model,
        val_size=args.val_size,
        protocol=args.protocol,
        attack_variant=args.attack_variant,
        knowledge_tier=args.knowledge_tier,
        on_off_k=args.on_off_k,
    )

    print("\n[Done] Final results:")
    print(json.dumps({k: v for k, v in result.items()
                       if not isinstance(v, (list, dict))}, indent=2))


if __name__ == "__main__":
    main()
