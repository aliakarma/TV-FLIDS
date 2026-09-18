"""
fl/client.py — Flower FL client with adversarial attack injection.
Reference: IEEE TIFS Manuscript §III, §VII, and Supplementary §S4.
"""

from time import perf_counter as _perf_counter
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
from flwr.common import NDArrays, Scalar

from attacks.adversarial import AdversarialAttackFactory
from attacks.knowledge import KnowledgeTier, ValidationEstimateProvider, NSLKDD_VAL_QUOTAS
from trust.verification import compute_class_balanced_loss_from_tensors


class TVFLIDSClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        device: torch.device,
        config: dict,
        class_weights: Optional[np.ndarray] = None,
        model_class=None,
        model_kwargs: Optional[dict] = None,
        is_malicious: bool = False,
        attack_type: Optional[str] = None,
        attack_kwargs: Optional[dict] = None,
        proxy_val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        self.client_id = client_id
        self.device = device
        self.config = config
        self.is_malicious = is_malicious
        self.attack_type = attack_type
        self.attack_kwargs = attack_kwargs or {}
        self.seed = self.attack_kwargs.get("seed", 42)
        self.factory = AdversarialAttackFactory()
        self.proxy_val_data = proxy_val_data

        self.X_train_np = X_train.copy()
        self.y_train_np = y_train.copy()
        self.X_val = torch.tensor(X_val, dtype=torch.float32)
        self.y_val = torch.tensor(y_val, dtype=torch.long)

        if model_class is None:
            from models.mlp import IDSMLP
            model_class = IDSMLP
        self.model = model_class(**(model_kwargs or {})).to(device)

        w = torch.tensor(class_weights, dtype=torch.float32).to(device) \
            if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=w)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.get('local_lr', 0.001))

    def get_parameters(self, config: dict) -> NDArrays:
        return self.model.get_parameters()

    def set_parameters(self, parameters: NDArrays) -> None:
        self.model.set_parameters(parameters)

    def fit(self, parameters: NDArrays, config: dict) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        global_params = [p.copy() for p in parameters]

        server_round = int(config.get("server_round", 0))
        run_seed = int(config.get("run_seed", self.seed))
        local_seed = (run_seed * 1_000_003
                      + self.client_id * 10_007
                      + server_round) % (2 ** 31 - 1)
        loader_generator = torch.Generator()
        loader_generator.manual_seed(local_seed)
        torch.manual_seed(local_seed)
        np.random.seed(local_seed % (2 ** 32))

        X, y = self.X_train_np.copy(), self.y_train_np.copy()

        aux_X_t: Optional[torch.Tensor] = None
        aux_y_t: Optional[torch.Tensor] = None
        is_ack3 = False
        rho_a = float(self.attack_kwargs.get("rho_a", self.attack_kwargs.get("aux_loss_weight", 1.0)))
        margin_m = float(self.attack_kwargs.get("m", 0.0))
        base_val_loss = 0.0
        base_bal_loss = 0.0

        # Data-level attacks BEFORE training
        if self.is_malicious and self.attack_type:
            effective_type = self.attack_type

            # Check on-off scheduling for client-side attacks
            if effective_type in ("on_off_lf", "on_off_label_flip"):
                k = self.attack_kwargs.get("k", 30)
                if self.factory.is_on_off_active(server_round, k=k):
                    effective_type = "label_flip"
                else:
                    effective_type = "honest"

            if effective_type in ('label_flip', 'lf'):
                y = self.factory.label_flip(
                    y,
                    target_class=self.attack_kwargs.get('target_class', 0),
                    flip_ratio=self.attack_kwargs.get('flip_ratio', 1.0),
                    seed=self.seed + self.client_id,
                )
            elif effective_type in ('lf_r', 'label_flip_random'):
                y = self.factory.label_flip_random(
                    y,
                    num_classes=self.attack_kwargs.get('num_classes', 5),
                    flip_ratio=self.attack_kwargs.get('flip_ratio', 1.0),
                    seed=self.seed + self.client_id,
                )
            elif effective_type == 'backdoor':
                X, y = self.factory.backdoor_attack(
                    X, y,
                    trigger_feature_idx=self.attack_kwargs.get('trigger_feature_idx', 0),
                    trigger_value=self.attack_kwargs.get('trigger_value', 1.0),
                    target_class=self.attack_kwargs.get('target_class', 0),
                    poison_ratio=self.attack_kwargs.get('poison_ratio', 0.1),
                    seed=self.seed + self.client_id,
                )
            elif effective_type in ('ack1', 'ack1_evasion'):
                # ACK1: Relabel all attack samples to benign (0)
                y = self.factory.label_flip(
                    y,
                    target_class=self.attack_kwargs.get('target_class', 0),
                    flip_ratio=self.attack_kwargs.get('flip_ratio', 1.0),
                    seed=self.seed + self.client_id,
                )
                if self.proxy_val_data is not None:
                    pX, py = self.proxy_val_data
                else:
                    _, _, pX, py = self.factory.ack1_prepare_proxy_val(
                        self.X_train_np, self.y_train_np,
                        proxy_ratio=self.attack_kwargs.get('proxy_val_ratio', 0.15),
                        seed=self.seed + self.client_id,
                    )
                if len(pX) > 0:
                    aux_X_t = torch.tensor(pX, dtype=torch.float32).to(self.device)
                    aux_y_t = torch.tensor(py, dtype=torch.long).to(self.device)
                    with torch.no_grad():
                        self.model.eval()
                        base_val_loss = self.criterion(self.model(aux_X_t), aux_y_t).item()
                        self.model.train()

            elif effective_type in ('ack3', 'ack3_evasion'):
                # ACK3: Relabel only majority attack classes (DoS=1, Probe=2) to benign (0)
                is_ack3 = True
                dos_probe_idx = np.where((y == 1) | (y == 2))[0]
                if len(dos_probe_idx) > 0:
                    y[dos_probe_idx] = 0

                if self.proxy_val_data is not None:
                    pX, py = self.proxy_val_data
                else:
                    _, _, pX, py = self.factory.ack1_prepare_proxy_val(
                        self.X_train_np, self.y_train_np,
                        proxy_ratio=self.attack_kwargs.get('proxy_val_ratio', 0.15),
                        seed=self.seed + self.client_id,
                    )
                if len(pX) > 0:
                    aux_X_t = torch.tensor(pX, dtype=torch.float32).to(self.device)
                    aux_y_t = torch.tensor(py, dtype=torch.long).to(self.device)
                    with torch.no_grad():
                        self.model.eval()
                        base_val_loss = self.criterion(self.model(aux_X_t), aux_y_t).item()
                        _, base_bal_loss, _, _ = compute_class_balanced_loss_from_tensors(
                            self.model(aux_X_t), aux_y_t
                        )
                        self.model.train()

        # Local training
        bs = self.config.get('local_batch_size', 256)
        loader = DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32),
                          torch.tensor(y, dtype=torch.long)),
            batch_size=bs, shuffle=True,
            generator=loader_generator,
            drop_last=(len(X) > bs))

        self.model.train()
        _train_t0 = _perf_counter()
        for _ in range(self.config.get('local_epochs', 5)):
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(Xb), yb)

                if aux_X_t is not None:
                    # Auxiliary validation evaluation on proxy slice
                    logits_val = self.model(aux_X_t)
                    cur_val_loss = self.criterion(logits_val, aux_y_t)
                    # Hinge 1: rho_a * max(0, l_val(w) - l_val(w_global) + m)
                    h1 = rho_a * torch.relu(cur_val_loss - base_val_loss + margin_m)
                    loss = loss + h1

                    if is_ack3:
                        # Hinge 2: rho_a * max(0, l_bal(w) - l_bal(w_global))
                        _, cur_bal_loss, _, _ = compute_class_balanced_loss_from_tensors(
                            logits_val, aux_y_t
                        )
                        h2 = rho_a * torch.relu(torch.tensor(cur_bal_loss - base_bal_loss, device=self.device))
                        loss = loss + h2

                loss.backward()
                self.optimizer.step()

        train_time_ms = (_perf_counter() - _train_t0) * 1000.0

        # Model-level attacks AFTER training
        new_params = self.model.get_parameters()
        if self.is_malicious and self.attack_type:
            if self.attack_type == 'gradient_scale':
                new_params = self.factory.gradient_scale(
                    new_params, global_params,
                    scale_factor=self.attack_kwargs.get('scale_factor', 10.0))
            elif self.attack_type == 'noise':
                new_params = self.factory.noise_injection(
                    new_params, noise_std=self.attack_kwargs.get('noise_std', 0.5))

        val_loss = self._val_loss(new_params)
        return new_params, len(X), {
            'val_loss': float(val_loss),
            'train_time_ms': float(train_time_ms),
        }

    def evaluate(self, parameters: NDArrays, config: dict) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        self.model.eval()
        with torch.no_grad():
            Xv, yv = self.X_val.to(self.device), self.y_val.to(self.device)
            logits = self.model(Xv)
            loss = self.criterion(logits, yv)
            acc = (torch.argmax(logits, 1) == yv).float().mean().item()
        return float(loss), len(self.X_val), {'accuracy': float(acc)}

    def _val_loss(self, params: NDArrays) -> float:
        orig = self.model.get_parameters()
        self.model.set_parameters(params)
        self.model.eval()
        with torch.no_grad():
            Xv, yv = self.X_val.to(self.device), self.y_val.to(self.device)
            loss = self.criterion(self.model(Xv), yv)
        self.model.set_parameters(orig)
        self.model.train()
        return float(loss)
