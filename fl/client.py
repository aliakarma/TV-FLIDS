"""
fl/client.py — Flower FL client with optional adversarial attack injection.
Reference: Guide §6.2
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple
import flwr as fl
from flwr.common import NDArrays, Scalar
from attacks.adversarial import AdversarialAttackFactory


class TVFLIDSClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray, device: torch.device,
                 config: dict, class_weights: Optional[np.ndarray] = None,
                 model_class=None, model_kwargs: Optional[dict] = None,
                 is_malicious: bool = False, attack_type: Optional[str] = None,
                 attack_kwargs: Optional[dict] = None):
        self.client_id = client_id
        self.device = device
        self.config = config
        self.is_malicious = is_malicious
        self.attack_type = attack_type
        self.attack_kwargs = attack_kwargs or {}
        self.factory = AdversarialAttackFactory()

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

        X, y = self.X_train_np.copy(), self.y_train_np.copy()

        # Data-level attacks BEFORE training
        if self.is_malicious:
            if self.attack_type == 'label_flip':
                y = self.factory.label_flip(y, target_class=self.attack_kwargs.get('target_class', 0),
                                             flip_ratio=self.attack_kwargs.get('flip_ratio', 1.0))
            elif self.attack_type == 'backdoor':
                X, y = self.factory.backdoor_attack(
                    X, y,
                    trigger_feature_idx=self.attack_kwargs.get('trigger_feature_idx', 0),
                    trigger_value=self.attack_kwargs.get('trigger_value', 1.0),
                    target_class=self.attack_kwargs.get('target_class', 0),
                    poison_ratio=self.attack_kwargs.get('poison_ratio', 0.1))

        # Local training
        bs = self.config.get('local_batch_size', 256)
        loader = DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32),
                          torch.tensor(y, dtype=torch.long)),
            batch_size=bs, shuffle=True,
            drop_last=(len(X) > bs))

        self.model.train()
        for _ in range(self.config.get('local_epochs', 5)):
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                self.criterion(self.model(Xb), yb).backward()
                self.optimizer.step()

        # Model-level attacks AFTER training
        new_params = self.model.get_parameters()
        if self.is_malicious:
            if self.attack_type == 'gradient_scale':
                new_params = self.factory.gradient_scale(
                    new_params, global_params,
                    scale_factor=self.attack_kwargs.get('scale_factor', 10.0))
            elif self.attack_type == 'noise':
                new_params = self.factory.noise_injection(
                    new_params, noise_std=self.attack_kwargs.get('noise_std', 0.5))

        val_loss = self._val_loss(new_params)
        return new_params, len(X), {'val_loss': float(val_loss)}

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
