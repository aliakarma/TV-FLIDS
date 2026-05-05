"""
models/mlp.py — IDS classification models for TV-FLIDS.
Primary: IDSMLP  (tabular data — NSL-KDD, UNSW-NB15)
Secondary: IDSBiLSTM (sequential — UNSW-NB15 extension)
Reference: Guide §5.2
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List


class IDSMLP(nn.Module):
    """
    MLP for network intrusion detection.
    Input → 256 → 128 → 64 → num_classes (with BN + Dropout).
    """
    def __init__(self, input_dim: int = 41, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_parameters(self) -> List[np.ndarray]:
        """Return model parameters as list of numpy arrays (Flower interface)."""
        return [p.cpu().detach().numpy().copy() for p in self.parameters()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Load parameters from list of numpy arrays (Flower interface)."""
        with torch.no_grad():
            for p, new_val in zip(self.parameters(), parameters):
                p.copy_(torch.tensor(new_val, dtype=p.dtype))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class IDSBiLSTM(nn.Module):
    """
    Bidirectional LSTM for sequential traffic analysis (UNSW-NB15 extension).
    Input shape: (batch, seq_len, features)
    """
    def __init__(self, input_dim: int = 49, hidden_dim: int = 128,
                 num_layers: int = 2, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])

    def get_parameters(self) -> List[np.ndarray]:
        return [p.cpu().detach().numpy().copy() for p in self.parameters()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        with torch.no_grad():
            for p, v in zip(self.parameters(), parameters):
                p.copy_(torch.tensor(v, dtype=p.dtype))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(model_type: str = 'mlp', input_dim: int = 41,
                num_classes: int = 5, dropout: float = 0.3) -> nn.Module:
    if model_type == 'mlp':
        return IDSMLP(input_dim=input_dim, num_classes=num_classes, dropout=dropout)
    elif model_type == 'bilstm':
        return IDSBiLSTM(input_dim=input_dim, num_classes=num_classes, dropout=dropout)
    raise ValueError(f"Unknown model type: {model_type}. Use 'mlp' or 'bilstm'.")
