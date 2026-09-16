"""
Unit test: ensure `TVFLIDSStrategy._eval_model` caches results.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fl.strategy import TVFLIDSStrategy


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)

    # Minimal get/set parameters expected by strategy
    def get_parameters(self):
        return [p.detach().cpu().numpy() for p in self.parameters()]

    def set_parameters(self, params):
        for p, a in zip(self.parameters(), params):
            p.data = torch.from_numpy(a).to(p.data.dtype)


def test_eval_cache_behavior():
    torch.manual_seed(0)
    # tiny validation set
    X = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(X, y), batch_size=4)

    model = DummyModel()
    device = torch.device('cpu')
    strat = TVFLIDSStrategy(num_clients=1, config={}, val_loader=loader,
                           model=model, device=device, evaluate_fn=None)

    # prepare a parameter vector (use current model params)
    params = model.get_parameters()

    # Ensure cache empty
    strat._eval_cache.clear()

    # First eval — should populate cache
    loss1 = strat._eval_model(params)
    assert len(strat._eval_cache) == 1

    # Second eval with same params — should hit cache and return same value
    loss2 = strat._eval_model(params)
    assert loss1 == loss2
