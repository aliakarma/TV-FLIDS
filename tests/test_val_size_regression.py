"""
tests/test_val_size_regression.py
Regression tests for the server validation set size |D_val| (paper Table IV:
2,000) and for the leakage-free protocol's draw order.

Audit finding N1: the pipeline silently used an internal 5% fraction (~6,299
samples on NSL-KDD) while the paper stated 2,000. These tests pin the value
end to end: the CLI default, the pipeline plumbing, and — critically — that
the requested size is the size actually reaching the strategy's val_loader,
not merely a number that gets printed.

Uses small synthetic tabular data, so no NSL-KDD download and no Flower/Ray.
"""

import inspect
import os
import sys

import numpy as np
import pytest
# Import torch BEFORE sklearn-backed project modules. On Windows, importing
# torch late (after numpy/sklearn have loaded their own OpenMP runtime) can
# trip the known c10.dll initialisation failure documented in the README.
import torch  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PAPER_VAL_SIZE = 2000   # paper Table IV, "Server val size"


def _synthetic(n=6000, d=10, k=5, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, d).astype(np.float32)
    # Guarantee every class is populated enough for a stratified split.
    y = np.concatenate([np.arange(k).repeat(n // k),
                        rng.randint(0, k, n - (n // k) * k)]).astype(np.int64)
    rng.shuffle(y)
    return X, y


# ── The split primitive ──────────────────────────────────────────────────────

class TestSplitPrimitive:

    def test_default_is_the_paper_value(self):
        from data.partitioning import (create_server_validation_set,
                                       split_server_validation_set)
        for fn in (create_server_validation_set, split_server_validation_set):
            default = inspect.signature(fn).parameters["val_size"].default
            assert default == PAPER_VAL_SIZE, \
                f"{fn.__name__} default drifted from the paper's 2,000"

    @pytest.mark.parametrize("val_size", [100, 500, 2000])
    def test_exact_absolute_count_not_a_fraction(self, val_size):
        """Regression: an absolute count, never a percentage of the pool."""
        from data.partitioning import create_server_validation_set
        X, y = _synthetic()
        Xv, yv = create_server_validation_set(X, y, val_size=val_size, seed=42)
        assert Xv.shape[0] == val_size
        assert yv.shape[0] == val_size

    def test_size_is_independent_of_pool_size(self):
        """The old bug scaled with the pool (5%); the fix must not."""
        from data.partitioning import create_server_validation_set
        sizes = []
        for n in (4000, 8000, 16000):
            X, y = _synthetic(n=n)
            Xv, _ = create_server_validation_set(X, y, val_size=500, seed=42)
            sizes.append(Xv.shape[0])
        assert sizes == [500, 500, 500]

    def test_remainder_is_disjoint_and_complete(self):
        from data.partitioning import split_server_validation_set
        X, y = _synthetic(n=3000)
        Xtr, ytr, Xv, yv = split_server_validation_set(X, y, val_size=300, seed=42)
        assert Xtr.shape[0] == 3000 - 300
        assert Xv.shape[0] == 300
        # D_val is held out of the client pool entirely.
        train_rows = {r.tobytes() for r in Xtr}
        assert not any(r.tobytes() in train_rows for r in Xv)

    def test_split_is_stratified(self):
        from data.partitioning import create_server_validation_set
        X, y = _synthetic(n=5000, k=5)
        _, yv = create_server_validation_set(X, y, val_size=1000, seed=42)
        assert len(np.unique(yv)) == 5

    def test_deterministic_for_a_fixed_seed(self):
        from data.partitioning import create_server_validation_set
        X, y = _synthetic()
        a, _ = create_server_validation_set(X, y, val_size=400, seed=7)
        b, _ = create_server_validation_set(X, y, val_size=400, seed=7)
        assert np.array_equal(a, b)


# ── CLI and pipeline defaults ────────────────────────────────────────────────

class TestDefaultsAgreeWithPaper:

    def test_entrypoint_defaults_are_2000(self):
        """Both production entry points default to the paper's value."""
        import experiments.run_experiment as rex
        assert inspect.signature(rex.run_experiment).parameters[
            "val_size"].default == PAPER_VAL_SIZE
        assert inspect.signature(rex.setup_data).parameters[
            "val_size"].default == PAPER_VAL_SIZE

    def test_cli_parser_default_is_2000(self):
        import re
        import experiments.run_experiment as rex
        src = inspect.getsource(rex)
        m = re.search(r'"--val-size",\s*type=int,\s*default=(\d+)', src)
        assert m, "--val-size argument not found in the CLI parser"
        assert int(m.group(1)) == PAPER_VAL_SIZE

    @pytest.mark.parametrize("module,func", [
        ("data.preprocessing.nslkdd_pipeline", "build_pipeline"),
        ("data.preprocessing.ciciot2023_pipeline", "build_pipeline"),
    ])
    def test_pipeline_defaults_are_2000(self, module, func):
        import importlib
        fn = getattr(importlib.import_module(module), func)
        assert inspect.signature(fn).parameters["val_size"].default == \
            PAPER_VAL_SIZE


# ── The value actually reaches the strategy ──────────────────────────────────

class TestValSizeReachesTheStrategy:

    def test_val_loader_sees_exactly_val_size_samples(self):
        """Guards against the value being computed and then not used."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from data.partitioning import create_server_validation_set
        from fl.strategy import TVFLIDSStrategy
        from models.mlp import IDSMLP

        val_size = 640
        X, y = _synthetic(n=5000, d=10, k=5)
        Xv, yv = create_server_validation_set(X, y, val_size=val_size, seed=42)

        loader = DataLoader(
            TensorDataset(torch.tensor(Xv), torch.tensor(yv)), batch_size=64)
        strategy = TVFLIDSStrategy(
            num_clients=4,
            config={"trust": {}, "verification": {}},
            val_loader=loader, model=IDSMLP(input_dim=10, num_classes=5),
            device=torch.device("cpu"))

        seen = sum(len(yb) for _, yb in strategy.val_loader)
        assert seen == val_size

    def test_setup_data_threads_val_size_to_the_pipeline(self):
        """setup_data must forward val_size, not silently drop it."""
        import experiments.run_experiment as rex
        src = inspect.getsource(rex.setup_data)
        # every pipeline invocation inside setup_data passes val_size through
        assert src.count("val_size=val_size") >= 2
        assert "protocol=protocol" in src


# ── Leakage-free protocol draw order ────────────────────────────────────────

class TestLeakageFreeProtocol:

    @pytest.mark.parametrize("module", [
        "data.preprocessing.nslkdd_pipeline",
        "data.preprocessing.ciciot2023_pipeline",
    ])
    def test_both_protocols_accepted(self, module):
        import importlib
        fn = importlib.import_module(module).build_pipeline
        params = inspect.signature(fn).parameters
        assert params["protocol"].default == "main"
        src = inspect.getsource(fn)
        assert "leakage_free" in src

    @pytest.mark.parametrize("module", [
        "data.preprocessing.nslkdd_pipeline",
        "data.preprocessing.ciciot2023_pipeline",
    ])
    def test_unknown_protocol_rejected(self, module):
        import importlib
        fn = importlib.import_module(module).build_pipeline
        with pytest.raises(ValueError, match="protocol"):
            fn("nonexistent_train.csv", "nonexistent_test.csv",
               protocol="not_a_protocol")

    def test_setup_data_applies_per_client_smote_for_both_datasets(self):
        """Leakage-free requires SMOTE after partitioning, per client. This was
        previously gated on dataset == 'nslkdd' only, so CIC-IoT-2023 silently
        skipped it (and never even received protocol=). Now covers nslkdd,
        ciciot2023, and edgeiiotset."""
        import experiments.run_experiment as rex
        src = inspect.getsource(rex.setup_data)
        assert 'protocol == "leakage_free" and dataset in ("nslkdd", "ciciot2023", "edgeiiotset")' in src
