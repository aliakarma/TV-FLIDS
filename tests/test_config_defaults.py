"""
tests/test_config_defaults.py
Guards the shipped configuration against silently disagreeing with the paper.

The specific trap this file closes: `--strategy tvflids_fixed` is ablation A6,
which the paper (Section IX) defines as alpha = beta = gamma = 1/3 with the
meta-gradient disabled. config/fl_config.yaml used to ship 0.4/0.4/0.2, and
only experiments/run_ablation.py's `trust_override` pushed the arm back to
1/3. A direct CLI invocation therefore produced a DIFFERENT arm than the
ablation runner produced under the same name -- a reproducibility trap that no
test caught, because the ablation runner's override made its own path correct.

The base configuration must already BE the intended A6 configuration, so that
correctness does not depend on a caller overriding it.

Also pinned here: the other Table IV / Section IV-A defaults that a previous
pass found silently disabled or mis-set (adaptive threshold annealing, warmup
length, server validation size).
"""

import os
import sys

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.run_ablation import ABLATION_CONFIGS  # noqa: E402
from fl.strategy import TVFLIDSStrategy  # noqa: E402
from models.mlp import IDSMLP  # noqa: E402
from trust.adaptive_trust_scorer import AdaptiveTrustScorer  # noqa: E402
from trust.trust_scorer import TrustScorer  # noqa: E402

FL_CONFIG = os.path.join(ROOT, "config", "fl_config.yaml")
DATASET_CONFIG = os.path.join(ROOT, "config", "dataset_config.yaml")
THIRD = 1.0 / 3.0


@pytest.fixture(scope="module")
def cfg():
    with open(FL_CONFIG, "r") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def dataset_cfg():
    with open(DATASET_CONFIG, "r") as fh:
        return yaml.safe_load(fh)


def _strategy(config, adaptive):
    """Build the strategy exactly as run_experiment.make_strategy does.

    make_strategy passes `adaptive = (strategy_name != "tvflids_fixed")` and
    the loaded config straight through, so constructing it this way exercises
    the same code path a CLI invocation takes.
    """
    torch.manual_seed(0)
    model = IDSMLP(input_dim=8, num_classes=3)
    X = torch.rand(16, 8)
    y = torch.randint(0, 3, (16,))
    loader = DataLoader(TensorDataset(X, y), batch_size=8)
    return TVFLIDSStrategy(num_clients=4, config=config, val_loader=loader,
                           model=model, device=torch.device("cpu"),
                           adaptive=adaptive, seed=42)


class TestTVFLIDSFixedWeights:

    def test_base_config_ships_equal_weights(self, cfg):
        t = cfg["trust"]
        for key in ("alpha", "beta", "gamma"):
            assert t[key] == pytest.approx(THIRD, abs=1e-9), (
                "config/fl_config.yaml trust.%s must be 1/3 (paper Table IV, "
                "ablation A6), got %r" % (key, t[key]))
        assert t["alpha"] + t["beta"] + t["gamma"] == pytest.approx(1.0, abs=1e-5)

    def test_direct_cli_invocation_gives_a6_weights(self, cfg):
        """`--strategy tvflids_fixed` with no overrides == A6."""
        s = _strategy(cfg, adaptive=False)
        assert isinstance(s.trust_scorer, TrustScorer)
        assert not isinstance(s.trust_scorer, AdaptiveTrustScorer)
        assert s.trust_scorer.alpha == pytest.approx(THIRD, abs=1e-9)
        assert s.trust_scorer.beta == pytest.approx(THIRD, abs=1e-9)
        assert s.trust_scorer.gamma == pytest.approx(THIRD, abs=1e-9)

    def test_run_ablation_a6_produces_the_same_weights(self, cfg):
        """The ablation runner and the bare CLI must now agree exactly."""
        a6 = ABLATION_CONFIGS["A6: Fixed Equal Weights"]
        assert a6["strategy"] == "tvflids_fixed"

        overridden = dict(cfg["trust"])
        overridden.update(a6["trust_override"])
        for key in ("alpha", "beta", "gamma"):
            assert overridden[key] == pytest.approx(cfg["trust"][key], abs=1e-9), (
                "run_ablation's A6 override for %s no longer matches the base "
                "config; the CLI trap is back" % key)

    def test_strategy_defaults_match_the_config(self):
        """The in-code fallback must not disagree with the shipped YAML."""
        s = _strategy({"trust": {}, "verification": {}}, adaptive=False)
        assert s.trust_scorer.alpha == pytest.approx(THIRD, abs=1e-9)
        assert s.trust_scorer.beta == pytest.approx(THIRD, abs=1e-9)
        assert s.trust_scorer.gamma == pytest.approx(THIRD, abs=1e-9)

    def test_adaptive_strategy_is_unaffected(self, cfg):
        """`--strategy tvflids` never reads these keys, before or after.

        AdaptiveTrustScorer initializes log-weights at 0, so softmax gives
        (1/3, 1/3, 1/3) per Eq. (11) regardless of the config. This pins that
        the config change did not perturb the proposed method.
        """
        s = _strategy(cfg, adaptive=True)
        assert isinstance(s.trust_scorer, AdaptiveTrustScorer)
        w = s.trust_scorer.get_current_weights()
        assert w["alpha"] == pytest.approx(THIRD, abs=1e-6)
        assert w["beta"] == pytest.approx(THIRD, abs=1e-6)
        assert w["gamma"] == pytest.approx(THIRD, abs=1e-6)


class TestOtherAblationArmsUnchanged:
    """The A6 alignment must not have moved any other arm."""

    @pytest.mark.parametrize("name,expected", [
        ("A3: Similarity Only", {"alpha": 1.0, "beta": 0.0, "gamma": 0.0}),
        ("A4: Accuracy Only", {"alpha": 0.0, "beta": 1.0, "gamma": 0.0}),
    ])
    def test_single_signal_arms_keep_their_overrides(self, name, expected):
        assert ABLATION_CONFIGS[name]["trust_override"] == expected

    def test_full_and_a1_a2_a5_do_not_pin_weights(self):
        for name in ("TV-FLIDS (Full)", "A1: No Verification", "A5: IID Data"):
            assert ABLATION_CONFIGS[name]["trust_override"] is None
        assert ABLATION_CONFIGS["A2: No Memory (decay=0)"]["trust_override"] == {
            "memory_decay": 0.0}


class TestGateDefaults:
    """Table IV / Section IV-A values that a prior pass found disabled."""

    def test_adaptive_thresholds_enabled_by_default(self, cfg):
        assert cfg["verification"]["adaptive_thresholds"] is True

    def test_strategy_honours_the_config_flag(self, cfg):
        assert _strategy(cfg, adaptive=True).use_adaptive_thresholds is True
        off = {**cfg, "verification": {**cfg["verification"],
                                       "adaptive_thresholds": False}}
        assert _strategy(off, adaptive=True).use_adaptive_thresholds is False

    def test_nominal_thresholds_match_table_iv(self, cfg):
        v = cfg["verification"]
        assert v["loss_threshold"] == 0.0          # tau_L
        assert v["cosine_threshold"] == 0.0        # tau_C, never annealed
        assert v["zscore_threshold"] == 2.5        # tau_z
        assert v["warmup_rounds"] == 20            # T_warm

    def test_warmup_endpoints_match_the_schedule(self, cfg):
        v = cfg["verification"]
        assert v["warmup_loss_threshold"] == -0.1
        # tau_z(0) = zscore_threshold + offset = 2.5 + 0.5 = 3.0 (Eq. (5)).
        assert v["zscore_threshold"] + v["warmup_zscore_offset"] == 3.0

    def test_single_warmup_knob_drives_both_schedules(self, cfg):
        s = _strategy(cfg, adaptive=True)
        assert s.warmup_rounds == cfg["verification"]["warmup_rounds"]


class TestServerValidationSize:

    def test_val_size_is_two_thousand(self, dataset_cfg):
        assert dataset_cfg["server"]["val_size"] == 2000
        assert dataset_cfg["server"]["val_stratified"] is True


class TestOtherFLDefaults:

    def test_protocol_constants_match_table_iv(self, cfg):
        fl = cfg["federated_learning"]
        assert fl["num_clients"] == 20
        assert fl["num_rounds"] == 100
        assert fl["fraction_fit"] == 0.5
        assert fl["local_epochs"] == 5
        assert fl["local_batch_size"] == 256
        assert fl["local_lr"] == 0.001

    def test_trust_constants_match_table_iv(self, cfg):
        t = cfg["trust"]
        assert t["memory_decay"] == 0.9
        assert t["min_trust"] == 0.01
        assert t["meta_lr"] == 0.01

    def test_seeds_match_table_iv(self, cfg):
        assert cfg["evaluation"]["seeds"] == [42, 123, 456, 789, 1337]
