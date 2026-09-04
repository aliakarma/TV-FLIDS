"""
tests/test_manuscript_figure_generation.py
Exercises scripts/generate_manuscript_figures.py, the results -> manuscript
figure path.

Two properties matter and are tested here:

  1. REFUSAL. With no backing artifact, nothing is written. The manuscript's
     provisional coordinates must never be silently replaced by an invented
     curve, so a missing input is a skip, not a fallback.

  2. FIDELITY. With a backing artifact present, the emitted file contains
     pgfplots \\addplot bodies whose numbers come from that artifact and
     nowhere else.

Every artifact used below is built inside a tmp_path fixture and is obviously
synthetic (0.10, 0.20, ... ). Nothing is written into the repository's own
results/ tree, and no experiment is run.
"""

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import scripts.generate_manuscript_figures as gen  # noqa: E402

ATTACK = "label_flip_30"
SEEDS = [42, 123]
N_ROUNDS = 10


def _write_comparison_log(results_root, strategy, seed, *, accuracy,
                          trust_history=None, malicious_ids=None,
                          strategy_round_logs=None):
    d = os.path.join(results_root, "logs", "comparison",
                     f"{strategy}_{ATTACK}_seed{seed}")
    os.makedirs(d, exist_ok=True)
    extra = {}
    if trust_history is not None:
        extra["trust_history"] = trust_history
    if malicious_ids is not None:
        extra["malicious_ids"] = malicious_ids
    if strategy_round_logs is not None:
        extra["strategy_round_logs"] = strategy_round_logs
    payload = {
        "experiment_name": f"{strategy}_seed{seed}",
        "rounds": [{"round": r + 1, "accuracy": accuracy[r],
                    "f1_macro": accuracy[r] - 0.02}
                   for r in range(len(accuracy))],
        "summary": {"final_accuracy": accuracy[-1]},
        "extra": extra,
    }
    with open(os.path.join(d, "experiment_log.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


@pytest.fixture
def empty_root(tmp_path):
    return str(tmp_path / "results")


@pytest.fixture
def populated_root(tmp_path):
    """A minimal but complete synthetic results tree."""
    root = str(tmp_path / "results")

    # Distinct constant-per-strategy accuracies make provenance checkable.
    accs = {"tvflids": 0.88, "fltrust": 0.86, "krum": 0.81, "fedavg": 0.60}
    trust_history = {"0": [0.9] * N_ROUNDS, "1": [0.8] * N_ROUNDS,
                     "2": [0.02] * N_ROUNDS}
    strategy_round_logs = [
        {"round": r + 1, "adaptive_alpha": 0.30, "adaptive_beta": 0.50,
         "adaptive_gamma": 0.20}
        for r in range(N_ROUNDS)
    ]

    for strategy, acc in accs.items():
        for seed in SEEDS:
            extras = {}
            if strategy == "tvflids":
                extras = dict(trust_history=trust_history, malicious_ids=[2],
                              strategy_round_logs=strategy_round_logs)
            _write_comparison_log(root, strategy, seed,
                                  accuracy=[acc] * N_ROUNDS, **extras)

    tables = os.path.join(root, "tables")
    os.makedirs(tables, exist_ok=True)
    with open(os.path.join(tables, "ratio_sweep_results.json"), "w",
              encoding="utf-8") as fh:
        json.dump({s: {"0.0": {"accuracy_mean": 0.95, "accuracy_std": 0.01},
                       "0.3": {"accuracy_mean": 0.88, "accuracy_std": 0.01}}
                   for s in ("tvflids", "fltrust", "krum", "fedavg")}, fh)
    with open(os.path.join(tables, "dataset_comparison_results.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"ciciot2023": {
            s: [{"final_accuracy": 0.90, "final_f1_macro": 0.87}]
            for s in ("fedavg", "krum", "fltrust", "tvflids")}}, fh)
    return root


class TestRefusesToInvent:

    def test_nothing_written_when_artifacts_are_absent(self, empty_root, tmp_path):
        out = str(tmp_path / "figures")
        rc = gen.main(["--results-root", empty_root, "--out-dir", out])
        assert rc == 0                      # non-check mode does not hard-fail
        assert not os.path.exists(out) or os.listdir(out) == []

    def test_check_mode_exits_nonzero_when_incomplete(self, empty_root, tmp_path):
        rc = gen.main(["--results-root", empty_root,
                       "--out-dir", str(tmp_path / "figures"), "--check"])
        assert rc == 1

    def test_check_mode_writes_nothing_even_when_complete(self, populated_root,
                                                          tmp_path):
        out = str(tmp_path / "figures")
        rc = gen.main(["--results-root", populated_root, "--out-dir", out,
                       "--check"])
        assert rc == 0
        assert not os.path.exists(out) or os.listdir(out) == []

    def test_partial_artifacts_generate_only_what_is_backed(self, populated_root,
                                                            tmp_path):
        os.remove(os.path.join(populated_root, "tables",
                               "ratio_sweep_results.json"))
        out = str(tmp_path / "figures")
        gen.main(["--results-root", populated_root, "--out-dir", out])
        assert os.path.exists(os.path.join(out, "fig_convergence.tex"))
        assert not os.path.exists(os.path.join(out, "fig_robustness.tex"))

    def test_missing_extras_block_trust_and_weights_only(self, tmp_path):
        """A pre-logger-change run has rounds but no extra.* payload."""
        root = str(tmp_path / "results")
        for strategy in ("tvflids", "fltrust", "krum", "fedavg"):
            _write_comparison_log(root, strategy, 42, accuracy=[0.8] * N_ROUNDS)
        out = str(tmp_path / "figures")
        gen.main(["--results-root", root, "--out-dir", out])
        assert os.path.exists(os.path.join(out, "fig_convergence.tex"))
        assert not os.path.exists(os.path.join(out, "fig_trust.tex"))
        assert not os.path.exists(os.path.join(out, "fig_weights.tex"))


class TestGeneratesValidPgfplots:

    @pytest.fixture
    def generated(self, populated_root, tmp_path):
        out = str(tmp_path / "figures")
        rc = gen.main(["--results-root", populated_root, "--out-dir", out])
        assert rc == 0
        return out

    @pytest.mark.parametrize("name", ["convergence", "trust", "weights",
                                      "robustness", "ciciot"])
    def test_every_figure_is_produced(self, generated, name):
        assert os.path.exists(os.path.join(generated, f"fig_{name}.tex"))

    def _read(self, generated, name):
        with open(os.path.join(generated, f"fig_{name}.tex"),
                  encoding="utf-8") as fh:
            return fh.read()

    @pytest.mark.parametrize("name", ["convergence", "trust", "weights",
                                      "robustness", "ciciot"])
    def test_files_carry_a_generated_banner_and_addplots(self, generated, name):
        text = self._read(generated, name)
        assert "GENERATED FILE -- DO NOT EDIT BY HAND" in text
        assert "make manuscript-figures" in text
        assert r"\addplot" in text
        # Balanced braces, so \input cannot break the surrounding axis.
        assert text.count("{") == text.count("}")

    def test_convergence_values_come_from_the_artifact(self, generated):
        text = self._read(generated, "convergence")
        for value, label in [("0.8800", "TV-FLIDS"), ("0.8600", "FLTrust"),
                             ("0.8100", "Krum"), ("0.6000", "FedAvg")]:
            assert value in text
            assert rf"\addlegendentry{{{label}}}" in text

    def test_trust_separates_honest_from_byzantine(self, generated):
        text = self._read(generated, "trust")
        assert "0.8500" in text          # mean of honest clients 0.9 and 0.8
        assert "0.0200" in text          # the single Byzantine client
        assert "N_{\\mathcal{H}}{=}2" in text
        assert "f{=}1" in text
        assert "Trust floor" in text

    def test_weights_emit_all_three_signals(self, generated):
        text = self._read(generated, "weights")
        assert "0.3000" in text and "0.5000" in text and "0.2000" in text
        for sym in (r"\alpha", r"\beta", r"\gamma"):
            assert sym in text

    def test_robustness_uses_ratio_sweep_values(self, generated):
        text = self._read(generated, "robustness")
        assert "(0.00,0.9500)" in text
        assert "(0.30,0.8800)" in text

    def test_ciciot_is_a_bar_series_over_named_strategies(self, generated):
        text = self._read(generated, "ciciot")
        for name in ("FedAvg", "Krum", "FLTrust", "TV-FLIDS"):
            assert f"({name}," in text
        assert r"\legend{Accuracy, Macro-F1}" in text

    def test_regeneration_is_idempotent(self, populated_root, tmp_path):
        out = str(tmp_path / "figures")
        gen.main(["--results-root", populated_root, "--out-dir", out])
        first = self._read(out, "convergence")
        gen.main(["--results-root", populated_root, "--out-dir", out])
        assert self._read(out, "convergence") == first


class TestDownsampling:

    def test_last_round_is_always_kept(self):
        pts = gen._downsample(list(range(100)), stride=5)
        assert pts[0][0] == 1            # rounds are 1-indexed
        assert pts[-1][0] == 100
        assert pts[-1][1] == 99.0

    def test_short_series_is_not_lost(self):
        assert gen._downsample([0.5], stride=5) == [(1, 0.5)]

    def test_empty_series_yields_no_points(self):
        assert gen._downsample([], stride=5) == []

    def test_mean_over_seeds_truncates_to_shortest(self):
        assert gen._mean_over_seeds([[1.0, 3.0, 5.0], [3.0, 5.0]]) == [2.0, 4.0]

    def test_mean_over_seeds_ignores_empty_runs(self):
        assert gen._mean_over_seeds([[], [2.0, 4.0]]) == [2.0, 4.0]
        assert gen._mean_over_seeds([[], []]) == []
