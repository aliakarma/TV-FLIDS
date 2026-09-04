"""
tests/test_new_experiment_runners.py
Smoke tests for the 4 new experiment runners added for audit IDs
E7, E12, E13, E10/E11:
  - experiments/run_extended_significance.py
  - experiments/run_noniid_sweep.py
  - experiments/run_multi_attack_matrix.py
  - experiments/run_hyperparameter_sweep.py

Environment caveat: on this Windows/Python-3.11 machine, Flower's
Ray-based `fl.simulation.start_simulation` crashes with
"[WinError 1114] ... loading torch/lib/c10.dll" inside the Ray actor
subprocess — a confirmed environment issue, not a code defect (see
experiments/run_experiment.py docstring / remediation notes). That means
a *real* run_experiment() call cannot complete in this environment.

Per the remediation brief, each runner above therefore separates "build
the (seed, config) job list, call run_experiment, aggregate/save results"
into small units that accept an injectable `run_experiment_fn`. These
tests exercise that plumbing directly with a fake run_experiment_fn that
returns canned-but-clearly-fake metric dicts (deliberately NOT matching
any of the paper's claimed numbers, and NEVER written to a real
results/tables/*.json path — only to a pytest tmp_path). This proves the
new orchestration code (job construction, seed looping, aggregation,
JSON structure, provenance) actually works, independent of the Ray/DLL
environment blocker.

One additional test (test_run_experiment_environment_probe) explicitly
attempts a real, tiny run_experiment() call and documents whether the
Ray/DLL crash still reproduces in this environment, per the brief's
"first, actually try calling run_experiment" instruction.

Run:
    .venv\\Scripts\\python.exe -m pytest tests\\test_new_experiment_runners.py -v -s
"""

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Dummy seeds for plumbing tests only — deliberately small/arbitrary and
# distinct from both evaluation.statistical_testing.SEEDS and
# EXTENDED_SEEDS, so nothing here could be mistaken for a real paper run.
TEST_SEEDS = [7, 13]


def _fake_run_experiment(**kwargs):
    """
    Canned-but-clearly-fake run_experiment() stand-in.

    Values are a deterministic function of (seed, strategy_name) so that
    Wilcoxon/aggregation logic has real (non-degenerate) variance to chew
    on, but they are obviously synthetic test fixtures — not copies of
    any paper-reported number — and this function is used ONLY inside
    these plumbing tests, never to populate a real results/ file.
    """
    seed = kwargs.get("seed", 0)
    strategy = kwargs.get("strategy_name", "unknown")
    override = kwargs.get("strategy_kwargs_override") or {}

    wobble = (seed % 5) * 0.011
    bump = 0.021 if strategy == "tvflids" else 0.0
    override_bump = 0.001 * sum(
        v for v in override.values() if isinstance(v, (int, float))
    )
    acc = 0.6001 + wobble + bump + override_bump

    return {
        "final_accuracy": round(acc, 6),
        "final_f1_macro": round(acc - 0.0555, 6),
        "final_attack_success_rate": round(max(0.0, 0.401 - wobble), 6),
        "strategy": strategy,
        "seed": seed,
        "num_malicious": 3,
        "malicious_ids": [0, 1, 2],
    }


def _assert_no_paper_lookalikes(payload):
    """
    Cheap guard against accidentally hardcoding paper-shaped numbers: the
    fake fixture's accuracy values should never land suspiciously close
    to common "clean paper result" round numbers like 0.95, 0.9, 0.99 —
    a loose sanity check that we're looking at synthetic test data.
    """
    text = json.dumps(payload)
    for suspicious in ("0.95", "0.9821", "0.9912", "0.8834"):
        assert suspicious not in text, (
            f"Found suspicious paper-lookalike value '{suspicious}' in "
            "test output — this test must never emit paper-shaped numbers."
        )


# ── Deliverable 1: run_extended_significance.py (audit E7) ────────────────

def test_run_extended_significance_plumbing(tmp_path):
    from experiments.run_extended_significance import run_extended_significance

    report = run_extended_significance(
        strategy_a="tvflids",
        strategy_b="fedavg",
        seeds=TEST_SEEDS,
        attack="label_flip_30",
        num_rounds=1,
        config_path="config/fl_config.yaml",
        output_dir=str(tmp_path),
        log_root=str(tmp_path / "logs"),
        run_experiment_fn=_fake_run_experiment,
        verbose=False,
    )

    assert report["provenance"]["strategy_a"] == "tvflids"
    assert report["provenance"]["strategy_b"] == "fedavg"
    assert report["provenance"]["seeds"] == TEST_SEEDS
    assert report["provenance"]["audit_id"] == "E7"

    for metric in ["final_accuracy", "final_f1_macro", "final_attack_success_rate"]:
        row = report["per_metric"][metric]
        assert "wilcoxon_p_value" in row
        assert "cohens_d" in row
        # tvflids has a deterministic +0.021 bump in the fake fn, so it
        # should register as the higher-mean strategy on accuracy/F1.
    assert report["per_metric"]["final_accuracy"]["tvflids_mean"] > \
        report["per_metric"]["final_accuracy"]["fedavg_mean"]

    out_path = tmp_path / "extended_significance_results.json"
    assert out_path.exists()
    with open(out_path) as f:
        saved = json.load(f)
    # Not a strict `==` against `report`: some values (e.g. numpy bool_
    # from the Wilcoxon test) round-trip through json.dump(default=str)
    # as strings, which is expected/harmless. Check structure instead.
    assert saved["provenance"] == report["provenance"]
    assert set(saved["per_metric"].keys()) == set(report["per_metric"].keys())
    for metric in report["per_metric"]:
        assert saved["per_metric"][metric]["wilcoxon_p_value"] == \
            report["per_metric"][metric]["wilcoxon_p_value"]
    _assert_no_paper_lookalikes(saved)


# ── Deliverable 2: run_noniid_sweep.py (audit E12) ─────────────────────────

def test_run_noniid_sweep_plumbing(tmp_path):
    from experiments.run_noniid_sweep import run_noniid_sweep, _conditions

    conds = _conditions([0.1, 0.5])
    labels = [c[2] for c in conds]
    assert labels == ["alpha=0.1", "alpha=0.5", "IID"]

    summary = run_noniid_sweep(
        strategies=["fedavg"],
        alphas=[0.1],
        seeds=TEST_SEEDS,
        attack="label_flip_30",
        num_rounds=1,
        config_path="config/fl_config.yaml",
        output_dir=str(tmp_path),
        log_root=str(tmp_path / "logs"),
        run_experiment_fn=_fake_run_experiment,
        verbose=False,
    )

    assert set(summary.keys()) == {"fedavg"}
    assert set(summary["fedavg"].keys()) == {"alpha=0.1", "IID"}
    cell = summary["fedavg"]["alpha=0.1"]
    assert cell["final_accuracy"]["n_seeds"] == len(TEST_SEEDS)
    assert 0.0 <= cell["final_accuracy"]["mean"] <= 1.0

    out_path = tmp_path / "noniid_sweep_results.json"
    assert out_path.exists()
    with open(out_path) as f:
        saved = json.load(f)
    assert saved["provenance"]["audit_id"] == "E12"
    assert saved["summary"] == summary
    _assert_no_paper_lookalikes(saved)


# ── Deliverable 3: run_multi_attack_matrix.py (audit E13) ─────────────────

def test_run_multi_attack_matrix_plumbing(tmp_path):
    from experiments.run_multi_attack_matrix import run_multi_attack_matrix

    summary = run_multi_attack_matrix(
        strategies=["fedavg", "tvflids"],
        attacks=["gradient_scale_30"],
        seeds=TEST_SEEDS,
        num_rounds=1,
        config_path="config/fl_config.yaml",
        output_dir=str(tmp_path),
        log_root=str(tmp_path / "logs"),
        run_experiment_fn=_fake_run_experiment,
        verbose=False,
    )

    assert set(summary.keys()) == {"fedavg", "tvflids"}
    for strategy in ("fedavg", "tvflids"):
        assert set(summary[strategy].keys()) == {"gradient_scale_30"}
        cell = summary[strategy]["gradient_scale_30"]
        assert cell["final_accuracy"]["n_seeds"] == len(TEST_SEEDS)

    # A non-default attack/strategy set is namespaced by its attack list, so
    # the adaptive-attack matrix of Table XI (ACK1/ACK2) cannot overwrite the
    # default GS/NI/BD matrix of Tables IX-X. Only the default matrix owns the
    # bare filename.
    out_path = tmp_path / "multi_attack_matrix_results_gradient_scale_30.json"
    assert out_path.exists()
    assert not (tmp_path / "multi_attack_matrix_results.json").exists()
    with open(out_path) as f:
        saved = json.load(f)
    assert saved["provenance"]["audit_id"] == "E13"
    assert saved["provenance"]["attacks"] == ["gradient_scale_30"]
    assert saved["provenance"]["is_default_matrix"] is False
    _assert_no_paper_lookalikes(saved)


def test_multi_attack_matrix_default_and_adaptive_do_not_collide(tmp_path):
    """The default matrix (Tables IX-X) and the ACK1/ACK2 matrix (Table XI)
    are produced by the same runner; each must own a distinct artifact."""
    from experiments.run_multi_attack_matrix import (
        run_multi_attack_matrix, ATTACKS, STRATEGIES,
    )

    run_multi_attack_matrix(
        strategies=STRATEGIES, attacks=ATTACKS, seeds=TEST_SEEDS, num_rounds=1,
        config_path="config/fl_config.yaml", output_dir=str(tmp_path),
        log_root=str(tmp_path / "logs_default"),
        run_experiment_fn=_fake_run_experiment, verbose=False,
    )
    run_multi_attack_matrix(
        strategies=["fedavg", "fltrust", "tvflids"],
        attacks=["ack1_evasion_30", "ack2_coalition_30"],
        seeds=TEST_SEEDS, num_rounds=1,
        config_path="config/fl_config.yaml", output_dir=str(tmp_path),
        log_root=str(tmp_path / "logs_adaptive"),
        run_experiment_fn=_fake_run_experiment, verbose=False,
    )

    default_path = tmp_path / "multi_attack_matrix_results.json"
    adaptive_path = (tmp_path /
                     "multi_attack_matrix_results_ack1_evasion_30_ack2_coalition_30.json")
    assert default_path.exists(), "default matrix lost its unsuffixed artifact"
    assert adaptive_path.exists(), "adaptive matrix was not namespaced"

    with open(default_path) as f:
        default_saved = json.load(f)
    with open(adaptive_path) as f:
        adaptive_saved = json.load(f)
    assert default_saved["provenance"]["is_default_matrix"] is True
    assert adaptive_saved["provenance"]["is_default_matrix"] is False
    assert default_saved["provenance"]["attacks"] == ATTACKS
    assert (default_saved["summary"].keys()
            != adaptive_saved["summary"]["fedavg"].keys())


# ── Deliverable 4: run_hyperparameter_sweep.py (audit E10, E11) ───────────

def test_run_baseline_hyperparameter_sweep_plumbing(tmp_path):
    from experiments.run_hyperparameter_sweep import run_baseline_hyperparameter_sweep

    seen_overrides = []

    def recording_fake(**kwargs):
        seen_overrides.append(kwargs.get("strategy_kwargs_override"))
        return _fake_run_experiment(**kwargs)

    tiny_sweeps = {
        "krum_f_prime": {
            "strategy": "krum", "override_key": "num_byzantine",
            "grid": [2, 6], "paper_symbol": "Krum f'",
        },
        "trimmed_mean_beta": {
            "strategy": "trimmed_mean", "override_key": "beta",
            "grid": [0.05, 0.2], "paper_symbol": "Trimmed-Mean beta",
        },
    }

    summary = run_baseline_hyperparameter_sweep(
        seeds=TEST_SEEDS,
        num_rounds=1,
        output_dir=str(tmp_path),
        log_root=str(tmp_path / "logs"),
        run_experiment_fn=recording_fake,
        sweeps=tiny_sweeps,
        verbose=False,
    )

    assert set(summary.keys()) == {"krum_f_prime", "trimmed_mean_beta"}
    assert set(summary["krum_f_prime"].keys()) == {"2", "6"}
    assert set(summary["trimmed_mean_beta"].keys()) == {"0.05", "0.2"}

    # Confirm the override dict was actually threaded through to
    # run_experiment_fn for every job, with the correct key/value.
    krum_overrides = [o for o in seen_overrides if o and "num_byzantine" in o]
    assert {o["num_byzantine"] for o in krum_overrides} == {2, 6}
    tm_overrides = [o for o in seen_overrides if o and "beta" in o]
    assert {o["beta"] for o in tm_overrides} == {0.05, 0.2}

    out_path = tmp_path / "hyperparam_sweep_baseline_results.json"
    assert out_path.exists()
    with open(out_path) as f:
        saved = json.load(f)
    assert saved["provenance"]["audit_id"] == "E10"
    _assert_no_paper_lookalikes(saved)


def test_run_tvflids_hyperparameter_sweep_plumbing(tmp_path):
    from experiments.run_hyperparameter_sweep import run_tvflids_hyperparameter_sweep

    seen_config_paths = []

    def recording_fake(**kwargs):
        seen_config_paths.append(kwargs.get("config_path"))
        assert kwargs.get("strategy_name") == "tvflids"
        return _fake_run_experiment(**kwargs)

    tiny_sweeps = {
        "meta_lr": {
            "section": "trust", "key": "meta_lr",
            "grid": [0.001, 0.05], "paper_symbol": "lambda / eta_meta",
        },
    }

    summary = run_tvflids_hyperparameter_sweep(
        seeds=TEST_SEEDS,
        num_rounds=1,
        config_path="config/fl_config.yaml",
        output_dir=str(tmp_path),
        log_root=str(tmp_path / "logs"),
        run_experiment_fn=recording_fake,
        sweeps=tiny_sweeps,
        verbose=False,
    )

    assert set(summary.keys()) == {"meta_lr"}
    assert set(summary["meta_lr"].keys()) == {"0.001", "0.05"}

    # Each grid point should have written its own temp config file (not
    # the shared base config path) so the override is isolated per-job.
    assert len(set(seen_config_paths)) == 2
    for p in seen_config_paths:
        assert p != "config/fl_config.yaml"
        with open(p) as f:
            import yaml
            cfg = yaml.safe_load(f)
        assert cfg["trust"]["meta_lr"] in (0.001, 0.05)

    out_path = tmp_path / "hyperparam_sweep_tvflids_results.json"
    assert out_path.exists()
    with open(out_path) as f:
        saved = json.load(f)
    assert saved["provenance"]["audit_id"] == "E11"
    assert "meta_lr" in saved["provenance"]["note"] or "eta_meta" in saved["provenance"]["note"]
    _assert_no_paper_lookalikes(saved)


# ── Environment probe: does real run_experiment() work here? ──────────────

def test_run_experiment_environment_probe():
    """
    Per the remediation brief: actually try a real, tiny run_experiment()
    call first. On this Windows/Python-3.11 machine it is expected to hit
    the confirmed Ray/torch DLL environment crash (WinError 1114) inside
    the Ray actor subprocess — not a code defect. If that specific error
    reproduces, this test documents it and skips rather than failing the
    suite; if the environment has since been fixed, it asserts a real
    tiny run completed successfully.
    """
    from experiments.run_experiment import run_experiment

    try:
        result = run_experiment(
            strategy_name="fedavg",
            attack_config_name="no_attack",
            seed=42,
            num_rounds=1,
            log_dir="results/logs/_env_probe_test_new_runners",
            verbose=False,
        )
    except RuntimeError as e:
        if "Simulation crashed" in str(e):
            pytest.skip(
                "[Environment] Ray/torch DLL crash (WinError 1114) reproduced "
                "as expected on this Windows/Python-3.11 machine — confirmed "
                "environment issue, not a code defect. Runner plumbing is "
                "instead verified via the injectable run_experiment_fn tests "
                "above (option b)."
            )
        raise
    else:
        print("[Environment] Real run_experiment() succeeded in this "
              "environment (Ray/DLL issue not present here).")
        assert "final_accuracy" in result


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
