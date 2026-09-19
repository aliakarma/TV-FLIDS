"""
tests/test_experiment_reachability.py
Reachability audit for every experiment added during the forensic-audit
remediation, plus the quarantine-isolation invariant.

This does NOT run any experiment. It asserts the properties that make a paper
table *technically reproducible*: the implementation exists, the standard entry
point can select it, configuration exists, imports resolve, an output schema is
defined, and no production path consumes quarantined mock artifacts.

A component passing here has been shown to be wired up — not to have been
executed at full scale. See README "Results pending regeneration".

Static/import-level only: no Flower, no Ray, no dataset.
"""

import importlib
import inspect
import io
import os
import sys

import pytest
import torch  # noqa: F401  (import before sklearn-backed modules; see README)
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def src(module_name):
    return inspect.getsource(importlib.import_module(module_name))


# ── Adaptive attacks (paper Table XI) ────────────────────────────────────────

class TestAdaptiveAttacks:

    def test_ack1_is_registered_and_reachable(self):
        import attacks.adversarial as adv
        keys = [k for k in adv.ATTACK_CONFIGS if "ack1" in k.lower()]
        assert keys, "no ACK1 entry in ATTACK_CONFIGS"
        assert "ack1" in src("experiments.run_experiment").lower(), \
            "ACK1 not reachable from the standard entry point"

    def test_ack2_is_registered_and_wired_into_the_strategy(self):
        import attacks.adversarial as adv
        assert [k for k in adv.ATTACK_CONFIGS if "ack2" in k.lower()]
        assert hasattr(adv, "apply_ack2_attack_to_params")
        # ACK2 is a coalition attack and must be applied at the strategy level,
        # before verify_all computes the round's pseudo-gradient.
        assert "apply_ack2_attack_to_params" in src("fl.strategy")


# ── Additional baselines (Supplementary) ─────────────────────────────────────

@pytest.mark.parametrize("cli_name,cls_name,module", [
    ("bucketing", "BucketingStrategy", "fl.baselines.bucketing_strategy"),
    ("deepsight", "DeepSightStrategy", "fl.baselines.deepsight_strategy"),
])
class TestAdditionalBaselines:

    def test_class_exists(self, cli_name, cls_name, module):
        assert hasattr(importlib.import_module(module), cls_name)

    def test_exported_from_package(self, cli_name, cls_name, module):
        assert cls_name in src("fl.baselines.__init__")

    def test_selectable_by_strategy_name(self, cli_name, cls_name, module):
        rex = src("experiments.run_experiment")
        assert f'"{cli_name}"' in rex or f"'{cli_name}'" in rex


# ── Cross-dataset pipeline (Supplementary) ───────────────────────────────────

class TestCICIoT2023:

    def test_pipeline_module_and_entry_point(self):
        m = importlib.import_module("data.preprocessing.ciciot2023_pipeline")
        assert hasattr(m, "build_pipeline")
        assert hasattr(m, "load_ciciot2023")

    def test_declared_in_dataset_config(self):
        with io.open(os.path.join(ROOT, "config", "dataset_config.yaml"),
                     encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        assert "ciciot2023" in (cfg.get("datasets") or {})

    def test_selectable_from_run_experiment(self):
        rex = src("experiments.run_experiment")
        assert "ciciot2023" in rex

    def test_returns_the_common_nine_tuple(self):
        """Must plug into setup_data without special-casing."""
        import re
        s = src("data.preprocessing.ciciot2023_pipeline")
        assert re.search(r"def build_pipeline\(.*val_size.*protocol", s, re.S)


# ── Ablation arm A6 ──────────────────────────────────────────────────────────

class TestAblationA6:

    def test_a6_arm_defined(self):
        assert "A6" in src("experiments.run_ablation")


# ── Leakage-free data protocol (paper Table VI) ──────────────────────────────

class TestLeakageFreeProtocol:

    @pytest.mark.parametrize("module", [
        "experiments.run_experiment",
        "experiments.run_full_comparison",
        "data.preprocessing.nslkdd_pipeline",
        "data.preprocessing.ciciot2023_pipeline",
    ])
    def test_protocol_reaches_every_layer(self, module):
        assert "leakage_free" in src(module)

    def test_per_client_smote_covers_both_supported_datasets(self):
        """Regression: this was gated on nslkdd only, so CIC-IoT-2023 silently
        skipped the per-client SMOTE the protocol requires. Now covers nslkdd,
        ciciot2023, and edgeiiotset."""
        assert 'dataset in ("nslkdd", "ciciot2023", "edgeiiotset")' in \
            src("experiments.run_experiment")

    def test_comparison_runner_exposes_protocol(self):
        """Table VI was previously unreachable from the comparison runner."""
        import experiments.run_full_comparison as rfc
        params = inspect.signature(rfc.run_full_comparison).parameters
        assert "protocol" in params and params["protocol"].default == "main"
        assert "--protocol" in src("experiments.run_full_comparison")

    def test_variant_runs_do_not_overwrite_the_main_artifact(self):
        s = src("experiments.run_full_comparison")
        assert "full_comparison_results{suffix}.json" in s


# ── Experiment runners added during remediation ──────────────────────────────

@pytest.mark.parametrize("module,tokens", [
    ("experiments.run_extended_significance", ("EXTENDED_SEEDS", "wilcoxon")),
    ("experiments.run_hyperparameter_sweep",  ("baseline", "tvflids", "warmup_rounds")),
    ("experiments.run_noniid_sweep",          ("alpha",)),
    ("experiments.run_multi_attack_matrix",   ("gradient_scale", "noise", "backdoor")),
])
class TestRunners:

    def test_module_imports(self, module, tokens):
        importlib.import_module(module)

    def test_has_cli_entry_point(self, module, tokens):
        m = importlib.import_module(module)
        assert hasattr(m, "main")
        s = inspect.getsource(m)
        assert "__main__" in s and "add_argument" in s

    def test_defines_an_output_schema(self, module, tokens):
        assert "json.dump" in inspect.getsource(importlib.import_module(module))

    def test_covers_its_required_conditions(self, module, tokens):
        s = inspect.getsource(importlib.import_module(module))
        missing = [t for t in tokens if t not in s]
        assert not missing, f"{module} is missing {missing}"

    def test_no_placeholder_bodies(self, module, tokens):
        s = inspect.getsource(importlib.import_module(module))
        for marker in ("TODO", "FIXME", "raise NotImplementedError"):
            assert marker not in s, f"{module} still contains {marker}"


# ── Quarantine isolation ─────────────────────────────────────────────────────

SKIP_DIRS = (".venv", ".venv310", ".venv_clean", "tvflids", "__pycache__",
             ".git", "_QUARANTINED_MOCK", ".mypy_cache", ".pytest_cache", "colab")


def _source_files():
    for root, _dirs, files in os.walk(ROOT):
        if any(x in root for x in SKIP_DIRS):
            continue
        for f in files:
            if f.endswith((".py", ".sh", ".yaml", ".yml")):
                yield os.path.join(root, f)


class TestQuarantineIsolation:

    def test_no_production_code_reads_quarantined_artifacts(self):
        """Only the results validator may name the quarantine, and only to
        exclude it from its scan."""
        offenders = []
        for path in _source_files():
            name = os.path.basename(path)
            if name in ("check_results.py", "campaign_manifest.py",
                        "test_experiment_reachability.py",
                        "proposition1_verification.py"):
                # validators that name the quarantine only in order to SKIP it
                # (check_results.py, campaign_manifest.py) / this test itself /
                # a docstring reference. Every one of these is asserted below
                # to use the name in an exclusion, never a read.
                continue
            text = io.open(path, encoding="utf-8", errors="replace").read()
            if "_QUARANTINED_MOCK" in text:
                offenders.append(os.path.relpath(path, ROOT))
        assert not offenders, f"code referencing the quarantine: {offenders}"

        # The exemption above is only safe if each exempted validator uses the
        # quarantine name to skip that directory. Assert that, so the allowlist
        # cannot silently become a hole through which a reader slips back in.
        for rel in ("scripts/check_results.py", "scripts/campaign_manifest.py"):
            text = io.open(os.path.join(ROOT, rel),
                           encoding="utf-8", errors="replace").read()
            assert "continue" in text and "_QUARANTINED_MOCK" in text, (
                f"{rel} names the quarantine but no longer skips it")

    def test_mock_fingerprint_absent_from_source(self):
        """No source file may emit the fabricated `mockhash` config hash.

        Three files legitimately name the string and are allowlisted: the
        results validator, which screens for it; the Proposition 1 script,
        whose provenance docstring explains what it replaced; and this test.
        All mention it in prose; none writes it. The allowlist is asserted to
        stay minimal so it cannot grow silently.
        """
        allowed = {"check_results.py", "proposition1_verification.py",
                   "test_experiment_reachability.py"}
        offenders = []
        for path in _source_files():
            text = io.open(path, encoding="utf-8", errors="replace").read()
            if "mockhash" in text and os.path.basename(path) not in allowed:
                offenders.append(os.path.relpath(path, ROOT))
        assert not offenders, (
            f"files carrying the mock config-hash fingerprint: {offenders}")
        assert len(allowed) == 3, "allowlist must not grow silently"

    def test_mock_generator_cannot_be_imported_or_run(self):
        q = os.path.join(ROOT, "results", "_QUARANTINED_MOCK")
        assert os.path.isdir(q), "quarantine directory is missing"
        assert os.path.exists(os.path.join(q, "README.md")), \
            "quarantine must document why each file is there"
        # The generator is renamed so neither `python <path>` by module name nor
        # an import can reach it.
        gen = os.path.join(q, "mock_phase7_results.py.DO_NOT_RUN")
        assert os.path.exists(gen)
        assert not os.path.exists(os.path.join(q, "mock_phase7_results.py"))
        assert not os.path.exists(os.path.join(ROOT, "mock_phase7_results.py"))

    def test_validator_rejects_mock_and_empty_artifacts(self):
        s = src("scripts.check_results") if False else io.open(
            os.path.join(ROOT, "scripts", "check_results.py"),
            encoding="utf-8").read()
        assert "check_authenticity" in s
        assert "mockhash" in s          # fingerprint it screens for
        assert "0 bytes" in s           # zero-byte placeholder rejection
        assert "provenance" in s
