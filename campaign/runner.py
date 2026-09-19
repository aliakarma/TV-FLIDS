"""
campaign/runner.py
Experimental campaign runner with checkpointing, resume, failure handling,
seed isolation, and dataset blocker protection.
Reference: IEEE TIFS Manuscript §VI, §VII, and Supplementary §S3–S8.

Guarantees:
  - Idempotent resume: completed runs with valid metrics are never re-executed.
  - Interrupted / corrupted runs are explicitly detected and safely recovered.
  - Configuration mismatches during resume are loud errors, preventing cross-run contamination.
  - Dataset blockers: runs on absent raw datasets (CIC-IoT-2023, Edge-IIoTset) fail with BLOCKED
    and never silently substitute synthetic fixtures.
  - Seed isolation: every run begins from an isolated, reproducible RNG state.
  - Git provenance & dirty tree protection: refuses uncommitted source modifications.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import traceback
from typing import Any, Dict, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from campaign.run_spec import RunSpecification
from campaign.config_freezer import freeze_configuration, resolve_dataset_provenance
from campaign.artifacts import (
    get_artifact_directory,
    read_metrics,
    read_status,
    write_metrics,
    write_status,
)
from utils.provenance import git_state
from utils.seed import set_all_seeds
from experiments.run_experiment import run_experiment


def check_dataset_availability(dataset: str) -> Tuple[bool, str]:
    """
    Check if the required raw dataset files are genuinely present on disk.
    Returns (is_available, reason).
    """
    if dataset == "nslkdd":
        # Check standard NSL-KDD raw files
        train_path = os.path.join(ROOT, "data", "raw", "KDDTrain+.txt")
        test_path = os.path.join(ROOT, "data", "raw", "KDDTest+.txt")
        if os.path.exists(train_path) and os.path.exists(test_path):
            return True, "NSL-KDD raw files present."
        return False, f"Missing raw files: {train_path} or {test_path}"

    elif dataset == "ciciot2023":
        train_path = os.path.join(ROOT, "data", "raw", "CICIoT2023_train.csv")
        test_path = os.path.join(ROOT, "data", "raw", "CICIoT2023_test.csv")
        if os.path.exists(train_path) and os.path.exists(test_path):
            return True, "CIC-IoT-2023 raw files present."
        return False, "Raw multi-GB shards for CIC-IoT-2023 are not present on disk (Stage 8 BLOCKED state)."

    elif dataset == "edgeiiotset":
        train_path = os.path.join(ROOT, "data", "raw", "EdgeIIoTset_train.csv")
        test_path = os.path.join(ROOT, "data", "raw", "EdgeIIoTset_test.csv")
        if os.path.exists(train_path) and os.path.exists(test_path):
            return True, "Edge-IIoTset raw files present."
        return False, "Raw multi-GB files for Edge-IIoTset are not present on disk (Stage 8 BLOCKED state)."

    return False, f"Unknown dataset '{dataset}'."


def check_provenance_compatibility(
    spec: RunSpecification,
    stored_config: Dict[str, Any],
    current_git: Optional[Dict[str, Any]] = None,
    current_dataset_prov: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Verify that an existing completed run artifact matches the requested
    implementation commit, dataset provenance, and scientific configuration.

    Returns (is_compatible, error_message).
    """
    # 1. Scientific configuration match
    stored_sci = stored_config.get("scientific_configuration", {})
    curr_sci = spec.to_scientific_dict()
    if stored_sci != curr_sci:
        return False, (
            f"Completed artifact exists for run_id={spec.run_id}, but scientific configuration differs!\n"
            f"Stored: {stored_sci}\n"
            f"Requested: {curr_sci}\n"
            f"Refusing cached-result reuse."
        )

    # 2. Implementation commit match
    curr_git_info = current_git if current_git is not None else git_state()
    curr_commit = curr_git_info.get("git_commit")
    stored_commit = stored_config.get("git_provenance", {}).get("git_commit")

    if stored_commit != curr_commit:
        return False, (
            f"Completed artifact exists for run_id={spec.run_id}, but provenance differs:\n"
            f"stored_commit={stored_commit}\n"
            f"current_commit={curr_commit}\n"
            f"Refusing cached-result reuse."
        )

    # 3. Dataset provenance match
    curr_data_prov = (
        current_dataset_prov
        if current_dataset_prov is not None
        else resolve_dataset_provenance(spec.dataset)
    )
    stored_data_prov = stored_config.get("dataset_provenance")

    if stored_data_prov != curr_data_prov:
        return False, (
            f"Completed artifact exists for run_id={spec.run_id}, but dataset provenance differs:\n"
            f"stored_dataset_provenance={stored_data_prov}\n"
            f"current_dataset_provenance={curr_data_prov}\n"
            f"Refusing cached-result reuse."
        )

    return True, None


class CampaignRunner:
    """
    Executes or resumes experimental runs with full reproducibility guarantees.
    """

    def __init__(
        self,
        base_results_dir: str = "results",
        allow_dirty: bool = False,
        current_git: Optional[Dict[str, Any]] = None,
        current_dataset_prov: Optional[Dict[str, Any]] = None,
    ):
        self.base_results_dir = base_results_dir
        self.allow_dirty = allow_dirty
        self.current_git = current_git
        self.current_dataset_prov = current_dataset_prov

    def execute_run(
        self,
        spec: RunSpecification,
        dry_run: bool = False,
        force: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute a single RunSpecification adhering to all reproducibility and safety invariants.
        """
        output_dir = get_artifact_directory(spec, base_dir=self.base_results_dir)
        os.makedirs(output_dir, exist_ok=True)

        # ── 1. Check Dataset Availability (Part M: Dataset Blocker Handling) ──
        available, reason = check_dataset_availability(spec.dataset)
        if not available:
            if verbose:
                print(f"[Runner] Execution BLOCKED for run {spec.run_id}: {reason}")
            write_status(
                output_dir,
                status="BLOCKED",
                error_message=reason,
                extra_info={"dataset": spec.dataset, "run_id": spec.run_id},
            )
            return {
                "status": "BLOCKED",
                "run_id": spec.run_id,
                "reason": reason,
                "dataset": spec.dataset,
            }

        # ── 2. Checkpoint & Resume Evaluation (Provenance-Safe) ───────────────
        existing_status = read_status(output_dir)
        config_path = os.path.join(output_dir, "config.json")

        if not force and existing_status and existing_status.get("status") == "COMPLETED":
            # Verify configuration and provenance match
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        stored_config = json.load(f)
                except (OSError, json.JSONDecodeError) as exc:
                    if verbose:
                        print(f"[Runner] Corrupted config.json in {output_dir}: {exc}. Restarting run.")
                    stored_config = None

                if stored_config is not None:
                    stored_run_id = stored_config.get("run_id")
                    if stored_run_id != spec.run_id:
                        raise RuntimeError(
                            f"Configuration mismatch during resume! Stored run_id "
                            f"'{stored_run_id}' does not match expected run_id '{spec.run_id}'."
                        )

                    compatible, reason = check_provenance_compatibility(
                        spec=spec,
                        stored_config=stored_config,
                        current_git=self.current_git,
                        current_dataset_prov=self.current_dataset_prov,
                    )
                    if not compatible:
                        raise RuntimeError(reason)
            else:
                if verbose:
                    print(f"[Runner] Status was COMPLETED but config.json is missing in {output_dir}. Restarting run.")

            cached_metrics = read_metrics(output_dir)
            if cached_metrics is not None:
                if verbose:
                    print(f"[Runner] Run {spec.run_id} is already COMPLETED with matching provenance. Reusing cached results.")
                return {
                    "status": "COMPLETED",
                    "run_id": spec.run_id,
                    "metrics": cached_metrics,
                    "resumed": True,
                }
            else:
                if verbose:
                    print(f"[Runner] Status was COMPLETED but metrics.json is missing or corrupted. Restarting.")

        elif not force and existing_status and existing_status.get("status") == "RUNNING":
            if verbose:
                print(f"[Runner] Found interrupted run (status=RUNNING) in {output_dir}. Restarting cleanly.")

        # ── 3. Seed Isolation (Part K) ─────────────────────────────────────────
        set_all_seeds(spec.seed)

        # ── 4. Configuration Freezing (Part F) & Dirty Tree Protection (Part L) ─
        try:
            frozen_config = freeze_configuration(
                spec,
                output_dir,
                allow_dirty=self.allow_dirty,
                git_info=self.current_git,
                dataset_prov=self.current_dataset_prov,
            )
        except RuntimeError as exc:
            # Dirty tree failure
            write_status(output_dir, status="FAILED", error_message=str(exc))
            raise

        # ── 5. Dry-Run Handling ────────────────────────────────────────────────
        if dry_run:
            write_status(
                output_dir,
                status="SKIPPED",
                extra_info={"reason": "Dry run execution requested"},
            )
            if verbose:
                print(f"[Runner] Dry run completed for {spec.run_id} (block {spec.block}, {spec.strategy} on {spec.dataset}).")
            return {
                "status": "SKIPPED",
                "run_id": spec.run_id,
                "dry_run": True,
            }

        # ── 6. Execution Lifecycle: Mark RUNNING (Part H) ──────────────────────
        write_status(output_dir, status="RUNNING")

        # ── 7. Execute Simulation ──────────────────────────────────────────────
        start_time = datetime.datetime.now(datetime.timezone.utc)
        try:
            # Execute through main experiment runner
            summary = run_experiment(
                strategy_name=spec.strategy,
                attack_config_name=spec.attack,
                dataset=spec.dataset,
                partition_type=spec.partition_type,
                alpha=spec.alpha,
                seed=spec.seed,
                num_rounds=spec.num_rounds,
                log_dir=os.path.join(output_dir, "logs"),
                verbose=verbose,
                val_size=spec.val_size,
                protocol=spec.protocol,
                strategy_kwargs_override=spec.strategy_params if spec.strategy_params else None,
                attack_variant=spec.attack_variant,
                knowledge_tier=spec.knowledge_tier,
                on_off_k=spec.on_off_k,
            )

            # Format and save canonical metrics (Part J)
            metrics_payload = {
                "final_accuracy": summary.get("final_accuracy", 0.0),
                "final_f1_macro": summary.get("final_f1_macro", 0.0),
                "final_attack_success_rate": summary.get("final_attack_success_rate", 0.0),
                "final_attack_recall": summary.get("final_attack_recall"),
                "final_benign_routing_phi": summary.get("final_benign_routing_phi"),
                "compute_overhead_ms": summary.get("compute_overhead_ms", {}),
                "strategy_diagnostics": {
                    "comm_overhead_pct": summary.get("comm_overhead_pct"),
                    "num_malicious": summary.get("num_malicious"),
                },
                "attack_diagnostics": {
                    "attack": spec.attack,
                    "knowledge_tier": spec.knowledge_tier,
                    "variant": spec.attack_variant,
                },
            }
            write_metrics(output_dir, metrics_payload)

            # Write metadata.json
            elapsed = (datetime.datetime.now(datetime.timezone.utc) - start_time).total_seconds()
            metadata_path = os.path.join(output_dir, "metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump({
                    "run_id": spec.run_id,
                    "commit_hash": frozen_config["git_provenance"]["git_commit"],
                    "elapsed_seconds": elapsed,
                    "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                }, f, indent=2)

            # Mark COMPLETED
            write_status(output_dir, status="COMPLETED")
            if verbose:
                print(f"[Runner] Successfully completed run {spec.run_id}.")

            return {
                "status": "COMPLETED",
                "run_id": spec.run_id,
                "metrics": metrics_payload,
            }

        except Exception as exc:
            # Failure handling (Part H)
            tb = traceback.format_exc()
            err_msg = str(exc)
            write_status(
                output_dir,
                status="FAILED",
                error_message=err_msg,
                traceback_str=tb,
            )
            if verbose:
                print(f"[Runner] Execution FAILED for {spec.run_id}: {err_msg}")
            return {
                "status": "FAILED",
                "run_id": spec.run_id,
                "error": err_msg,
                "traceback": tb,
            }


def main():
    parser = argparse.ArgumentParser(description="TV-FLIDS Campaign Runner")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry-run without executing simulation")
    parser.add_argument("--block", type=str, default="B2", help="Block identifier (default: B2)")
    parser.add_argument("--dataset", type=str, default="nslkdd", help="Dataset name (default: nslkdd)")
    parser.add_argument("--strategy", type=str, default="tvflids", help="Strategy name (default: tvflids)")
    parser.add_argument("--attack", type=str, default="label_flip_30", help="Attack name (default: label_flip_30)")
    parser.add_argument("--seed", type=int, default=42, help="Seed (default: 42)")
    parser.add_argument("--rounds", type=int, default=None, help="Override number of rounds")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow running on dirty git working tree")
    parser.add_argument("--force", action="store_true", help="Force rerun even if already completed")
    parser.add_argument("--output-dir", type=str, default="results", help="Base output directory")

    args = parser.parse_args()

    # Find matching run from enumerator or construct single spec
    from campaign.enumerator import CampaignEnumerator
    all_runs = CampaignEnumerator.enumerate_all()
    matching = CampaignEnumerator.filter_runs(
        all_runs,
        block=args.block,
        dataset=args.dataset,
        strategy=args.strategy,
        attack=args.attack,
        seed=args.seed,
    )

    if matching:
        spec = matching[0]
        if args.rounds is not None:
            # Create spec with overridden rounds
            spec_dict = spec.to_scientific_dict()
            spec_dict["num_rounds"] = args.rounds
            # Reconstruct spec
            spec = RunSpecification(
                block=spec.block,
                purpose=spec.purpose,
                dataset=spec.dataset,
                protocol=spec.protocol,
                partition_type=spec.partition_type,
                alpha=spec.alpha,
                val_size=spec.val_size,
                num_clients=spec.num_clients,
                fraction_fit=spec.fraction_fit,
                fraction_evaluate=spec.fraction_evaluate,
                strategy=spec.strategy,
                strategy_params=spec.strategy_params,
                attack=spec.attack,
                attack_params=spec.attack_params,
                attack_variant=spec.attack_variant,
                knowledge_tier=spec.knowledge_tier,
                attack_strength=spec.attack_strength,
                on_off_k=spec.on_off_k,
                model_type=spec.model_type,
                num_rounds=args.rounds,
                local_epochs=spec.local_epochs,
                local_lr=spec.local_lr,
                local_batch_size=spec.local_batch_size,
                seed=spec.seed,
                dp_noise_multiplier=spec.dp_noise_multiplier,
                is_profiling=spec.is_profiling,
                profiling_d=spec.profiling_d,
                profiling_n=spec.profiling_n,
            )
    else:
        print(f"[Runner] No exact match in canonical enumerator for block={args.block}, "
              f"dataset={args.dataset}, strategy={args.strategy}, attack={args.attack}, seed={args.seed}. "
              "Constructing custom specification.")
        spec = RunSpecification(
            block=args.block,
            purpose=f"Custom run: {args.strategy} on {args.dataset}",
            dataset=args.dataset,
            strategy=args.strategy,
            attack=args.attack,
            seed=args.seed,
            num_rounds=args.rounds or 100,
        )

    runner = CampaignRunner(base_results_dir=args.output_dir, allow_dirty=args.allow_dirty)
    result = runner.execute_run(spec, dry_run=args.dry_run, force=args.force, verbose=True)
    print("\n[Execution Result]")
    print(json.dumps({k: v for k, v in result.items() if k != "traceback"}, indent=2))


if __name__ == "__main__":
    main()
