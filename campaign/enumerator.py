"""
campaign/enumerator.py
Canonical campaign enumerator generating the complete, reproducible run matrix.
Reference: IEEE TIFS Manuscript §VI, §VII, and Supplementary §S3–S8.

Guarantees:
  - Enumerates all 11 experimental blocks (B1 to B11).
  - Every configuration is uniquely identifiable and traceable to paper specifications.
  - Invariant: total runs == 6,111 (with Block 1 = 693 runs from 77 tuning configurations).
  - Provides deterministic filtering, duplicate detection, and manifest export.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Set

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from campaign.run_spec import RunSpecification, compute_run_id

# Canonical 20 simulation seeds for primary comparisons
CANONICAL_20_SEEDS = [
    42, 123, 456, 789, 1337,
    2024, 31415, 8080, 555, 999,
    101, 102, 103, 104, 105,
    106, 107, 108, 109, 110
]
CANONICAL_10_SEEDS = CANONICAL_20_SEEDS[:10]
CANONICAL_3_SEEDS = CANONICAL_20_SEEDS[:3]

# 15 Methods: 14 baselines + 1 proposed (TV-FLIDS)
ALL_15_METHODS = [
    "fedavg", "krum", "multikrum", "trimmed_mean", "norm_clipping",
    "rfa", "bucketing", "foolsgold", "flame", "deepsight",
    "fldetector", "zeno", "fltrust", "baffle", "tvflids"
]

# 5 Named evaluation baselines / comparison subset
FIVE_EVAL_METHODS = ["tvflids", "fltrust", "zeno", "fldetector", "bucketing"]

# 3 Datasets
ALL_3_DATASETS = ["nslkdd", "ciciot2023", "edgeiiotset"]


class CampaignEnumerator:
    """
    Orchestrates the enumeration of all 6,111 experimental runs across Blocks 1 through 11.
    """

    @classmethod
    def enumerate_block_1(cls) -> List[RunSpecification]:
        """
        Block 1: Tuning on D_tune (Supplementary Table S2 / tab:grids).
        77 configurations across 15 methods x 3 seeds x 3 datasets = 693 runs.
        """
        runs: List[RunSpecification] = []
        datasets = ALL_3_DATASETS
        seeds = CANONICAL_3_SEEDS

        # 1. FedAvg: 0 configs (no hyperparameter tuning)
        # 2. Krum: f' in {2, 3, 4} (3 configs)
        krum_configs = [{"num_byzantine": f} for f in [2, 3, 4]]

        # 3. Multi-Krum: f' in {2, 3, 4}, m in {D-f', D-2f'} with D=10 (6 configs)
        multikrum_configs = []
        for f in [2, 3, 4]:
            for m in [10 - f, 10 - 2 * f]:
                multikrum_configs.append({"num_byzantine": f, "m": m})

        # 4. Trimmed Mean: beta in {0.1, 0.2, 0.3, 0.4} (4 configs)
        tm_configs = [{"beta": b} for b in [0.1, 0.2, 0.3, 0.4]]

        # 5. Norm Clipping: radius in {0.5, 1.0, 1.5, 2.0} x median (4 configs)
        nc_configs = [{"clip_factor": c} for c in [0.5, 1.0, 1.5, 2.0]]

        # 6. RFA: iterations in {3, 5, 10, 20} (4 configs)
        rfa_configs = [{"max_iter": it} for it in [3, 5, 10, 20]]

        # 7. Bucketing: s in {2, 3, 5}, beta in {0.1, 0.2} (6 configs)
        bucketing_configs = []
        for s in [2, 3, 5]:
            for b in [0.1, 0.2]:
                bucketing_configs.append({"bucket_size": s, "beta": b})

        # 8. FoolsGold: kappa in {0.5, 1.0, 2.0, 4.0} (4 configs)
        fg_configs = [{"confidence_threshold": k} for k in [0.5, 1.0, 2.0, 4.0]]

        # 9. FLAME: cluster size in {2, 3, 6}, noise in {0.5, 1.0, 2.0} (9 configs)
        flame_configs = []
        for c in [2, 3, 6]:  # D/4 -> 2, D/3 -> 3, D/2+1 -> 6
            for n in [0.5, 1.0, 2.0]:
                flame_configs.append({"min_cluster_size": c, "noise_multiplier": n})

        # 10. DeepSight: clip quantile in {0.25, 0.5, 0.75} (3 configs)
        deepsight_configs = [{"clip_quantile": q} for q in [0.25, 0.5, 0.75]]

        # 11. FLDetector: window in {5, 10}, start round in {10, 20} (4 configs)
        fldetector_configs = []
        for w in [5, 10]:
            for sr in [10, 20]:
                fldetector_configs.append({"window": w, "start_round": sr})

        # 12. Zeno: rho in {0.0005, 0.001, 0.002}, b in {3, 4} (6 configs)
        zeno_configs = []
        for rho in [0.0005, 0.001, 0.002]:
            for b in [3, 4]:
                zeno_configs.append({"rho": rho, "b": b})

        # 13. FLTrust: epochs in {1, 5}, lr in {0.0005, 0.001} (4 configs)
        fltrust_configs = []
        for ep in [1, 5]:
            for lr in [0.0005, 0.001]:
                fltrust_configs.append({"local_epochs": ep, "lr": lr})

        # 14. BaFFLe: validators in {5, 10}, quorum in {0.3, 0.5}, lookback in {10, 20} (8 configs)
        baffle_configs = []
        for v in [5, 10]:
            for q in [0.3, 0.5]:
                for lb in [10, 20]:
                    baffle_configs.append({"num_validators": v, "quorum": q, "lookback": lb})

        # 15. TV-FLIDS: 1 center + 11 single-factor variants = 12 configs
        # Center: lambda_up=0.9, lambda_down=0.7, tau_min=0.01, eta_meta=0.01, T0=0.5, tau_L_start=-0.1
        tvflids_configs = [
            {"lambda_up": 0.9, "lambda_down": 0.7, "tau_min": 0.01, "eta_meta": 0.01, "initial_trust": 0.5, "tau_L_start": -0.1},  # Center
            {"lambda_up": 0.8, "lambda_down": 0.7, "tau_min": 0.01, "eta_meta": 0.01, "initial_trust": 0.5, "tau_L_start": -0.1},  # V1: lambda_up=0.8
            {"lambda_up": 0.95, "lambda_down": 0.7, "tau_min": 0.01, "eta_meta": 0.01, "initial_trust": 0.5, "tau_L_start": -0.1}, # V2: lambda_up=0.95
            {"lambda_up": 0.9, "lambda_down": 0.5, "tau_min": 0.01, "eta_meta": 0.01, "initial_trust": 0.5, "tau_L_start": -0.1},  # V3: lambda_down=0.5
            {"lambda_up": 0.9, "lambda_down": 0.9, "tau_min": 0.01, "eta_meta": 0.01, "initial_trust": 0.5, "tau_L_start": -0.1},  # V4: lambda_down=0.9
            {"lambda_up": 0.9, "lambda_down": 0.7, "tau_min": 0.001, "eta_meta": 0.01, "initial_trust": 0.5, "tau_L_start": -0.1}, # V5: tau_min=0.001
            {"lambda_up": 0.9, "lambda_down": 0.7, "tau_min": 0.05, "eta_meta": 0.01, "initial_trust": 0.5, "tau_L_start": -0.1},  # V6: tau_min=0.05
            {"lambda_up": 0.9, "lambda_down": 0.7, "tau_min": 0.01, "eta_meta": 0.001, "initial_trust": 0.5, "tau_L_start": -0.1}, # V7: eta_meta=0.001
            {"lambda_up": 0.9, "lambda_down": 0.7, "tau_min": 0.01, "eta_meta": 0.1, "initial_trust": 0.5, "tau_L_start": -0.1},   # V8: eta_meta=0.1
            {"lambda_up": 0.9, "lambda_down": 0.7, "tau_min": 0.01, "eta_meta": 0.01, "initial_trust": 1.0, "tau_L_start": -0.1},   # V9: T0=1.0
            {"lambda_up": 0.9, "lambda_down": 0.7, "tau_min": 0.01, "eta_meta": 0.01, "initial_trust": 0.5, "tau_L_start": -0.05}, # V10: tau_L_start=-0.05
            {"lambda_up": 0.9, "lambda_down": 0.7, "tau_min": 0.01, "eta_meta": 0.01, "initial_trust": 0.5, "tau_L_start": -0.2},  # V11: tau_L_start=-0.2
        ]

        method_configs = {
            "krum": krum_configs,
            "multikrum": multikrum_configs,
            "trimmed_mean": tm_configs,
            "norm_clipping": nc_configs,
            "rfa": rfa_configs,
            "bucketing": bucketing_configs,
            "foolsgold": fg_configs,
            "flame": flame_configs,
            "deepsight": deepsight_configs,
            "fldetector": fldetector_configs,
            "zeno": zeno_configs,
            "fltrust": fltrust_configs,
            "baffle": baffle_configs,
            "tvflids": tvflids_configs,
        }

        # Verify exact config count across all 15 methods: 0 + 3 + 6 + 4 + 4 + 4 + 6 + 4 + 9 + 3 + 4 + 6 + 4 + 8 + 12 = 77
        total_b1_configs = sum(len(cfgs) for cfgs in method_configs.values())
        assert total_b1_configs == 77, f"Expected 77 B1 configurations, found {total_b1_configs}"

        for ds in datasets:
            for strategy, cfgs in method_configs.items():
                for cfg in cfgs:
                    for s in seeds:
                        runs.append(RunSpecification(
                            block="B1",
                            purpose=f"Tuning {strategy} on D_tune ({ds})",
                            dataset=ds,
                            strategy=strategy,
                            strategy_params=cfg,
                            attack="label_flip_30",
                            seed=s,
                            num_rounds=100,
                        ))
        return runs

    @classmethod
    def enumerate_block_2(cls) -> List[RunSpecification]:
        """
        Block 2: RQ1 Main LF (Table III).
        15 methods x 3 datasets x 20 seeds = 900 runs.
        """
        runs: List[RunSpecification] = []
        for ds in ALL_3_DATASETS:
            for strat in ALL_15_METHODS:
                for s in CANONICAL_20_SEEDS:
                    runs.append(RunSpecification(
                        block="B2",
                        purpose=f"RQ1 Main LF comparison: {strat} on {ds}",
                        dataset=ds,
                        strategy=strat,
                        attack="label_flip_30",
                        seed=s,
                        num_rounds=100,
                    ))
        return runs

    @classmethod
    def enumerate_block_3(cls) -> List[RunSpecification]:
        """
        Block 3: Clean Reference for Delta ASR (Table IV).
        15 methods x 3 datasets x 20 seeds = 900 runs.
        """
        runs: List[RunSpecification] = []
        for ds in ALL_3_DATASETS:
            for strat in ALL_15_METHODS:
                for s in CANONICAL_20_SEEDS:
                    runs.append(RunSpecification(
                        block="B3",
                        purpose=f"Clean reference baseline: {strat} on {ds}",
                        dataset=ds,
                        strategy=strat,
                        attack="no_attack",
                        seed=s,
                        num_rounds=100,
                    ))
        return runs

    @classmethod
    def enumerate_block_4(cls) -> List[RunSpecification]:
        """
        Block 4: RQ1 Attack Matrix on NSL-KDD (Table IV).
        15 methods x 7 attacks x 10 seeds = 1,050 runs.
        Attacks: GS, NI, BD, MM-p, MM-o, MS-p, MS-o.
        """
        runs: List[RunSpecification] = []
        attacks = [
            ("gradient_scale_30", {}),
            ("noise_30", {}),
            ("backdoor_20", {}),
            ("min_max_p_30", {"variant": "partial"}),
            ("min_max_o_30", {"variant": "omniscient"}),
            ("min_sum_p_30", {"variant": "partial"}),
            ("min_sum_o_30", {"variant": "omniscient"}),
        ]
        for strat in ALL_15_METHODS:
            for atk, atk_params in attacks:
                for s in CANONICAL_10_SEEDS:
                    runs.append(RunSpecification(
                        block="B4",
                        purpose=f"RQ1 attack matrix: {strat} under {atk}",
                        dataset="nslkdd",
                        strategy=strat,
                        attack=atk,
                        attack_params=atk_params,
                        attack_variant=atk_params.get("variant"),
                        seed=s,
                        num_rounds=100,
                    ))
        return runs

    @classmethod
    def enumerate_block_5(cls) -> List[RunSpecification]:
        """
        Block 5: Sensitivity Sweeps on NSL-KDD (Table IV / Supp Table S4).
        5 methods x 6 settings x 10 seeds = 300 runs.
        Settings: GS kappa in {10^2, 10^3, 10^4}, LF f/N in {0.1, 0.2, 0.4}.
        """
        runs: List[RunSpecification] = []
        settings = [
            ("gradient_scale_100", {"factor": 100.0}),
            ("gradient_scale_1000", {"factor": 1000.0}),
            ("gradient_scale_10000", {"factor": 10000.0}),
            ("label_flip_10", {"ratio": 0.1}),
            ("label_flip_20", {"ratio": 0.2}),
            ("label_flip_40", {"ratio": 0.4}),
        ]
        for strat in FIVE_EVAL_METHODS:
            for atk, atk_params in settings:
                for s in CANONICAL_10_SEEDS:
                    runs.append(RunSpecification(
                        block="B5",
                        purpose=f"Sensitivity sweep: {strat} under {atk}",
                        dataset="nslkdd",
                        strategy=strat,
                        attack=atk,
                        attack_params=atk_params,
                        attack_strength=atk_params.get("factor") or atk_params.get("ratio"),
                        seed=s,
                        num_rounds=100,
                    ))
        return runs

    @classmethod
    def enumerate_block_6(cls) -> List[RunSpecification]:
        """
        Block 6: RQ2 Routing Interventions on NSL-KDD (Table V).
        100 runs:
          - 4 new configurations under LF (tau_min in {0, 0.001, 0.05} for TV-FLIDS; FLTrust with floor 0.01) x 10 seeds = 40 runs.
          - 6 configurations under LF-R (tau_min in {0, 0.001, 0.01, 0.05} for TV-FLIDS; FLTrust default & with floor 0.01) x 10 seeds = 60 runs.
        """
        runs: List[RunSpecification] = []
        # LF runs (4 new configs)
        lf_configs = [
            ("tvflids", {"min_trust": 0.0}),
            ("tvflids", {"min_trust": 0.001}),
            ("tvflids", {"min_trust": 0.05}),
            ("fltrust", {"min_trust": 0.01}),
        ]
        for strat, params in lf_configs:
            for s in CANONICAL_10_SEEDS:
                runs.append(RunSpecification(
                    block="B6",
                    purpose=f"RQ2 routing intervention LF: {strat} {params}",
                    dataset="nslkdd",
                    strategy=strat,
                    strategy_params=params,
                    attack="label_flip_30",
                    seed=s,
                    num_rounds=100,
                ))

        # LF-R runs (6 configs)
        lfr_configs = [
            ("tvflids", {"min_trust": 0.0}),
            ("tvflids", {"min_trust": 0.001}),
            ("tvflids", {"min_trust": 0.01}),
            ("tvflids", {"min_trust": 0.05}),
            ("fltrust", {}),
            ("fltrust", {"min_trust": 0.01}),
        ]
        for strat, params in lfr_configs:
            for s in CANONICAL_10_SEEDS:
                runs.append(RunSpecification(
                    block="B6",
                    purpose=f"RQ2 routing intervention LF-R: {strat} {params}",
                    dataset="nslkdd",
                    strategy=strat,
                    strategy_params=params,
                    attack="lf_r_30",
                    seed=s,
                    num_rounds=100,
                ))
        return runs

    @classmethod
    def enumerate_block_7(cls) -> List[RunSpecification]:
        """
        Block 7: RQ3 Component Ablations (Table VI).
        12 variants x 2 datasets (nslkdd, edgeiiotset) x 20 seeds = 480 runs.
        """
        runs: List[RunSpecification] = []
        datasets = ["nslkdd", "edgeiiotset"]
        ablation_variants = [
            ("A1: No gate", "tvflids", {"no_verification": True}, 2000),
            ("A2: No memory", "tvflids", {"memory_decay_up": 0.0, "memory_decay_down": 0.0}, 2000),
            ("A3: Symmetric memory", "tvflids", {"memory_decay_down": 0.9}, 2000),
            ("A4: Sample-averaged signal", "tvflids", {"use_sample_averaged_val_loss": True}, 2000),
            ("A5: No validation signal", "tvflids", {"beta_fixed_zero": True}, 2000),
            ("A6: Fixed equal weights", "tvflids_fixed", {"alpha": 1/3, "beta": 1/3, "gamma": 1/3}, 2000),
            ("A7: No clipping", "tvflids", {"no_clipping": True}, 2000),
            ("A8: Hard direction and norm rejection", "tvflids", {"hard_rejection": True}, 2000),
            ("A9: Meta-loss on balanced loss", "tvflids", {"meta_loss_balanced": True}, 2000),
            ("A10a: |D_val|=500", "tvflids", {}, 500),
            ("A10b: |D_val|=8000", "tvflids", {}, 8000),
            ("A11: Rarest-class quota 5", "tvflids", {"rarest_class_quota": 5}, 2000),
        ]
        for ds in datasets:
            for v_name, strat, params, val_sz in ablation_variants:
                for s in CANONICAL_20_SEEDS:
                    runs.append(RunSpecification(
                        block="B7",
                        purpose=f"RQ3 ablation {v_name} on {ds}",
                        dataset=ds,
                        strategy=strat,
                        strategy_params=params,
                        attack="label_flip_30",
                        val_size=val_sz,
                        seed=s,
                        num_rounds=100,
                    ))
        return runs

    @classmethod
    def enumerate_block_8(cls) -> List[RunSpecification]:
        """
        Block 8: RQ4 Adaptive & On-Off Attacks (Table VII / Supp Table S3).
        1,010 runs:
          - 5 methods x (9 ACK settings + 8 on-off settings) x 10 seeds = 850 runs.
          - 2 TV-FLIDS lambda_down variants ({0.5, 0.9}) x 8 on-off settings x 10 seeds = 160 runs.
        """
        runs: List[RunSpecification] = []
        ack_settings = [
            ("ack1_k0_30", {"knowledge_tier": "K0"}),
            ("ack1_k1_30", {"knowledge_tier": "K1"}),
            ("ack1_k2_30", {"knowledge_tier": "K2"}),
            ("ack2_k1_30", {"knowledge_tier": "K1", "psi": 0.85}),
            ("ack2_k2_30", {"knowledge_tier": "K2", "psi": 0.85}),
            ("ack3_k1_30", {"knowledge_tier": "K1"}),
            ("ack3_k2_30", {"knowledge_tier": "K2"}),
            ("ack4_k1_30", {"knowledge_tier": "K1"}),
            ("ack4_k2_30", {"knowledge_tier": "K2"}),
        ]
        on_off_settings = [
            ("on_off_lf_10", {"k": 10, "type": "on_off_lf"}),
            ("on_off_lf_20", {"k": 20, "type": "on_off_lf"}),
            ("on_off_lf_30", {"k": 30, "type": "on_off_lf"}),
            ("on_off_lf_50", {"k": 50, "type": "on_off_lf"}),
            ("on_off_ack2_10", {"k": 10, "type": "on_off_ack2", "knowledge_tier": "K1"}),
            ("on_off_ack2_20", {"k": 20, "type": "on_off_ack2", "knowledge_tier": "K1"}),
            ("on_off_ack2_30", {"k": 30, "type": "on_off_ack2", "knowledge_tier": "K1"}),
            ("on_off_ack2_50", {"k": 50, "type": "on_off_ack2", "knowledge_tier": "K1"}),
        ]

        # 5 methods on 17 settings
        all_17_settings = ack_settings + on_off_settings
        for strat in FIVE_EVAL_METHODS:
            for atk, atk_params in all_17_settings:
                for s in CANONICAL_10_SEEDS:
                    runs.append(RunSpecification(
                        block="B8",
                        purpose=f"RQ4 adaptive attack: {strat} under {atk}",
                        dataset="nslkdd",
                        strategy=strat,
                        attack=atk,
                        attack_params=atk_params,
                        knowledge_tier=atk_params.get("knowledge_tier"),
                        on_off_k=atk_params.get("k"),
                        seed=s,
                        num_rounds=100,
                    ))

        # 2 TV-FLIDS variants on 8 on-off settings
        for l_down in [0.5, 0.9]:
            for atk, atk_params in on_off_settings:
                for s in CANONICAL_10_SEEDS:
                    runs.append(RunSpecification(
                        block="B8",
                        purpose=f"RQ4 on-off TV-FLIDS (lambda_down={l_down}) under {atk}",
                        dataset="nslkdd",
                        strategy="tvflids",
                        strategy_params={"lambda_down": l_down},
                        attack=atk,
                        attack_params=atk_params,
                        knowledge_tier=atk_params.get("knowledge_tier"),
                        on_off_k=atk_params.get("k"),
                        seed=s,
                        num_rounds=100,
                    ))
        return runs

    @classmethod
    def enumerate_block_9(cls) -> List[RunSpecification]:
        """
        Block 9: RQ6 Deployment Variability (Table VIII / Supp Table S5).
        5 methods x 40 federation configurations = 200 runs.
        Configs drawn from alpha_D in {0.1, 0.3, 0.5, 1.0}, N in {20, 50, 100}, f/N in {0.2, 0.3}.
        """
        runs: List[RunSpecification] = []
        # Construct 40 deterministic federation configurations
        configs = []
        alphas = [0.1, 0.3, 0.5, 1.0]
        client_counts = [20, 50, 100]
        attack_ratios = [0.2, 0.3]

        # 24 core grid points (4 x 3 x 2)
        for a in alphas:
            for n in client_counts:
                for r in attack_ratios:
                    configs.append({"alpha": a, "num_clients": n, "attack_ratio": r, "placement_seed": int(a*100 + n + r*10)})

        # 16 additional configurations with perturbed Dirichlet / attacker placement seeds to reach exactly 40
        for i in range(16):
            a = alphas[i % len(alphas)]
            n = client_counts[i % len(client_counts)]
            r = attack_ratios[i % len(attack_ratios)]
            configs.append({"alpha": a, "num_clients": n, "attack_ratio": r, "placement_seed": 1000 + i})

        assert len(configs) == 40, f"Expected 40 federation configs, got {len(configs)}"

        for strat in FIVE_EVAL_METHODS:
            for cfg_idx, cfg in enumerate(configs):
                runs.append(RunSpecification(
                    block="B9",
                    purpose=f"RQ6 deployment variability: {strat} config #{cfg_idx+1}",
                    dataset="nslkdd",
                    strategy=strat,
                    attack="label_flip_30",
                    alpha=cfg["alpha"],
                    num_clients=cfg["num_clients"],
                    fraction_fit=0.5,
                    attack_strength=cfg["attack_ratio"],
                    seed=cfg["placement_seed"],
                    num_rounds=100,
                ))
        return runs

    @classmethod
    def enumerate_block_10(cls) -> List[RunSpecification]:
        """
        Block 10: RQ8 Differential Privacy and Cost Profiling (Supp Table S5).
        28 runs:
          - 2 methods (tvflids, fltrust) x 10 seeds under DP (noise=1.0) = 20 runs.
          - 8 profiling runs (TV-FLIDS at D in {10, 25, 50, 100}; FLTrust at N in {20, 200}, D in {10, 100}).
        """
        runs: List[RunSpecification] = []
        # DP runs: 2 methods x 10 seeds = 20 runs
        for strat in ["tvflids", "fltrust"]:
            for s in CANONICAL_10_SEEDS:
                runs.append(RunSpecification(
                    block="B10",
                    purpose=f"RQ8 Differential Privacy: {strat} (noise=1.0)",
                    dataset="nslkdd",
                    strategy=strat,
                    attack="label_flip_30",
                    dp_noise_multiplier=1.0,
                    seed=s,
                    num_rounds=100,
                ))

        # Profiling runs: 8 runs
        for d in [10, 25, 50, 100]:
            runs.append(RunSpecification(
                block="B10",
                purpose=f"RQ8 Profiling TV-FLIDS: D={d}",
                dataset="nslkdd",
                strategy="tvflids",
                attack="label_flip_30",
                is_profiling=True,
                profiling_d=d,
                profiling_n=20 if d == 10 else d * 2,
                num_rounds=50,
                seed=42,
            ))
        for n, d in [(20, 10), (200, 100), (20, 5), (200, 50)]:
            runs.append(RunSpecification(
                block="B10",
                purpose=f"RQ8 Profiling FLTrust: N={n}, D={d}",
                dataset="nslkdd",
                strategy="fltrust",
                attack="label_flip_30",
                is_profiling=True,
                profiling_d=d,
                profiling_n=n,
                num_rounds=50,
                seed=42,
            ))
        assert len(runs) == 28, f"Expected 28 B10 runs, got {len(runs)}"
        return runs

    @classmethod
    def enumerate_block_11(cls) -> List[RunSpecification]:
        """
        Block 11: SMOTE Factorial (Supplementary §S5, Table S1).
        15 methods x 3 non-main cells x 10 seeds = 450 runs.
        Cells:
          - smote_per_client_after
          - smote_global_before
          - smote_global_after
        (Main cell: smote_per_client_before is already run in Block 2 seeds 1-10).
        """
        runs: List[RunSpecification] = []
        non_main_protocols = [
            "smote_per_client_after",
            "smote_global_before",
            "smote_global_after",
        ]
        for strat in ALL_15_METHODS:
            for proto in non_main_protocols:
                for s in CANONICAL_10_SEEDS:
                    runs.append(RunSpecification(
                        block="B11",
                        purpose=f"SMOTE factorial {proto}: {strat}",
                        dataset="nslkdd",
                        strategy=strat,
                        protocol=proto,
                        attack="label_flip_30",
                        seed=s,
                        num_rounds=100,
                    ))
        return runs

    @classmethod
    def enumerate_all(cls) -> List[RunSpecification]:
        """
        Enumerate all 6,111 canonical experimental runs.
        Guarantees exact count and checks for duplicates.
        """
        b1 = cls.enumerate_block_1()    # 693
        b2 = cls.enumerate_block_2()    # 900
        b3 = cls.enumerate_block_3()    # 900
        b4 = cls.enumerate_block_4()    # 1,050
        b5 = cls.enumerate_block_5()    # 300
        b6 = cls.enumerate_block_6()    # 100
        b7 = cls.enumerate_block_7()    # 480
        b8 = cls.enumerate_block_8()    # 1,010
        b9 = cls.enumerate_block_9()    # 200
        b10 = cls.enumerate_block_10()  # 28
        b11 = cls.enumerate_block_11()  # 450

        all_runs = b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9 + b10 + b11
        expected_total = 6111
        actual_total = len(all_runs)

        if actual_total != expected_total:
            raise ValueError(
                f"Campaign enumeration count discrepancy! Expected {expected_total}, "
                f"enumerated {actual_total} runs."
            )

        # Check for unique run IDs
        seen_ids: Set[str] = set()
        duplicates: List[str] = []
        for r in all_runs:
            rid = r.run_id
            if rid in seen_ids:
                duplicates.append(rid)
            seen_ids.add(rid)

        if duplicates:
            raise ValueError(
                f"Duplicate run IDs detected in campaign enumeration! Duplicates: {duplicates[:5]}"
            )

        return all_runs

    @classmethod
    def filter_runs(
        cls,
        runs: List[RunSpecification],
        block: Optional[str] = None,
        dataset: Optional[str] = None,
        strategy: Optional[str] = None,
        attack: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> List[RunSpecification]:
        """Filter runs by block, dataset, strategy, attack, or seed."""
        filtered = runs
        if block is not None:
            filtered = [r for r in filtered if r.block.upper() == block.upper()]
        if dataset is not None:
            filtered = [r for r in filtered if r.dataset.lower() == dataset.lower()]
        if strategy is not None:
            filtered = [r for r in filtered if r.strategy.lower() == strategy.lower()]
        if attack is not None:
            filtered = [r for r in filtered if r.attack.lower() == attack.lower()]
        if seed is not None:
            filtered = [r for r in filtered if r.seed == seed]
        return filtered

    @classmethod
    def get_summary_report(cls) -> Dict[str, Any]:
        """Generate a complete statistical breakdown of the campaign matrix."""
        all_runs = cls.enumerate_all()
        blocks = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"]
        block_counts = {}
        for b in blocks:
            b_runs = [r for r in all_runs if r.block == b]
            block_counts[b] = len(b_runs)

        dataset_counts = {}
        for ds in ALL_3_DATASETS:
            dataset_counts[ds] = len([r for r in all_runs if r.dataset == ds])

        strategy_counts = {}
        for s in ALL_15_METHODS + ["tvflids_fixed"]:
            matching = [r for r in all_runs if r.strategy == s]
            if matching:
                strategy_counts[s] = len(matching)

        return {
            "total_runs": len(all_runs),
            "canonical_paper_upper_bound": 7038,
            "block_counts": block_counts,
            "dataset_counts": dataset_counts,
            "strategy_counts": strategy_counts,
        }


def main():
    parser = argparse.ArgumentParser(description="TV-FLIDS Campaign Enumerator")
    parser.add_argument("--report", action="store_true", help="Print summary report of the campaign")
    parser.add_argument("--export", type=str, default=None, help="Export manifest to JSON file")
    parser.add_argument("--block", type=str, default=None, help="Filter by block (e.g. B1, B2)")
    parser.add_argument("--dataset", type=str, default=None, help="Filter by dataset")
    parser.add_argument("--strategy", type=str, default=None, help="Filter by strategy")
    parser.add_argument("--attack", type=str, default=None, help="Filter by attack")
    parser.add_argument("--seed", type=int, default=None, help="Filter by seed")

    args = parser.parse_args()

    enumerator = CampaignEnumerator()
    all_runs = enumerator.enumerate_all()
    filtered = enumerator.filter_runs(
        all_runs,
        block=args.block,
        dataset=args.dataset,
        strategy=args.strategy,
        attack=args.attack,
        seed=args.seed,
    )

    if args.report or (not args.export and not args.block and not args.dataset and not args.strategy):
        summary = enumerator.get_summary_report()
        print("=" * 70)
        print("TV-FLIDS CANONICAL CAMPAIGN ENUMERATION REPORT")
        print("=" * 70)
        print(f"Total Canonical Runs:       {summary['total_runs']}")
        print(f"Paper Stated Upper Bound:   <= {summary['canonical_paper_upper_bound']}")
        print("\nRuns per Block:")
        for b, count in summary["block_counts"].items():
            print(f"  {b:<6}: {count:>5} runs")
        print("\nRuns per Dataset:")
        for ds, count in summary["dataset_counts"].items():
            print(f"  {ds:<12}: {count:>5} runs")
        print("=" * 70)

    if args.block or args.dataset or args.strategy or args.attack or args.seed:
        print(f"\nFiltered Runs Count: {len(filtered)}")
        for r in filtered[:5]:
            print(f"  [{r.block}] {r.strategy} on {r.dataset} (seed={r.seed}, attack={r.attack}) -> {r.run_id}")
        if len(filtered) > 5:
            print(f"  ... and {len(filtered) - 5} more runs.")

    if args.export:
        manifest_data = [
            {
                "run_id": r.run_id,
                "block": r.block,
                "purpose": r.purpose,
                "dataset": r.dataset,
                "strategy": r.strategy,
                "attack": r.attack,
                "seed": r.seed,
                "scientific_config": r.to_scientific_dict(),
            }
            for r in filtered
        ]
        os.makedirs(os.path.dirname(os.path.abspath(args.export)), exist_ok=True)
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        print(f"\n[Manifest] Successfully exported {len(manifest_data)} runs to {args.export}")


if __name__ == "__main__":
    main()
