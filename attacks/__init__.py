"""
attacks/__init__.py
Adversarial attacks and threat models for TV-FLIDS.
"""

from attacks.knowledge import KnowledgeTier, ValidationEstimateProvider, NSLKDD_VAL_QUOTAS
from attacks.adversarial import (
    AdversarialAttackFactory,
    ATTACK_CONFIGS,
    apply_round_attacks,
    apply_min_max_attack_to_params,
    apply_min_sum_attack_to_params,
    apply_ack2_attack_to_params,
    get_malicious_client_ids,
    min_max_attack,
    min_sum_attack,
    label_flip,
    label_flip_random,
    gradient_scale,
    noise_injection,
    backdoor_attack,
    ack2_generate_updates,
    ack4_generate_updates,
    is_on_off_active,
)

__all__ = [
    "KnowledgeTier",
    "ValidationEstimateProvider",
    "NSLKDD_VAL_QUOTAS",
    "AdversarialAttackFactory",
    "ATTACK_CONFIGS",
    "apply_round_attacks",
    "apply_min_max_attack_to_params",
    "apply_min_sum_attack_to_params",
    "apply_ack2_attack_to_params",
    "get_malicious_client_ids",
    "min_max_attack",
    "min_sum_attack",
    "label_flip",
    "label_flip_random",
    "gradient_scale",
    "noise_injection",
    "backdoor_attack",
    "ack2_generate_updates",
    "ack4_generate_updates",
    "is_on_off_active",
]
