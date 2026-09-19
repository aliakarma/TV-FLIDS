"""
campaign/__init__.py
TV-FLIDS Experimental Campaign Orchestration & Reproducibility Infrastructure.
"""

from campaign.run_spec import RunSpecification, compute_run_id
from campaign.enumerator import (
    CampaignEnumerator,
    CANONICAL_20_SEEDS,
    CANONICAL_10_SEEDS,
    CANONICAL_3_SEEDS,
    ALL_15_METHODS,
    FIVE_EVAL_METHODS,
    ALL_3_DATASETS,
)
from campaign.config_freezer import freeze_configuration, resolve_model_configuration
from campaign.artifacts import (
    get_artifact_directory,
    write_status,
    read_status,
    write_metrics,
    read_metrics,
    validate_result_schema,
    VALID_STATUSES,
)
from campaign.runner import CampaignRunner, check_dataset_availability

__all__ = [
    "RunSpecification",
    "compute_run_id",
    "CampaignEnumerator",
    "freeze_configuration",
    "resolve_model_configuration",
    "get_artifact_directory",
    "write_status",
    "read_status",
    "write_metrics",
    "read_metrics",
    "validate_result_schema",
    "VALID_STATUSES",
    "CampaignRunner",
    "check_dataset_availability",
    "CANONICAL_20_SEEDS",
    "CANONICAL_10_SEEDS",
    "CANONICAL_3_SEEDS",
    "ALL_15_METHODS",
    "FIVE_EVAL_METHODS",
    "ALL_3_DATASETS",
]
