"""Dataset ingestion and validation utilities."""
"""Dataset ingestion utilities."""

from .semeval2016 import (
    SUPPORTED_LABELS,
    TASK_A_TARGETS,
    StanceDataError,
    TaskAFileSpec,
    load_task_a,
    official_task_a_file_specs,
    validate_task_a,
)
from .splitting import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_VALIDATION_SIZE,
    OFFICIAL_TASK_A_TRAIN_SIZE,
    ValidationSplit,
    ValidationSplitReport,
    prepare_task_a_validation_split,
)

__all__ = [
    "SUPPORTED_LABELS",
    "TASK_A_TARGETS",
    "StanceDataError",
    "TaskAFileSpec",
    "load_task_a",
    "official_task_a_file_specs",
    "validate_task_a",
    "DEFAULT_RANDOM_SEED",
    "DEFAULT_VALIDATION_SIZE",
    "OFFICIAL_TASK_A_TRAIN_SIZE",
    "ValidationSplit",
    "ValidationSplitReport",
    "prepare_task_a_validation_split",
]
