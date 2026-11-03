"""Reproducible in-memory partitions for the SemEval Task A baseline."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Collection

import pandas as pd
from sklearn.model_selection import train_test_split

from .semeval2016 import SUPPORTED_LABELS, TASK_A_TARGETS, StanceDataError


OFFICIAL_TASK_A_TRAIN_SIZE = 2_914
DEFAULT_RANDOM_SEED = 42
DEFAULT_VALIDATION_SIZE = 0.20


@dataclass(frozen=True)
class ValidationSplitReport:
    """Aggregate provenance and validation details; contains no tweet text."""

    random_seed: int
    validation_size: float
    eligible_rows: int
    development_train_rows: int
    validation_rows: int
    stratum_counts: dict[str, int]
    development_train_stratum_counts: dict[str, int]
    validation_stratum_counts: dict[str, int]
    missing_combinations: tuple[str, ...]
    small_strata: dict[str, int]
    encoding_anomaly_assignments: dict[int, str]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class ValidationSplit:
    """In-memory development partitions derived from official training rows."""

    development_train: pd.DataFrame
    validation: pd.DataFrame
    report: ValidationSplitReport


def prepare_task_a_validation_split(
    data: pd.DataFrame,
    *,
    expected_train_size: int = OFFICIAL_TASK_A_TRAIN_SIZE,
    expected_targets: Collection[str] = TASK_A_TARGETS,
    expected_labels: Collection[str] = SUPPORTED_LABELS,
    validation_size: float = DEFAULT_VALIDATION_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
    small_stratum_threshold: int = 20,
) -> ValidationSplit:
    """Create a stratified 80/20 split from official training rows only.

    The original ``split`` column remains ``train`` in both outputs to preserve
    provenance. A new ``partition`` column identifies ``development_train`` and
    ``validation``. Nothing is written to disk.
    """

    required = {"text", "target", "label", "split"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise StanceDataError(f"Cannot split data; missing columns: {missing}.")
    if not data.index.is_unique:
        raise StanceDataError(
            "Cannot verify complete row assignment because input indices are not unique."
        )
    if not 0 < validation_size < 1:
        raise StanceDataError("validation_size must be strictly between 0 and 1.")

    unexpected_splits = sorted(set(data["split"]).difference({"train", "test"}))
    if unexpected_splits:
        raise StanceDataError(
            f"Unexpected official split value(s): {unexpected_splits}. Refusing "
            "to infer which rows are eligible."
        )

    eligible = data.loc[data["split"].eq("train")].copy()
    if len(eligible) != expected_train_size:
        raise StanceDataError(
            f"Expected {expected_train_size} official Task A training rows, found "
            f"{len(eligible)}. Refusing to split a partial or expanded dataset."
        )

    expected_targets_set = set(expected_targets)
    expected_labels_set = set(expected_labels)
    unexpected_targets = sorted(set(eligible["target"]).difference(expected_targets_set))
    unexpected_labels = sorted(set(eligible["label"]).difference(expected_labels_set))
    if unexpected_targets or unexpected_labels:
        raise StanceDataError(
            "Cannot stratify unexpected values: "
            f"targets={unexpected_targets}, labels={unexpected_labels}."
        )

    stratum = eligible["target"] + " || " + eligible["label"]
    counts = stratum.value_counts().sort_index()
    all_combinations = sorted(
        f"{target} || {label}"
        for target in expected_targets_set
        for label in expected_labels_set
    )
    stratum_counts = {key: int(counts.get(key, 0)) for key in all_combinations}
    missing_combinations = tuple(
        key for key, count in stratum_counts.items() if count == 0
    )
    unsplittable = {key: count for key, count in stratum_counts.items() if count == 1}
    if unsplittable:
        raise StanceDataError(
            "Combined target-label strata need at least two rows for stratified "
            f"train/validation assignment; unsplittable strata: {unsplittable}. "
            "No rows were dropped and no unstratified fallback was used."
        )

    observed_strata = int((counts > 0).sum())
    validation_rows = ceil(len(eligible) * validation_size)
    development_rows = len(eligible) - validation_rows
    if validation_rows < observed_strata or development_rows < observed_strata:
        raise StanceDataError(
            "Requested partition sizes cannot contain every observed target-label "
            "stratum. No unstratified fallback was used."
        )

    development_index, validation_index = train_test_split(
        eligible.index.to_numpy(),
        test_size=validation_size,
        random_state=random_seed,
        shuffle=True,
        stratify=stratum.to_numpy(),
    )
    development_index_set = set(development_index.tolist())
    validation_index_set = set(validation_index.tolist())
    eligible_index_set = set(eligible.index.tolist())
    official_test_indices = set(data.index[data["split"].eq("test")].tolist())

    if development_index_set & validation_index_set:
        raise StanceDataError("Development and validation partitions overlap.")
    assigned = development_index_set | validation_index_set
    if assigned != eligible_index_set:
        raise StanceDataError(
            "Eligible rows were not assigned exactly once to the derived partitions."
        )
    if assigned & official_test_indices:
        raise StanceDataError("Official test rows leaked into a derived partition.")

    development = data.loc[development_index].copy()
    validation = data.loc[validation_index].copy()
    development["partition"] = "development_train"
    validation["partition"] = "validation"

    development_counts = _partition_counts(development, all_combinations)
    validation_counts = _partition_counts(validation, all_combinations)
    lost_observed = [
        key
        for key, count in stratum_counts.items()
        if count > 0
        and (development_counts[key] == 0 or validation_counts[key] == 0)
    ]
    if lost_observed:
        raise StanceDataError(
            "Stratified split failed to preserve observed combination(s) in both "
            f"partitions: {lost_observed}. No rows were dropped."
        )

    small_strata = {
        key: count
        for key, count in stratum_counts.items()
        if 0 < count <= small_stratum_threshold
    }
    anomaly_assignments = _encoding_anomaly_assignments(
        data, eligible_index_set, development_index_set, validation_index_set
    )
    limitations: list[str] = []
    if missing_combinations:
        limitations.append(
            "Some expected target-label combinations have no source records; "
            "none were fabricated or dropped."
        )
    if small_strata:
        limitations.append(
            "Small target-label strata yield unstable validation metrics."
        )
    if anomaly_assignments:
        limitations.append(
            "Encoding-anomalous records were retained unchanged and their "
            "partition assignments are reported."
        )

    report = ValidationSplitReport(
        random_seed=random_seed,
        validation_size=validation_size,
        eligible_rows=len(eligible),
        development_train_rows=len(development),
        validation_rows=len(validation),
        stratum_counts=stratum_counts,
        development_train_stratum_counts=development_counts,
        validation_stratum_counts=validation_counts,
        missing_combinations=missing_combinations,
        small_strata=small_strata,
        encoding_anomaly_assignments=anomaly_assignments,
        limitations=tuple(limitations),
    )
    return ValidationSplit(development, validation, report)


def _partition_counts(
    partition: pd.DataFrame, all_combinations: list[str]
) -> dict[str, int]:
    keys = partition["target"] + " || " + partition["label"]
    counts = keys.value_counts()
    return {key: int(counts.get(key, 0)) for key in all_combinations}


def _encoding_anomaly_assignments(
    data: pd.DataFrame,
    eligible_indices: set[object],
    development_indices: set[object],
    validation_indices: set[object],
) -> dict[int, str]:
    assignments: dict[int, str] = {}
    for anomaly in data.attrs.get("encoding_anomalies", []):
        if anomaly.get("split") != "train":
            continue
        for index in anomaly.get("row_indices", []):
            if index not in eligible_indices:
                raise StanceDataError(
                    f"Reported training encoding anomaly row {index} is not in "
                    "the eligible official training rows."
                )
            if index in development_indices:
                assignments[int(index)] = "development_train"
            elif index in validation_indices:
                assignments[int(index)] = "validation"
            else:
                raise StanceDataError(
                    f"Encoding anomaly row {index} was not assigned to a partition."
                )
    return assignments
