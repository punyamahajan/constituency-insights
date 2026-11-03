import pandas as pd
import pytest

from constituency_insights.data import (
    StanceDataError,
    prepare_task_a_validation_split,
)


TARGETS = {"Policy Alpha", "Policy Beta"}
LABELS = {"FAVOR", "AGAINST", "NONE"}


def _dataset(rows_per_stratum: int = 10) -> pd.DataFrame:
    rows = []
    for target in sorted(TARGETS):
        for label in sorted(LABELS):
            for number in range(rows_per_stratum):
                rows.append(
                    {
                        "text": f"Synthetic {target} {label} statement {number}",
                        "target": target,
                        "label": label,
                        "split": "train",
                    }
                )
    rows.extend(
        {
            "text": f"Official test sentinel {number}",
            "target": "Policy Alpha",
            "label": "NONE",
            "split": "test",
        }
        for number in range(3)
    )
    data = pd.DataFrame(rows)
    data.attrs["encoding_anomalies"] = [
        {"split": "train", "row_indices": [0, 17]}
    ]
    return data


def _split(data: pd.DataFrame):
    return prepare_task_a_validation_split(
        data,
        expected_train_size=60,
        expected_targets=TARGETS,
        expected_labels=LABELS,
    )


def test_split_is_reproducible_and_has_expected_sizes() -> None:
    data = _dataset()

    first = _split(data)
    second = _split(data)

    assert first.development_train.index.tolist() == second.development_train.index.tolist()
    assert first.validation.index.tolist() == second.validation.index.tolist()
    assert len(first.development_train) == 48
    assert len(first.validation) == 12
    assert first.report.random_seed == 42
    assert first.report.validation_size == 0.20


def test_every_eligible_row_is_assigned_once_without_overlap_or_test_leakage() -> None:
    data = _dataset()
    result = _split(data)
    development = set(result.development_train.index)
    validation = set(result.validation.index)
    eligible = set(data.index[data["split"].eq("train")])
    official_test = set(data.index[data["split"].eq("test")])

    assert not development & validation
    assert development | validation == eligible
    assert not (development | validation) & official_test
    assert result.development_train["split"].eq("train").all()
    assert result.validation["split"].eq("train").all()
    assert set(result.development_train["partition"]) == {"development_train"}
    assert set(result.validation["partition"]) == {"validation"}


def test_all_expected_combinations_are_reported_and_preserved() -> None:
    result = _split(_dataset())

    assert len(result.report.stratum_counts) == 6
    assert set(result.report.stratum_counts.values()) == {10}
    assert set(result.report.development_train_stratum_counts.values()) == {8}
    assert set(result.report.validation_stratum_counts.values()) == {2}
    assert result.report.missing_combinations == ()


def test_singleton_stratum_fails_without_fallback_or_dropping() -> None:
    data = pd.DataFrame(
        [
            {
                "text": "Only example",
                "target": "Policy Alpha",
                "label": "FAVOR",
                "split": "train",
            }
        ]
    )

    with pytest.raises(StanceDataError, match="No rows were dropped"):
        prepare_task_a_validation_split(
            data,
            expected_train_size=1,
            expected_targets={"Policy Alpha"},
            expected_labels={"FAVOR"},
        )


def test_missing_combination_is_reported_not_fabricated() -> None:
    rows = [
        {
            "text": f"Synthetic {target} {label} {number}",
            "target": target,
            "label": label,
            "split": "train",
        }
        for target, label in (
            ("Policy Alpha", "FAVOR"),
            ("Policy Alpha", "NONE"),
            ("Policy Beta", "FAVOR"),
        )
        for number in range(4)
    ]

    result = prepare_task_a_validation_split(
        pd.DataFrame(rows),
        expected_train_size=12,
        expected_targets=TARGETS,
        expected_labels={"FAVOR", "NONE"},
        validation_size=0.25,
    )

    assert result.report.missing_combinations == ("Policy Beta || NONE",)
    assert result.report.stratum_counts["Policy Beta || NONE"] == 0
    assert "no source records" in result.report.limitations[0]


def test_wrong_official_training_count_fails_before_split() -> None:
    with pytest.raises(StanceDataError, match="partial or expanded dataset"):
        prepare_task_a_validation_split(
            _dataset(),
            expected_train_size=59,
            expected_targets=TARGETS,
            expected_labels=LABELS,
        )


def test_encoding_anomalies_are_retained_and_assigned() -> None:
    data = _dataset()
    result = _split(data)

    assert set(result.report.encoding_anomaly_assignments) == {0, 17}
    assert all(
        assignment in {"development_train", "validation"}
        for assignment in result.report.encoding_anomaly_assignments.values()
    )
    for index in (0, 17):
        partition = (
            result.development_train
            if result.report.encoding_anomaly_assignments[index] == "development_train"
            else result.validation
        )
        assert partition.loc[index, "text"] == data.loc[index, "text"]
