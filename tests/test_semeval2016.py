from pathlib import Path

import pandas as pd
import pytest

from constituency_insights.data.semeval2016 import (
    TASK_A_TARGETS,
    StanceDataError,
    TaskAFileSpec,
    load_task_a,
    official_task_a_file_specs,
    validate_task_a,
)


TARGETS = {"Policy Alpha", "Policy Beta"}


def _write_fixture(path: Path, rows: list[str]) -> None:
    path.write_text("id|topic|body|stance\n" + "\n".join(rows), encoding="utf-8")


def _spec(
    path: Path,
    split: str = "train",
    *,
    include_targets: set[str] | None = None,
    allow_c1_controls: bool = False,
) -> TaskAFileSpec:
    return TaskAFileSpec(
        path=path,
        split=split,
        columns={
            "source_id": "id",
            "target": "topic",
            "text": "body",
            "label": "stance",
        },
        read_csv_options={"sep": "|", "encoding": "utf-8", "header": 0},
        include_targets=include_targets,
        allow_c1_controls=allow_c1_controls,
    )


def test_load_task_a_normalizes_explicitly_configured_schema(tmp_path: Path) -> None:
    raw = tmp_path / "synthetic.txt"
    _write_fixture(
        raw,
        [
            "1|Policy Alpha|I support this proposal.|FAVOR",
            "2|Policy Beta|The text gives no position.|NONE",
        ],
    )

    result = load_task_a([_spec(raw)], expected_targets=TARGETS)

    assert list(result.columns) == ["text", "target", "label", "split", "source_id"]
    assert result.to_dict("records")[0] == {
        "text": "I support this proposal.",
        "target": "Policy Alpha",
        "label": "FAVOR",
        "split": "train",
        "source_id": 1,
    }


def test_missing_file_error_explains_manual_action(tmp_path: Path) -> None:
    with pytest.raises(StanceDataError, match="official SemEval page"):
        load_task_a([_spec(tmp_path / "missing.txt")], expected_targets=TARGETS)


def test_schema_mismatch_lists_available_columns(tmp_path: Path) -> None:
    raw = tmp_path / "synthetic.txt"
    _write_fixture(raw, ["1|Policy Alpha|Synthetic statement.|FAVOR"])
    bad_spec = TaskAFileSpec(
        path=raw,
        split="train",
        columns={"target": "topic", "text": "tweet", "label": "stance"},
        read_csv_options={"sep": "|", "encoding": "utf-8", "header": 0},
    )

    with pytest.raises(StanceDataError, match="Available columns"):
        load_task_a([bad_spec], expected_targets=TARGETS)


def test_explicit_target_filter_reports_excluded_rows(tmp_path: Path) -> None:
    raw = tmp_path / "combined.txt"
    _write_fixture(
        raw,
        [
            "1|Policy Alpha|Task A statement.|FAVOR",
            "2|Separate Task|Other task statement.|AGAINST",
        ],
    )

    result = load_task_a(
        [_spec(raw, split="test", include_targets=TARGETS)],
        expected_targets=TARGETS,
    )

    assert len(result) == 1
    assert result.attrs["target_filter_report"] == [
        {
            "path": str(raw),
            "split": "test",
            "excluded_rows": 1,
            "excluded_targets": {"Separate Task": 1},
            "reason": "Target was not in the explicitly configured Task A target set.",
        }
    ]


def test_unexpected_target_without_filter_still_raises(tmp_path: Path) -> None:
    raw = tmp_path / "unexpected.txt"
    _write_fixture(raw, ["1|Separate Task|Synthetic statement.|FAVOR"])

    with pytest.raises(StanceDataError, match="Unexpected Task A target"):
        load_task_a([_spec(raw)], expected_targets=TARGETS)


def test_target_filter_does_not_hide_missing_target(tmp_path: Path) -> None:
    raw = tmp_path / "missing-target.txt"
    _write_fixture(raw, ["1||Synthetic statement.|FAVOR"])

    with pytest.raises(StanceDataError, match="target contains a missing"):
        load_task_a(
            [_spec(raw, include_targets=TARGETS)], expected_targets=TARGETS
        )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("text", "   ", "empty value"),
        ("label", "MAYBE", "Unsupported stance label"),
        ("target", "Unknown Policy", "Unexpected Task A target"),
    ],
)
def test_validation_rejects_invalid_values(
    column: str, value: str, message: str
) -> None:
    record = {
        "text": "Synthetic statement.",
        "target": "Policy Alpha",
        "label": "AGAINST",
        "split": "train",
    }
    record[column] = value

    with pytest.raises(StanceDataError, match=message):
        validate_task_a(pd.DataFrame([record]), expected_targets=TARGETS)


def test_validation_rejects_duplicate_records() -> None:
    record = {
        "text": "Synthetic statement.",
        "target": "Policy Alpha",
        "label": "FAVOR",
        "split": "train",
    }

    with pytest.raises(StanceDataError, match="Duplicate normalized record"):
        validate_task_a(pd.DataFrame([record, record]), expected_targets=TARGETS)


def test_validation_rejects_duplicate_records_across_splits() -> None:
    records = [
        {
            "text": "Synthetic statement.",
            "target": "Policy Alpha",
            "label": "FAVOR",
            "split": split,
        }
        for split in ("train", "test")
    ]

    with pytest.raises(StanceDataError, match="crossing split boundaries"):
        validate_task_a(pd.DataFrame(records), expected_targets=TARGETS)


def test_validation_rejects_missing_required_value() -> None:
    data = pd.DataFrame(
        [
            {
                "text": pd.NA,
                "target": "Policy Alpha",
                "label": "FAVOR",
                "split": "train",
            }
        ]
    )

    with pytest.raises(StanceDataError, match="missing value"):
        validate_task_a(data, expected_targets=TARGETS)


def test_validation_rejects_source_id_crossing_splits() -> None:
    data = pd.DataFrame(
        [
            {
                "text": "First synthetic statement.",
                "target": "Policy Alpha",
                "label": "FAVOR",
                "split": "train",
                "source_id": "same-id",
            },
            {
                "text": "Second synthetic statement.",
                "target": "Policy Beta",
                "label": "NONE",
                "split": "test",
                "source_id": "same-id",
            },
        ]
    )

    with pytest.raises(StanceDataError, match="multiple splits"):
        validate_task_a(data, expected_targets=TARGETS)


def test_c1_control_character_requires_explicit_opt_in(tmp_path: Path) -> None:
    raw = tmp_path / "encoding.txt"
    raw.write_bytes(
        b"id|topic|body|stance\n1|Policy Alpha|Unclear byte: \x90|FAVOR"
    )
    spec = _spec(raw)
    spec = TaskAFileSpec(
        path=spec.path,
        split=spec.split,
        columns=spec.columns,
        read_csv_options={**spec.read_csv_options, "encoding": "iso-8859-1"},
    )

    with pytest.raises(StanceDataError, match="No characters were replaced"):
        load_task_a([spec], expected_targets=TARGETS)


def test_c1_control_character_is_retained_and_reported(tmp_path: Path) -> None:
    raw = tmp_path / "encoding.txt"
    raw.write_bytes(
        b"id|topic|body|stance\n1|Policy Alpha|Unclear byte: \x90|FAVOR"
    )
    base_spec = _spec(raw)
    spec = TaskAFileSpec(
        path=base_spec.path,
        split=base_spec.split,
        columns=base_spec.columns,
        read_csv_options={**base_spec.read_csv_options, "encoding": "iso-8859-1"},
        allow_c1_controls=True,
    )

    result = load_task_a([spec], expected_targets=TARGETS)

    assert "\x90" in result.loc[0, "text"]
    assert result.attrs["encoding_anomalies"][0]["row_indices"] == [0]
    assert result.attrs["encoding_anomalies"][0]["count"] == 1


def test_official_specs_encode_only_verified_schema(tmp_path: Path) -> None:
    train, test = official_task_a_file_specs(tmp_path)

    assert train.path == tmp_path / "train.csv"
    assert train.read_csv_options["encoding"] == "iso-8859-1"
    assert train.include_targets is None
    assert train.allow_c1_controls is True
    assert test.path == tmp_path / "test.csv"
    assert test.read_csv_options["encoding"] == "utf-8"
    assert set(test.include_targets or ()) == set(TASK_A_TARGETS)
    assert train.columns == {
        "text": "Tweet",
        "target": "Target",
        "label": "Stance",
    }
    assert train.read_csv_options["sep"] == ","
    assert train.read_csv_options["lineterminator"] == "\r"
