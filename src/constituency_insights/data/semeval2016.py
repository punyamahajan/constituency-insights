"""Local-only ingestion and validation for SemEval-2016 Task 6, Task A.

The official files are not bundled with this project.  Their parsing details must
be supplied explicitly *after* inspecting the manually downloaded files; this
module intentionally does not guess filenames, delimiters, encodings, headers,
or source column names.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Mapping, Sequence

import pandas as pd


CANONICAL_REQUIRED_COLUMNS = ("text", "target", "label")
SUPPORTED_LABELS = frozenset({"FAVOR", "AGAINST", "NONE"})
TASK_A_TARGETS = frozenset(
    {
        "Atheism",
        "Climate Change is a Real Concern",
        "Feminist Movement",
        "Hillary Clinton",
        "Legalization of Abortion",
    }
)


class StanceDataError(ValueError):
    """Raised when stance data cannot be parsed or fails validation."""


@dataclass(frozen=True)
class TaskAFileSpec:
    """Explicit parsing instructions for one inspected official data file.

    ``columns`` maps canonical names (``text``, ``target``, ``label``, and
    optionally ``source_id``) to the corresponding source column names.
    ``read_csv_options`` is passed to :func:`pandas.read_csv`; it must explicitly
    contain ``sep`` and ``encoding``. Header behavior must also be explicit via
    ``header`` or ``names``. ``include_targets`` is an opt-in source-row filter;
    without it, unexpected targets reach validation and raise an error.

    ``allow_c1_controls`` must be explicitly enabled for a losslessly decoded
    file known to contain bytes that map to Unicode C1 control characters. The
    characters are retained unchanged and their row positions are reported.
    """

    path: Path
    split: str
    columns: Mapping[str, str]
    read_csv_options: Mapping[str, object]
    include_targets: Collection[str] | None = None
    allow_c1_controls: bool = False


def official_task_a_file_specs(
    raw_dir: Path = Path("data/raw/stance/semeval2016_task6"),
) -> tuple[TaskAFileSpec, TaskAFileSpec]:
    """Return the verified, explicit configuration for the local official files."""

    columns = {"text": "Tweet", "target": "Target", "label": "Stance"}
    common_options: dict[str, object] = {
        "sep": ",",
        "header": 0,
        "dtype": str,
        "keep_default_na": False,
        "quotechar": '"',
        "doublequote": True,
        "lineterminator": "\r",
    }
    train = TaskAFileSpec(
        path=raw_dir / "train.csv",
        split="train",
        columns=columns,
        read_csv_options={**common_options, "encoding": "iso-8859-1"},
        allow_c1_controls=True,
    )
    test = TaskAFileSpec(
        path=raw_dir / "test.csv",
        split="test",
        columns=columns,
        read_csv_options={**common_options, "encoding": "utf-8"},
        include_targets=TASK_A_TARGETS,
    )
    return train, test


def load_task_a(
    files: Sequence[TaskAFileSpec],
    *,
    expected_targets: Collection[str],
) -> pd.DataFrame:
    """Load inspected Task A files and return validated canonical records.

    The returned columns are ``text``, ``target``, ``label``, ``split``, plus
    ``source_id`` when that field is mapped for at least one input file.
    Original files are read only and are never modified or exported.
    """

    if not files:
        raise StanceDataError(
            "No Task A files were configured. Manually download the official "
            "files into data/raw/stance/semeval2016_task6/, inspect them, and "
            "create a TaskAFileSpec for each official split."
        )
    if not expected_targets:
        raise StanceDataError(
            "expected_targets is empty. Confirm the exact Task A target strings "
            "from the supplied official files before loading them."
        )

    loaded = [_read_file(spec) for spec in files]
    frames = [frame for frame, _, _ in loaded]
    include_source_id = any("source_id" in frame.columns for frame in frames)
    if include_source_id:
        frames = [
            frame.assign(source_id=pd.NA)
            if "source_id" not in frame.columns
            else frame
            for frame in frames
        ]

    data = pd.concat(frames, ignore_index=True)
    canonical = [*CANONICAL_REQUIRED_COLUMNS, "split"]
    if include_source_id:
        canonical.append("source_id")
    data = data.loc[:, canonical]
    validate_task_a(data, expected_targets=expected_targets)
    data.attrs["target_filter_report"] = [
        report for _, report, _ in loaded if report is not None
    ]
    data.attrs["encoding_anomalies"] = [
        report for _, _, report in loaded if report is not None
    ]
    return data


def validate_task_a(
    data: pd.DataFrame,
    *,
    expected_targets: Collection[str],
) -> None:
    """Validate normalized Task A data, raising actionable errors on failure."""

    required = {*CANONICAL_REQUIRED_COLUMNS, "split"}
    missing_columns = sorted(required.difference(data.columns))
    if missing_columns:
        raise StanceDataError(
            f"Normalized data is missing required columns: {missing_columns}."
        )
    if data.empty:
        raise StanceDataError("Normalized Task A data contains no records.")

    for column in required:
        missing = data[column].isna()
        if missing.any():
            raise StanceDataError(
                f"Column {column!r} has {int(missing.sum())} missing value(s); "
                f"first affected row index: {missing[missing].index[0]}."
            )

    for column in ("text", "target", "label", "split"):
        non_strings = ~data[column].map(lambda value: isinstance(value, str))
        if non_strings.any():
            raise StanceDataError(
                f"Column {column!r} contains a non-string value at row "
                f"{non_strings[non_strings].index[0]}. Check the source schema."
            )
        empty = data[column].str.strip().eq("")
        if empty.any():
            raise StanceDataError(
                f"Column {column!r} has {int(empty.sum())} empty value(s); "
                f"first affected row index: {empty[empty].index[0]}."
            )

    labels = set(data["label"])
    unexpected_labels = sorted(labels.difference(SUPPORTED_LABELS))
    if unexpected_labels:
        raise StanceDataError(
            f"Unsupported stance label(s): {unexpected_labels}. Expected only "
            f"{sorted(SUPPORTED_LABELS)}; verify the label column mapping."
        )

    targets = set(data["target"])
    unexpected_targets = sorted(targets.difference(expected_targets))
    if unexpected_targets:
        raise StanceDataError(
            f"Unexpected Task A target(s): {unexpected_targets}. Confirm exact "
            "target spelling in the official files and expected_targets."
        )

    duplicate_subset = ["text", "target", "label"]
    duplicates = data.duplicated(subset=duplicate_subset, keep=False)
    if duplicates.any():
        first = duplicates[duplicates].index[0]
        raise StanceDataError(
            f"Duplicate normalized record(s) found; first affected row index: "
            f"{first}. Check for repeated files, duplicate source rows, or "
            "records crossing split boundaries."
        )

    if "source_id" in data.columns:
        present_ids = data.loc[data["source_id"].notna()].copy()
        present_ids["source_id"] = present_ids["source_id"].astype(str).str.strip()
        empty_ids = present_ids["source_id"].eq("")
        if empty_ids.any():
            raise StanceDataError("source_id contains an empty value.")
        split_counts = present_ids.groupby("source_id")["split"].nunique()
        crossing = split_counts[split_counts > 1]
        if not crossing.empty:
            raise StanceDataError(
                "A source_id occurs in multiple splits; first conflicting ID: "
                f"{crossing.index[0]!r}. Preserve official split boundaries."
            )


def _read_file(
    spec: TaskAFileSpec,
) -> tuple[pd.DataFrame, dict[str, object] | None, dict[str, object] | None]:
    path = Path(spec.path)
    if not path.is_file():
        raise StanceDataError(
            f"Configured Task A file does not exist: {path}. Download it from "
            "the official SemEval page and place it under "
            "data/raw/stance/semeval2016_task6/ without renaming or editing it."
        )

    missing_mappings = [
        column for column in CANONICAL_REQUIRED_COLUMNS if column not in spec.columns
    ]
    if missing_mappings:
        raise StanceDataError(
            f"Schema mapping for {path} lacks canonical field(s): "
            f"{missing_mappings}. Inspect the official file before configuring it."
        )
    unknown_mappings = sorted(
        set(spec.columns).difference({*CANONICAL_REQUIRED_COLUMNS, "source_id"})
    )
    if unknown_mappings:
        raise StanceDataError(
            f"Schema mapping for {path} has unsupported canonical field(s): "
            f"{unknown_mappings}."
        )

    options = dict(spec.read_csv_options)
    missing_options = [key for key in ("sep", "encoding") if key not in options]
    if "header" not in options and "names" not in options:
        missing_options.append("header or names")
    if missing_options:
        raise StanceDataError(
            f"Parsing options for {path} must explicitly specify "
            f"{', '.join(missing_options)} after inspecting the official file."
        )

    try:
        source = pd.read_csv(path, **options)
    except Exception as exc:
        raise StanceDataError(
            f"Could not parse {path} with the configured schema: {exc}. "
            "Re-check its encoding, delimiter, quoting, and header behavior."
        ) from exc

    missing_source_columns = sorted(set(spec.columns.values()).difference(source.columns))
    if missing_source_columns:
        raise StanceDataError(
            f"Unexpected schema in {path}; mapped source column(s) not found: "
            f"{missing_source_columns}. Available columns: {list(source.columns)}."
        )

    rename = {source_name: canonical for canonical, source_name in spec.columns.items()}
    normalized = source.loc[:, list(rename)].rename(columns=rename).copy()
    normalized["split"] = spec.split

    normalized, target_filter_report = _apply_target_filter(normalized, spec)
    encoding_report = _check_c1_controls(normalized, spec)
    return normalized, target_filter_report, encoding_report


def _apply_target_filter(
    normalized: pd.DataFrame, spec: TaskAFileSpec
) -> tuple[pd.DataFrame, dict[str, object] | None]:
    if spec.include_targets is None:
        return normalized, None

    target = normalized["target"]
    if target.isna().any() or not target.map(lambda value: isinstance(value, str)).all():
        raise StanceDataError(
            f"Cannot apply target filter to {spec.path}: target contains a "
            "missing or non-string value. Check the source schema first."
        )
    if target.str.strip().eq("").any():
        raise StanceDataError(
            f"Cannot apply target filter to {spec.path}: target contains an "
            "empty value."
        )

    included = set(spec.include_targets)
    keep = target.isin(included)
    excluded_counts = target.loc[~keep].value_counts().sort_index().to_dict()
    excluded_rows = int((~keep).sum())
    normalized = normalized.loc[keep].reset_index(drop=True)
    report = {
        "path": str(spec.path),
        "split": spec.split,
        "excluded_rows": excluded_rows,
        "excluded_targets": excluded_counts,
        "reason": "Target was not in the explicitly configured Task A target set.",
    }
    return normalized, report


def _check_c1_controls(
    normalized: pd.DataFrame, spec: TaskAFileSpec
) -> dict[str, object] | None:
    affected = normalized["text"].map(
        lambda value: isinstance(value, str)
        and any("\u0080" <= character <= "\u009f" for character in value)
    )
    if not affected.any():
        return None

    row_indices = affected[affected].index.tolist()
    if not spec.allow_c1_controls:
        raise StanceDataError(
            f"Decoded text in {spec.path} contains C1 control characters at "
            f"data row index/indices {row_indices}. No characters were replaced. "
            "Verify the source encoding or explicitly set allow_c1_controls=True "
            "to retain and report them unchanged."
        )
    return {
        "path": str(spec.path),
        "split": spec.split,
        "row_indices": row_indices,
        "count": len(row_indices),
        "limitation": (
            "The bytes were decoded one-to-one with ISO-8859-1 and retained, "
            "but their intended characters are not established."
        ),
    }
