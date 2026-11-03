"""Deterministic, privacy-reduced validation error analysis."""

from __future__ import annotations

import re
from typing import Any, Sequence

import pandas as pd


DEFAULT_EXCERPT_LENGTH = 160
DEFAULT_EXAMPLES_PER_PAIR = 2

_URL = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_EMAIL = re.compile(r"(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b")
_MENTION = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d .()-]{6,}\d)(?!\w)")
_WHITESPACE = re.compile(r"\s+")


def build_validation_error_report(
    validation: pd.DataFrame,
    predictions: Sequence[str],
    *,
    excerpt_length: int = DEFAULT_EXCERPT_LENGTH,
    examples_per_pair: int = DEFAULT_EXAMPLES_PER_PAIR,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return aggregate error counts and deterministic representative excerpts."""

    required = {"text", "target", "label", "split", "partition"}
    missing = sorted(required.difference(validation.columns))
    if missing:
        raise ValueError(f"Validation error analysis is missing columns: {missing}.")
    if not validation["split"].eq("train").all() or not validation[
        "partition"
    ].eq("validation").all():
        raise ValueError(
            "Error analysis accepts only the derived validation partition, "
            "never official test rows."
        )
    if len(predictions) != len(validation):
        raise ValueError("Prediction count does not match validation row count.")
    if excerpt_length < 1 or examples_per_pair < 1:
        raise ValueError("Excerpt length and examples per pair must be positive.")

    errors = validation.loc[:, ["text", "target", "label"]].copy()
    errors["predicted_label"] = list(predictions)
    errors = errors.loc[errors["label"] != errors["predicted_label"]].copy()
    errors["pair"] = errors["label"] + " -> " + errors["predicted_label"]

    pair_counts = {
        str(pair): int(count)
        for pair, count in errors["pair"].value_counts().sort_index().items()
    }
    target_counts: dict[str, dict[str, Any]] = {}
    for target, group in errors.groupby("target", sort=True):
        target_counts[str(target)] = {
            "total_errors": len(group),
            "by_pair": {
                str(pair): int(count)
                for pair, count in group["pair"].value_counts().sort_index().items()
            },
        }

    representative_rows: list[dict[str, Any]] = []
    for pair, group in errors.groupby("pair", sort=True):
        for index, row in group.sort_index().head(examples_per_pair).iterrows():
            representative_rows.append(
                {
                    "source_row_index": int(index),
                    "target": row["target"],
                    "true_label": row["label"],
                    "predicted_label": row["predicted_label"],
                    "text_excerpt": privacy_reduced_excerpt(
                        row["text"], max_length=excerpt_length
                    ),
                }
            )

    representatives = pd.DataFrame(
        representative_rows,
        columns=[
            "source_row_index",
            "target",
            "true_label",
            "predicted_label",
            "text_excerpt",
        ],
    )
    report = {
        "scope": "derived_validation_only",
        "validation_rows": len(validation),
        "total_errors": len(errors),
        "correct_predictions": len(validation) - len(errors),
        "error_counts_by_pair": pair_counts,
        "error_counts_by_target": target_counts,
        "representative_example_count": len(representatives),
        "excerpt_policy": {
            "max_characters": excerpt_length,
            "examples_per_error_pair": examples_per_pair,
            "redactions": ["user mentions", "URLs", "email addresses", "phone-like numbers"],
        },
    }
    return report, representatives


def privacy_reduced_excerpt(text: str, *, max_length: int) -> str:
    """Redact common identifiers and return a bounded one-line excerpt."""

    excerpt = _URL.sub("[URL]", text)
    excerpt = _EMAIL.sub("[EMAIL]", excerpt)
    excerpt = _MENTION.sub("[USER]", excerpt)
    excerpt = _PHONE.sub("[PHONE]", excerpt)
    excerpt = _WHITESPACE.sub(" ", excerpt).strip()
    if len(excerpt) <= max_length:
        return excerpt
    if max_length == 1:
        return "…"
    return excerpt[: max_length - 1].rstrip() + "…"


def extract_directional_errors(
    confusion_matrix: Sequence[Sequence[int]],
    label_order: Sequence[str] = ("AGAINST", "FAVOR", "NONE"),
) -> dict[str, int]:
    """Extract directional off-diagonal error pair counts from a confusion matrix."""
    if len(confusion_matrix) != len(label_order):
        raise ValueError("Confusion matrix row count does not match label order length.")
    errors: dict[str, int] = {}
    for i, true_label in enumerate(label_order):
        if len(confusion_matrix[i]) != len(label_order):
            raise ValueError(f"Row {i} column count does not match label order length.")
        for j, pred_label in enumerate(label_order):
            if i != j:
                pair = f"{true_label} -> {pred_label}"
                errors[pair] = int(confusion_matrix[i][j])
    return dict(sorted(errors.items()))

