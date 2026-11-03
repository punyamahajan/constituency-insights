"""Leakage-safe TF-IDF + Logistic Regression stance baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

from constituency_insights.data import (
    TASK_A_TARGETS,
    ValidationSplit,
    load_task_a,
    official_task_a_file_specs,
    prepare_task_a_validation_split,
)
from .error_analysis import build_validation_error_report


LABEL_ORDER = ("AGAINST", "FAVOR", "NONE")
RANDOM_SEED = 42


@dataclass(frozen=True)
class BaselineRun:
    """Fitted pipeline and validation-only outputs."""

    pipeline: Pipeline
    metrics: dict[str, Any]
    error_report: dict[str, Any]
    error_analysis: pd.DataFrame


def build_baseline_pipeline() -> Pipeline:
    """Construct the fixed initial baseline; settings are not tuned optima."""

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="word",
                    ngram_range=(1, 2),
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_SEED,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def fit_and_evaluate_baseline(
    split: ValidationSplit,
    *,
    label_order: Sequence[str] = LABEL_ORDER,
    small_support_threshold: int = 10,
) -> BaselineRun:
    """Fit on development-training only and evaluate on validation only."""

    development = split.development_train
    validation = split.validation
    _validate_partitions(development, validation)

    labels = list(label_order)
    pipeline = build_baseline_pipeline()
    pipeline.fit(development["text"], development["label"])
    predictions = pipeline.predict(validation["text"])

    majority_label = str(development["label"].value_counts().idxmax())
    majority_predictions = [majority_label] * len(validation)
    metrics: dict[str, Any] = {
        "experiment": "tfidf_word_1_2_logistic_regression_balanced",
        "setting_note": (
            "class_weight='balanced' is a fixed initial baseline setting, not a "
            "tuned optimum."
        ),
        "label_order": labels,
        "development_train_rows": len(development),
        "validation_rows": len(validation),
        "overall": _classification_metrics(
            validation["label"], predictions, labels
        ),
        "majority_baseline": {
            "label": majority_label,
            "accuracy": float(
                accuracy_score(validation["label"], majority_predictions)
            ),
            "macro_f1": float(
                f1_score(
                    validation["label"],
                    majority_predictions,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            ),
        },
        "by_target": {},
        "split_report": asdict(split.report),
    }

    for target in sorted(validation["target"].unique()):
        mask = validation["target"].eq(target)
        target_metrics = _classification_metrics(
            validation.loc[mask, "label"], predictions[mask.to_numpy()], labels
        )
        small_labels = {
            label: int(target_metrics["per_class"][label]["support"])
            for label in labels
            if int(target_metrics["per_class"][label]["support"])
            < small_support_threshold
        }
        target_metrics["small_support_warning"] = (
            {
                "threshold": small_support_threshold,
                "labels": small_labels,
                "message": "Per-class metrics with small support are unstable.",
            }
            if small_labels
            else None
        )
        metrics["by_target"][target] = target_metrics

    error_report, errors = build_validation_error_report(validation, predictions)

    return BaselineRun(
        pipeline=pipeline,
        metrics=metrics,
        error_report=error_report,
        error_analysis=errors,
    )


def run_official_validation_baseline(
    *,
    raw_dir: Path = Path("data/raw/stance/semeval2016_task6"),
    output_dir: Path = Path("experiments/stance_tfidf_logreg"),
) -> BaselineRun:
    """Run validation baseline without loading or accessing official test data."""

    train_spec, _ = official_task_a_file_specs(raw_dir)
    data = load_task_a([train_spec], expected_targets=TASK_A_TARGETS)
    split = prepare_task_a_validation_split(data)
    result = fit_and_evaluate_baseline(split)
    write_local_outputs(result, output_dir)
    return result


def write_local_outputs(result: BaselineRun, output_dir: Path) -> None:
    """Write ignored local outputs; aggregate JSON contains no tweet text."""

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "validation_metrics.json"
    metrics_path.write_text(
        json.dumps(result.metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result.error_analysis.to_csv(
        output_dir / "validation_error_analysis.csv", index=False, encoding="utf-8"
    )
    (output_dir / "validation_error_report.json").write_text(
        json.dumps(result.error_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _classification_metrics(
    truth: pd.Series, predictions: Sequence[str], labels: list[str]
) -> dict[str, Any]:
    report = classification_report(
        truth,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        label: {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }
        for label in labels
    }
    return {
        "rows": len(truth),
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(
            f1_score(
                truth,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            truth, predictions, labels=labels
        ).astype(int).tolist(),
    }


def _validate_partitions(
    development: pd.DataFrame, validation: pd.DataFrame
) -> None:
    if set(development.index) & set(validation.index):
        raise ValueError("Development-training and validation partitions overlap.")
    for name, frame, expected_partition in (
        ("development-training", development, "development_train"),
        ("validation", validation, "validation"),
    ):
        required = {"text", "target", "label", "split", "partition"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} partition is missing columns: {missing}.")
        if not frame["split"].eq("train").all():
            raise ValueError(
                f"{name} contains official test rows; refusing model evaluation."
            )
        if not frame["partition"].eq(expected_partition).all():
            raise ValueError(f"Unexpected partition marker in {name}.")
