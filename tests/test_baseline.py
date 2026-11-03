import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from constituency_insights.data import prepare_task_a_validation_split
from constituency_insights.nlp.baseline import (
    LABEL_ORDER,
    build_baseline_pipeline,
    fit_and_evaluate_baseline,
    write_local_outputs,
)


TARGETS = {"Policy Alpha", "Policy Beta"}
LABELS = set(LABEL_ORDER)


def _split():
    label_words = {"AGAINST": "oppose", "FAVOR": "support", "NONE": "unclear"}
    rows = [
        {
            "text": f"{label_words[label]} synthetic statement number {number}",
            "target": target,
            "label": label,
            "split": "train",
        }
        for target in sorted(TARGETS)
        for label in LABEL_ORDER
        for number in range(15)
    ]
    return prepare_task_a_validation_split(
        pd.DataFrame(rows),
        expected_train_size=90,
        expected_targets=TARGETS,
        expected_labels=LABELS,
    )


def test_pipeline_has_fixed_initial_configuration() -> None:
    pipeline = build_baseline_pipeline()
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]

    assert isinstance(vectorizer, TfidfVectorizer)
    assert vectorizer.lowercase is True
    assert vectorizer.analyzer == "word"
    assert vectorizer.ngram_range == (1, 2)
    assert isinstance(classifier, LogisticRegression)
    assert classifier.max_iter == 1000
    assert classifier.random_state == 42
    assert classifier.class_weight == "balanced"


def test_vectorizer_is_fit_on_development_training_only() -> None:
    split = _split()
    first_validation_index = split.validation.index[0]
    split.validation.loc[first_validation_index, "text"] = "validationonlytoken"

    result = fit_and_evaluate_baseline(split)
    vocabulary = result.pipeline.named_steps["tfidf"].vocabulary_

    assert "validationonlytoken" not in vocabulary


def test_metrics_use_explicit_label_order_and_expected_shapes() -> None:
    result = fit_and_evaluate_baseline(_split())
    metrics = result.metrics

    assert metrics["label_order"] == list(LABEL_ORDER)
    assert list(metrics["overall"]["per_class"]) == list(LABEL_ORDER)
    assert len(metrics["overall"]["confusion_matrix"]) == 3
    assert all(len(row) == 3 for row in metrics["overall"]["confusion_matrix"])
    assert sum(
        sum(row) for row in metrics["overall"]["confusion_matrix"]
    ) == metrics["validation_rows"]
    assert set(metrics["by_target"]) == TARGETS


def test_official_test_rows_are_rejected() -> None:
    split = _split()
    split.validation.loc[split.validation.index[0], "split"] = "test"

    with pytest.raises(ValueError, match="official test rows"):
        fit_and_evaluate_baseline(split)


def test_error_analysis_is_reproducible_and_has_required_columns() -> None:
    first = fit_and_evaluate_baseline(_split()).error_analysis
    second = fit_and_evaluate_baseline(_split()).error_analysis

    assert list(first.columns) == [
        "source_row_index",
        "target",
        "true_label",
        "predicted_label",
        "text_excerpt",
    ]
    pd.testing.assert_frame_equal(first, second)


def test_local_outputs_separate_aggregate_metrics_from_text(tmp_path: Path) -> None:
    result = fit_and_evaluate_baseline(_split())

    write_local_outputs(result, tmp_path)

    metrics_path = tmp_path / "validation_metrics.json"
    errors_path = tmp_path / "validation_error_analysis.csv"
    report_path = tmp_path / "validation_error_report.json"
    assert metrics_path.is_file()
    assert errors_path.is_file()
    assert report_path.is_file()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert metrics["label_order"] == list(LABEL_ORDER)
    assert "text" not in metrics_path.read_text(encoding="utf-8")
    assert report["scope"] == "derived_validation_only"
    assert "text_excerpt" not in report
