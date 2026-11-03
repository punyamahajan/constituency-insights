from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from constituency_insights.nlp.rigorous_error_analysis import (
    calibration_metrics,
    cross_fitted_temperature_scaling,
    fit_temperature,
    heuristic_tags,
    metrics_with_bootstrap,
    softmax,
    tag_error_rates,
)


def test_bootstrap_metrics_are_seeded_and_have_valid_intervals() -> None:
    truth = np.array([0, 0, 1, 1, 2, 2] * 4)
    predictions = np.array([0, 1, 1, 1, 2, 0] * 4)

    first = metrics_with_bootstrap(
        truth, predictions, label_names=("A", "B", "C"),
        resamples=100, confidence_level=0.95, seed=42,
    )
    second = metrics_with_bootstrap(
        truth, predictions, label_names=("A", "B", "C"),
        resamples=100, confidence_level=0.95, seed=42,
    )

    assert first == second
    for label in first.values():
        for metric in label.values():
            assert 0 <= metric["lower"] <= metric["upper"] <= 1


def test_calibration_metrics_match_simple_hand_computation() -> None:
    probabilities = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])
    labels = np.array([0, 1])

    result = calibration_metrics(probabilities, labels, bins=2)

    assert result["ece"] == pytest.approx(0.25)
    expected_brier = ((0.2**2 + 0.1**2 + 0.1**2) + (0.2**2 + 0.3**2 + 0.1**2)) / 2
    assert result["brier_score"] == pytest.approx(expected_brier)


def test_temperature_scaling_and_cross_fitting_are_deterministic() -> None:
    labels = np.array([0, 1, 2] * 10)
    logits = np.eye(3)[labels] * 4.0 + 0.2

    temperature = fit_temperature(logits, labels)
    first_probabilities, first_temperatures = cross_fitted_temperature_scaling(
        logits, labels, folds=5, seed=42
    )
    second_probabilities, second_temperatures = cross_fitted_temperature_scaling(
        logits, labels, folds=5, seed=42
    )

    assert temperature > 0
    np.testing.assert_allclose(first_probabilities, second_probabilities)
    assert first_temperatures == second_temperatures
    np.testing.assert_allclose(first_probabilities.sum(axis=1), 1.0)


def test_heuristic_tags_and_rates_use_only_surface_features() -> None:
    texts = ["No policy? #topic @user", "brief", "ordinary longer statement here"]
    truth = np.array([0, 1, 2])
    predictions = np.array([1, 1, 2])

    tags = heuristic_tags(texts[0], very_short_threshold=2)
    rates, _ = tag_error_rates(
        texts, truth, predictions, very_short_threshold=2
    )

    assert set(tags) == {
        "contains_negation", "contains_hashtag", "contains_question_mark",
        "contains_mention",
    }
    assert rates["contains_hashtag"] == {"count": 1, "errors": 1, "error_rate": 1.0}
    assert rates["very_short"] == {"count": 1, "errors": 0, "error_rate": 0.0}


def test_error_report_and_tracked_outputs_do_not_contain_local_case_text() -> None:
    report_path = Path("docs/error_analysis.md")
    local_path = Path("outputs/local/error_cases.csv")
    if not report_path.is_file() or not local_path.is_file():
        pytest.skip("Generated analysis artifacts are not present.")

    local_cases = pd.read_csv(local_path, dtype=str)
    assert "text" in local_cases.columns
    report = report_path.read_text(encoding="utf-8")
    assert "| Text |" not in report
    assert "text_excerpt" not in report

    candidate_files = [Path("README.md"), *Path("docs").glob("*.md")]
    tracked_output_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in candidate_files if path.is_file()
    )
    leaked = [
        text for text in local_cases["text"]
        if isinstance(text, str) and text and text in tracked_output_text
    ]
    assert leaked == [], "Raw local error-case text leaked into a tracked report."
