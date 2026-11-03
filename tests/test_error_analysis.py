import pandas as pd
import pytest

from constituency_insights.nlp.error_analysis import (
    build_validation_error_report,
    extract_directional_errors,
    privacy_reduced_excerpt,
)


def _validation() -> pd.DataFrame:
    labels = ["FAVOR", "AGAINST", "FAVOR", "AGAINST", "NONE", "NONE", "FAVOR"]
    return pd.DataFrame(
        {
            "text": [f"synthetic text {number}" for number in range(len(labels))],
            "target": ["Alpha", "Alpha", "Beta", "Beta", "Alpha", "Beta", "Alpha"],
            "label": labels,
            "split": "train",
            "partition": "validation",
        },
        index=[8, 3, 12, 5, 20, 15, 30],
    )


def test_report_counts_all_requested_error_directions_by_pair_and_target() -> None:
    predictions = ["AGAINST", "FAVOR", "NONE", "NONE", "FAVOR", "AGAINST", "FAVOR"]

    report, examples = build_validation_error_report(_validation(), predictions)

    assert report["total_errors"] == 6
    assert report["correct_predictions"] == 1
    assert report["error_counts_by_pair"] == {
        "AGAINST -> FAVOR": 1,
        "AGAINST -> NONE": 1,
        "FAVOR -> AGAINST": 1,
        "FAVOR -> NONE": 1,
        "NONE -> AGAINST": 1,
        "NONE -> FAVOR": 1,
    }
    assert report["error_counts_by_target"]["Alpha"]["total_errors"] == 3
    assert report["error_counts_by_target"]["Beta"]["total_errors"] == 3
    assert len(examples) == 6


def test_representative_examples_are_deterministic() -> None:
    predictions = ["AGAINST", "FAVOR", "NONE", "NONE", "FAVOR", "AGAINST", "FAVOR"]

    first = build_validation_error_report(_validation(), predictions)
    second = build_validation_error_report(_validation(), predictions)

    assert first[0] == second[0]
    pd.testing.assert_frame_equal(first[1], second[1])


def test_excerpt_redacts_common_identifiers_and_is_bounded() -> None:
    text = "@person email me at name@example.org or +91 98765 43210 https://example.org private tail"

    excerpt = privacy_reduced_excerpt(text, max_length=60)

    assert "@person" not in excerpt
    assert "name@example.org" not in excerpt
    assert "98765" not in excerpt
    assert "example.org" not in excerpt
    assert "[USER]" in excerpt
    assert len(excerpt) <= 60


def test_error_analysis_rejects_official_test_rows() -> None:
    validation = _validation()
    validation.loc[validation.index[0], "split"] = "test"

    with pytest.raises(ValueError, match="never official test rows"):
        build_validation_error_report(validation, ["NONE"] * len(validation))


def test_error_analysis_validates_shape_and_parameters() -> None:
    with pytest.raises(ValueError, match="Prediction count"):
        build_validation_error_report(_validation(), ["NONE"])
    with pytest.raises(ValueError, match="must be positive"):
        build_validation_error_report(
            _validation(), ["NONE"] * len(_validation()), excerpt_length=0
        )


def test_extract_directional_errors_from_confusion_matrix() -> None:
    # 3x3 confusion matrix matching AGAINST, FAVOR, NONE
    cm = [
        [205, 45, 30],
        [51, 72, 27],
        [58, 29, 66],
    ]
    errors = extract_directional_errors(cm, ("AGAINST", "FAVOR", "NONE"))
    assert errors == {
        "AGAINST -> FAVOR": 45,
        "AGAINST -> NONE": 30,
        "FAVOR -> AGAINST": 51,
        "FAVOR -> NONE": 27,
        "NONE -> AGAINST": 58,
        "NONE -> FAVOR": 29,
    }


def test_extract_directional_errors_validates_dimensions() -> None:
    with pytest.raises(ValueError, match="row count"):
        extract_directional_errors([[1, 2]], ("AGAINST", "FAVOR", "NONE"))
    with pytest.raises(ValueError, match="column count"):
        extract_directional_errors([[1, 2], [3, 4], [5, 6]], ("AGAINST", "FAVOR", "NONE"))


def test_error_analysis_zero_errors_on_perfect_predictions() -> None:
    val = _validation()
    perfect_preds = list(val["label"])
    report, reps = build_validation_error_report(val, perfect_preds)

    assert report["total_errors"] == 0
    assert report["correct_predictions"] == len(val)
    assert report["error_counts_by_pair"] == {}
    assert len(reps) == 0

