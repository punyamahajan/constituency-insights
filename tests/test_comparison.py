"""Tests for paired validation prediction generation, McNemar testing, and model comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import torch

from constituency_insights.data import ValidationSplit, ValidationSplitReport
from constituency_insights.nlp.baseline import LABEL_ORDER
from constituency_insights.nlp.comparison import (
    McNemarResult,
    compare_validation_predictions,
    compute_mcnemar_test,
    generate_baseline_validation_predictions,
    generate_distilbert_validation_predictions,
    save_comparison_report,
    save_validation_predictions,
)


def _make_synthetic_split(n_dev: int = 12, n_val: int = 6) -> ValidationSplit:
    """Construct minimal synthetic ValidationSplit for fast unit testing."""
    targets = ["Hillary Clinton", "Legalization of Abortion"]
    labels = ["AGAINST", "FAVOR", "NONE"]

    dev_rows: list[dict[str, Any]] = []
    for i in range(n_dev):
        dev_rows.append(
            {
                "text": f"Development statement text number {i}",
                "target": targets[i % len(targets)],
                "label": labels[i % len(labels)],
                "split": "train",
                "partition": "development_train",
            }
        )
    dev_df = pd.DataFrame(dev_rows)

    val_rows: list[dict[str, Any]] = []
    for i in range(n_val):
        val_rows.append(
            {
                "text": f"Validation statement text number {i}",
                "target": targets[i % len(targets)],
                "label": labels[i % len(labels)],
                "split": "train",
                "partition": "validation",
            }
        )
    val_df = pd.DataFrame(val_rows, index=range(100, 100 + n_val))

    report = ValidationSplitReport(
        random_seed=42,
        validation_size=0.2,
        eligible_rows=n_dev + n_val,
        development_train_rows=n_dev,
        validation_rows=n_val,
        stratum_counts={},
        development_train_stratum_counts={},
        validation_stratum_counts={},
        missing_combinations=(),
        small_strata={},
        encoding_anomaly_assignments={},
        limitations=(),
    )
    return ValidationSplit(
        development_train=dev_df, validation=val_df, report=report
    )


class _MockDistilBertModel(torch.nn.Module):
    """Mock PyTorch classification model producing deterministic logits."""

    def __init__(self, num_labels: int = 3) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> Any:
        batch_size = input_ids.shape[0]
        # Alternate highest logit across classes
        logits = torch.zeros((batch_size, self.num_labels), dtype=torch.float32)
        for i in range(batch_size):
            logits[i, i % self.num_labels] = 5.0
        output = type("ModelOutput", (), {"logits": logits})()
        return output


class _MockTokenizer:
    """Mock tokenizer returning uniform PyTorch tensors."""

    def __call__(
        self,
        text: list[str],
        text_pair: list[str] | None = None,
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 128,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        n = len(text)
        return {
            "input_ids": torch.ones((n, 8), dtype=torch.long),
            "attention_mask": torch.ones((n, 8), dtype=torch.long),
        }


def test_mcnemar_test_equal_discordant_pairs() -> None:
    """Equal discordant pairs should yield 0.0 statistic and p-value 1.0."""
    baseline = [True, True, False, False]
    candidate = [True, False, True, False]
    result = compute_mcnemar_test(baseline, candidate)

    assert result.both_correct == 1
    assert result.baseline_only_correct == 1
    assert result.candidate_only_correct == 1
    assert result.both_incorrect == 1
    assert result.discordant_pairs == 2
    assert result.statistic_continuity_corrected == 0.0
    assert result.p_value_continuity_corrected == 1.0
    assert result.exact_binomial_p_value == 1.0


def test_mcnemar_test_known_analytical_values() -> None:
    """Test Edwards continuity correction against hand-calculated analytical value."""
    # 10 baseline-only vs 25 candidate-only: diff=15, diff-1=14, (14^2)/35 = 196/35 = 5.6
    baseline = [True] * 10 + [False] * 25
    candidate = [False] * 10 + [True] * 25
    result = compute_mcnemar_test(baseline, candidate)

    assert result.baseline_only_correct == 10
    assert result.candidate_only_correct == 25
    assert result.discordant_pairs == 35
    assert result.statistic_continuity_corrected == pytest.approx(5.6, rel=1e-5)
    assert result.statistic_uncorrected == pytest.approx((15**2) / 35, rel=1e-5)
    assert result.is_statistically_significant_05 is True


def test_mcnemar_test_zero_discordant() -> None:
    """Zero discordant pairs should return 0.0 statistic and 1.0 p-value."""
    baseline = [True, False, True]
    candidate = [True, False, True]
    result = compute_mcnemar_test(baseline, candidate)

    assert result.discordant_pairs == 0
    assert result.statistic_continuity_corrected == 0.0
    assert result.p_value_continuity_corrected == 1.0
    assert result.exact_binomial_p_value == 1.0


def test_mcnemar_test_validation_errors() -> None:
    """Unequal lengths or empty inputs should raise ValueError."""
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_mcnemar_test([True], [True, False])
    with pytest.raises(ValueError, match="empty"):
        compute_mcnemar_test([], [])


def test_generate_baseline_predictions_synthetic() -> None:
    """Test generating baseline predictions on synthetic split."""
    split = _make_synthetic_split(n_dev=9, n_val=6)
    records = generate_baseline_validation_predictions(split)

    assert len(records) == 6
    expected_indices = list(split.validation.index)
    for idx, record in zip(expected_indices, records):
        assert record["row_index"] == idx
        assert record["predicted_label"] in LABEL_ORDER
        assert set(record["probabilities"].keys()) == set(LABEL_ORDER)
        assert pytest.approx(sum(record["probabilities"].values()), abs=1e-3) == 1.0
        # Privacy check: no raw text
        assert "text" not in record
        assert "text_excerpt" not in record


def test_generate_distilbert_predictions_synthetic() -> None:
    """Test generating DistilBERT predictions with mock model and tokenizer."""
    split = _make_synthetic_split(n_dev=9, n_val=6)
    mock_model = _MockDistilBertModel()
    mock_tokenizer = _MockTokenizer()

    records = generate_distilbert_validation_predictions(
        split,
        model=mock_model,
        tokenizer=mock_tokenizer,
        device="cpu",
    )

    assert len(records) == 6
    expected_indices = list(split.validation.index)
    for idx, record in zip(expected_indices, records):
        assert record["row_index"] == idx
        assert record["predicted_label"] in LABEL_ORDER
        assert set(record["probabilities"].keys()) == set(LABEL_ORDER)
        assert pytest.approx(sum(record["probabilities"].values()), abs=1e-3) == 1.0
        # Privacy check: no raw text
        assert "text" not in record


def test_compare_validation_predictions_alignment_checks() -> None:
    """Mismatched row counts, indices, or true labels must raise ValueError."""
    row1 = {
        "row_index": 1,
        "target": "Hillary Clinton",
        "true_label": "AGAINST",
        "predicted_label": "AGAINST",
        "probabilities": {"AGAINST": 0.9, "FAVOR": 0.05, "NONE": 0.05},
    }
    row2 = {
        "row_index": 2,
        "target": "Hillary Clinton",
        "true_label": "FAVOR",
        "predicted_label": "FAVOR",
        "probabilities": {"AGAINST": 0.1, "FAVOR": 0.8, "NONE": 0.1},
    }

    # Length mismatch
    with pytest.raises(ValueError, match="length mismatch"):
        compare_validation_predictions([row1], [row1, row2])

    # Row index mismatch
    mismatched_index = dict(row1, row_index=99)
    with pytest.raises(ValueError, match="Row index mismatch"):
        compare_validation_predictions([row1], [mismatched_index])

    # True label mismatch
    mismatched_truth = dict(row1, true_label="FAVOR")
    with pytest.raises(ValueError, match="True label mismatch"):
        compare_validation_predictions([row1], [mismatched_truth])

    # Target mismatch
    mismatched_target = dict(row1, target="Atheism")
    with pytest.raises(ValueError, match="Target mismatch"):
        compare_validation_predictions([row1], [mismatched_target])


def test_privacy_enforcement_on_saving(tmp_path: Path) -> None:
    """Ensure that attempting to serialize raw tweet text raises an error."""
    bad_record = {
        "row_index": 1,
        "target": "Hillary Clinton",
        "true_label": "AGAINST",
        "predicted_label": "AGAINST",
        "text": "This is raw forbidden tweet text",
    }
    with pytest.raises(ValueError, match="Privacy violation"):
        save_validation_predictions([bad_record], tmp_path / "preds.json")

    clean_record = {
        "row_index": 1,
        "target": "Hillary Clinton",
        "true_label": "AGAINST",
        "predicted_label": "AGAINST",
        "probabilities": {"AGAINST": 0.8, "FAVOR": 0.1, "NONE": 0.1},
    }
    output_file = tmp_path / "preds.json"
    save_validation_predictions([clean_record], output_file)
    assert output_file.is_file()
    content = output_file.read_text(encoding="utf-8")
    assert '"text":' not in content


def test_saved_validation_predictions_integrity() -> None:
    """Verify saved validation predictions artifacts match existing experiment benchmarks."""
    base_path = Path("experiments/stance_tfidf_logreg/validation_predictions.json")
    distil_path = Path("experiments/stance_distilbert_target/validation_predictions.json")
    report_path = Path("experiments/model_comparison_report.json")

    if not base_path.is_file() or not distil_path.is_file() or not report_path.is_file():
        pytest.skip("Saved experiment validation predictions not generated yet.")

    base_preds = json.loads(base_path.read_text(encoding="utf-8"))
    distil_preds = json.loads(distil_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert len(base_preds) == 583
    assert len(distil_preds) == 583
    assert report["validation_rows"] == 583

    # Check identical row alignment and true labels
    for b, d in zip(base_preds, distil_preds):
        assert b["row_index"] == d["row_index"]
        assert b["target"] == d["target"]
        assert b["true_label"] == d["true_label"]
        assert "text" not in b and "text" not in d

    # Check accuracy matching documented benchmarks
    base_acc = sum(b["predicted_label"] == b["true_label"] for b in base_preds) / 583
    distil_acc = sum(d["predicted_label"] == d["true_label"] for d in distil_preds) / 583

    assert base_acc == pytest.approx(0.588336, abs=1e-5)
    assert distil_acc == pytest.approx(0.665523, abs=1e-5)

    # Check McNemar results in report
    m = report["mcnemar_test"]
    assert m["total_instances"] == 583
    assert m["both_correct"] == 285
    assert m["baseline_only_correct"] == 58
    assert m["candidate_only_correct"] == 103
    assert m["both_incorrect"] == 137
    assert m["discordant_pairs"] == 161
    assert m["statistic_continuity_corrected"] == pytest.approx(12.024845, rel=1e-4)
    assert m["is_statistically_significant_01"] is True
