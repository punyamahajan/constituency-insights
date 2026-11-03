"""Synthetic unit tests for DistilBERT experiment code.

All tests mock the tokenizer and model; no weights are downloaded and no
network requests are made.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
import torch.nn as nn

from constituency_insights.data import prepare_task_a_validation_split
from constituency_insights.nlp.distilbert import (
    DEFAULT_MODEL_ID,
    LABEL_ORDER,
    DistilBertExperimentConfig,
    DistilBertRun,
    build_label_mappings,
    train_and_evaluate_distilbert,
    write_distilbert_outputs,
)


TARGETS = {"Policy Alpha", "Policy Beta"}
LABELS = set(LABEL_ORDER)


def _synthetic_split():
    """Create a reproducible synthetic in-memory split (72 dev-train, 18 val)."""
    rows = [
        {
            "text": f"synthetic statement {number} for label {label}",
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


class MockTokenizer:
    """Mock tokenizer returning deterministic tensor dictionaries."""

    def __init__(self, max_length: int = 128) -> None:
        self.max_length = max_length
        self.recorded_inputs: list[list[str]] = []
        self.recorded_text_pairs: list[list[str] | None] = []

    def __call__(
        self,
        texts: list[str],
        text_pair: list[str] | None = None,
        padding: str | bool = True,
        truncation: bool = True,
        max_length: int | None = None,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        self.recorded_inputs.append(list(texts))
        self.recorded_text_pairs.append(list(text_pair) if text_pair is not None else None)
        length = max_length or self.max_length
        batch_size = len(texts)
        # Deterministic pseudo-tokens based on sequence index
        input_ids = torch.zeros((batch_size, length), dtype=torch.long)
        attention_mask = torch.ones((batch_size, length), dtype=torch.long)
        for i, text in enumerate(texts):
            input_ids[i, 0] = 101  # [CLS] token
            input_ids[i, 1] = hash(text) % 1000 + 1000
            input_ids[i, 2] = 102  # [SEP] token
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class MockSequenceClassifier(nn.Module):
    """Mock classification model providing loss and logits."""

    def __init__(self, num_labels: int = 3, hidden_size: int = 128) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> Any:
        assert input_ids is not None
        batch_size, seq_len = input_ids.shape
        # Simple projection using input_ids as floats
        features = input_ids.float()
        logits = self.linear(features)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        class ModelOutput:
            def __init__(self, loss_val: torch.Tensor | None, logits_val: torch.Tensor) -> None:
                self.loss = loss_val
                self.logits = logits_val

        return ModelOutput(loss, logits)


def test_distilbert_config_loading_and_defaults() -> None:
    """Verify default parameters match the locked specification."""
    config = DistilBertExperimentConfig()

    assert config.model_name_or_path == DEFAULT_MODEL_ID
    assert config.random_seed == 42
    assert config.max_length == 128
    assert config.per_device_train_batch_size == 8
    assert config.per_device_eval_batch_size == 8
    assert config.gradient_accumulation_steps == 2
    assert config.effective_batch_size == 16
    assert config.learning_rate == 2e-5
    assert config.weight_decay == 0.01
    assert config.num_train_epochs == 3
    assert config.warmup_ratio == 0.0
    assert config.fp16 is True
    assert config.label_order == LABEL_ORDER
    assert config.small_support_threshold == 10

    # Load from config file
    loaded = DistilBertExperimentConfig.from_json("configs/distilbert_baseline.json")
    assert loaded == config


def test_label_mapping_bidirectional_and_order() -> None:
    """Verify bidirectional mappings maintain fixed LABEL_ORDER."""
    label_to_id, id_to_label = build_label_mappings(LABEL_ORDER)

    assert label_to_id == {"AGAINST": 0, "FAVOR": 1, "NONE": 2}
    assert id_to_label == {0: "AGAINST", 1: "FAVOR", 2: "NONE"}
    assert [id_to_label[i] for i in range(len(LABEL_ORDER))] == list(LABEL_ORDER)


def test_official_test_rows_are_rejected() -> None:
    """Refuse evaluation if any row comes from the official test split."""
    split = _synthetic_split()
    split.validation.loc[split.validation.index[0], "split"] = "test"

    config = DistilBertExperimentConfig(num_train_epochs=1)
    with pytest.raises(ValueError, match="official test rows"):
        train_and_evaluate_distilbert(
            split,
            config=config,
            tokenizer=MockTokenizer(),
            model=MockSequenceClassifier(),
            device="cpu",
        )


def test_partition_overlap_is_rejected() -> None:
    """Refuse evaluation if development and validation partitions overlap."""
    split = _synthetic_split()
    overlap_index = split.development_train.index[0]
    overlapping_val = pd.concat([split.validation, split.development_train.loc[[overlap_index]]])

    split_with_overlap = type(split)(
        development_train=split.development_train,
        validation=overlapping_val,
        report=split.report,
    )
    with pytest.raises(ValueError, match="partitions overlap"):
        train_and_evaluate_distilbert(
            split_with_overlap,
            tokenizer=MockTokenizer(),
            model=MockSequenceClassifier(),
            device="cpu",
        )


def test_text_only_input_no_target_concatenation() -> None:
    """Verify only normalized text is sent to the tokenizer, without target names."""
    split = _synthetic_split()
    tokenizer = MockTokenizer()
    model = MockSequenceClassifier()
    config = DistilBertExperimentConfig(num_train_epochs=1)

    train_and_evaluate_distilbert(
        split,
        config=config,
        tokenizer=tokenizer,
        model=model,
        device="cpu",
    )

    # All recorded inputs to the tokenizer must match the text column exactly
    train_texts = tokenizer.recorded_inputs[0]
    val_texts = tokenizer.recorded_inputs[1]

    assert train_texts == split.development_train["text"].tolist()
    assert val_texts == split.validation["text"].tolist()

    # Confirm no target names were injected into the text
    for target in TARGETS:
        assert not any(target in text for text in train_texts)
        assert not any(target in text for text in val_texts)


def test_synthetic_training_loop_and_metrics() -> None:
    """End-to-end run on synthetic split verifying metric schema and determinism."""
    split = _synthetic_split()
    config = DistilBertExperimentConfig(num_train_epochs=1)

    run = train_and_evaluate_distilbert(
        split,
        config=config,
        tokenizer=MockTokenizer(),
        model=MockSequenceClassifier(),
        device="cpu",
    )

    assert isinstance(run, DistilBertRun)
    metrics = run.metrics

    assert metrics["label_order"] == list(LABEL_ORDER)
    assert metrics["development_train_rows"] == 72
    assert metrics["validation_rows"] == 18
    assert metrics["effective_batch_size"] == 16
    assert metrics["fp16_enabled"] is False  # CPU run disables FP16

    # Overall metrics
    overall = metrics["overall"]
    assert 0.0 <= overall["macro_f1"] <= 1.0
    assert 0.0 <= overall["accuracy"] <= 1.0
    assert list(overall["per_class"]) == list(LABEL_ORDER)
    assert len(overall["confusion_matrix"]) == 3
    assert all(len(row) == 3 for row in overall["confusion_matrix"])
    assert sum(sum(r) for r in overall["confusion_matrix"]) == 18

    # Per-target metrics
    assert set(metrics["by_target"]) == TARGETS
    for target in TARGETS:
        target_m = metrics["by_target"][target]
        assert "macro_f1" in target_m
        assert "per_class" in target_m
        assert "confusion_matrix" in target_m
        assert "small_support_warning" in target_m
        # Synthetic split has support 3 for each label per target, so warning must be active
        warning = target_m["small_support_warning"]
        assert warning is not None
        assert warning["threshold"] == 10
        assert set(warning["labels"]) == LABELS

    # Majority baseline
    majority = metrics["majority_baseline"]
    assert majority["label"] in LABELS
    assert 0.0 <= majority["accuracy"] <= 1.0
    assert 0.0 <= majority["macro_f1"] <= 1.0

    # Run metadata
    metadata = run.run_metadata
    assert "timestamp_utc" in metadata
    assert "torch_version" in metadata
    assert metadata["device"] == "cpu"
    assert metadata["fp16_enabled"] is False


def test_write_distilbert_outputs_privacy_safe(tmp_path: Path) -> None:
    """Ensure aggregate results and metadata contain no tweet text or excerpts."""
    split = _synthetic_split()
    config = DistilBertExperimentConfig(num_train_epochs=1)

    run = train_and_evaluate_distilbert(
        split,
        config=config,
        tokenizer=MockTokenizer(),
        model=MockSequenceClassifier(),
        device="cpu",
    )

    out_dir = tmp_path / "stance_distilbert"
    write_distilbert_outputs(run, out_dir)

    metrics_file = out_dir / "validation_metrics.json"
    metadata_file = out_dir / "run_metadata.json"
    assert metrics_file.is_file()
    assert metadata_file.is_file()

    metrics_content = metrics_file.read_text(encoding="utf-8")
    metadata_content = metadata_file.read_text(encoding="utf-8")

    # Verify no raw text or excerpt leaks
    for forbidden in ('"text":', '"text_excerpt":', '"Tweet":'):
        assert forbidden not in metrics_content
        assert forbidden not in metadata_content

    # Verify error analysis CSV with excerpts was NOT created
    assert not (out_dir / "validation_error_analysis.csv").exists()


def test_distilbert_target_config_loading() -> None:
    """Verify Track B configuration loads target_conditioned=True while preserving other settings."""
    config_a = DistilBertExperimentConfig.from_json("configs/distilbert_baseline.json")
    config_b = DistilBertExperimentConfig.from_json("configs/distilbert_target_baseline.json")

    assert config_a.target_conditioned is False
    assert config_b.target_conditioned is True

    # Invariances: All training parameters must match between Track A and Track B
    assert config_b.model_name_or_path == config_a.model_name_or_path
    assert config_b.random_seed == config_a.random_seed
    assert config_b.max_length == config_a.max_length
    assert config_b.per_device_train_batch_size == config_a.per_device_train_batch_size
    assert config_b.per_device_eval_batch_size == config_a.per_device_eval_batch_size
    assert config_b.gradient_accumulation_steps == config_a.gradient_accumulation_steps
    assert config_b.effective_batch_size == config_a.effective_batch_size
    assert config_b.learning_rate == config_a.learning_rate
    assert config_b.weight_decay == config_a.weight_decay
    assert config_b.num_train_epochs == config_a.num_train_epochs
    assert config_b.fp16 == config_a.fp16
    assert config_b.label_order == config_a.label_order


def test_track_b_paired_input_tokenization() -> None:
    """Verify Track B passes the target as text_pair without concatenating into text."""
    split = _synthetic_split()
    tokenizer = MockTokenizer()
    model = MockSequenceClassifier()
    config = DistilBertExperimentConfig(num_train_epochs=1, target_conditioned=True)

    train_and_evaluate_distilbert(
        split,
        config=config,
        tokenizer=tokenizer,
        model=model,
        device="cpu",
    )

    # First call is train encoding, second call is val encoding
    train_texts = tokenizer.recorded_inputs[0]
    train_pairs = tokenizer.recorded_text_pairs[0]
    val_texts = tokenizer.recorded_inputs[1]
    val_pairs = tokenizer.recorded_text_pairs[1]

    # Primary text must be strictly unpolluted tweet text
    assert train_texts == split.development_train["text"].tolist()
    assert val_texts == split.validation["text"].tolist()

    # Paired inputs must be exactly the target column
    assert train_pairs == split.development_train["target"].tolist()
    assert val_pairs == split.validation["target"].tolist()

    # Confirm text and pairs are separate sequences
    assert all(pair in TARGETS for pair in train_pairs)
    assert all(pair in TARGETS for pair in val_pairs)


def test_track_b_synthetic_training_and_metrics(tmp_path: Path) -> None:
    """Verify Track B run records target_conditioned metadata and outputs to target directory."""
    split = _synthetic_split()
    config = DistilBertExperimentConfig(num_train_epochs=1, target_conditioned=True)

    run = train_and_evaluate_distilbert(
        split,
        config=config,
        tokenizer=MockTokenizer(),
        model=MockSequenceClassifier(),
        device="cpu",
    )

    assert run.config.target_conditioned is True
    assert run.metrics["target_conditioned"] is True
    assert run.metrics["experiment"] == "distilbert_base_uncased_target_conditioned_stance"
    assert "target-conditioned" in run.metrics["setting_note"]
    assert run.run_metadata["config"]["target_conditioned"] is True

    out_dir = tmp_path / "stance_distilbert_target"
    write_distilbert_outputs(run, out_dir)
    assert (out_dir / "validation_metrics.json").is_file()
    assert (out_dir / "run_metadata.json").is_file()

    metrics = json.loads((out_dir / "validation_metrics.json").read_text(encoding="utf-8"))
    assert metrics["target_conditioned"] is True


def test_distilbert_run_subscripting_and_get() -> None:
    """Verify DistilBertRun supports dictionary-like subscripting and .get()."""
    config = DistilBertExperimentConfig(num_train_epochs=1)
    metrics = {"overall": {"macro_f1": 0.65}, "custom_metric": 42}
    run = DistilBertRun(
        config=config,
        metrics=metrics,
        run_metadata={"device": "cpu"},
        predictions=["AGAINST", "FAVOR"],
    )

    # Subscripting attributes
    assert run["config"] == config
    assert run["predictions"] == ["AGAINST", "FAVOR"]

    # Subscripting top-level metrics
    assert run["overall"] == {"macro_f1": 0.65}
    assert run["custom_metric"] == 42

    # Safe get()
    assert run.get("overall") == {"macro_f1": 0.65}
    assert run.get("missing_key") is None
    assert run.get("missing_key", "default_val") == "default_val"

    # KeyError on missing key
    with pytest.raises(KeyError, match="DistilBertRun has no attribute or metric key"):
        _ = run["nonexistent"]


def test_distilbert_optional_save_predictions(tmp_path: Path) -> None:
    """Verify that row-level validation predictions can be optionally saved without raw text."""
    split = _synthetic_split()
    config = DistilBertExperimentConfig(num_train_epochs=1, save_predictions=False)

    run = train_and_evaluate_distilbert(
        split,
        config=config,
        tokenizer=MockTokenizer(),
        model=MockSequenceClassifier(),
        device="cpu",
    )

    assert run.validation_predictions is not None
    assert len(run.validation_predictions) == len(split.validation)
    sample = run.validation_predictions[0]
    assert "row_index" in sample
    assert "target" in sample
    assert "true_label" in sample
    assert "predicted_label" in sample
    assert "text" not in sample

    # When save_predictions is False (default), validation_predictions.json is NOT saved
    out_dir_no_save = tmp_path / "no_save"
    write_distilbert_outputs(run, out_dir_no_save, save_predictions=False)
    assert not (out_dir_no_save / "validation_predictions.json").exists()

    # When save_predictions is True, validation_predictions.json IS saved
    out_dir_save = tmp_path / "save"
    write_distilbert_outputs(run, out_dir_save, save_predictions=True)
    preds_file = out_dir_save / "validation_predictions.json"
    assert preds_file.is_file()

    preds_content = preds_file.read_text(encoding="utf-8")
    for forbidden in ('"text":', '"text_excerpt":', '"Tweet":'):
        assert forbidden not in preds_content

    saved_records = json.loads(preds_content)
    assert len(saved_records) == len(split.validation)
    assert saved_records[0]["row_index"] == split.validation.index[0]
    assert saved_records[0]["target"] == split.validation.iloc[0]["target"]
    assert saved_records[0]["true_label"] == split.validation.iloc[0]["label"]


def test_distilbert_save_predictions_privacy_guard(tmp_path: Path) -> None:
    """Verify that write_distilbert_outputs rejects leaked text in validation_predictions."""
    config = DistilBertExperimentConfig(num_train_epochs=1)
    leaked_record = [
        {"row_index": 1, "target": "Alpha", "true_label": "FAVOR", "predicted_label": "FAVOR", "text": "leaked tweet text"}
    ]
    run = DistilBertRun(
        config=config,
        metrics={},
        run_metadata={},
        predictions=["FAVOR"],
        validation_predictions=leaked_record,
    )
    with pytest.raises(ValueError, match="Privacy violation"):
        write_distilbert_outputs(run, tmp_path / "privacy_leak", save_predictions=True)


