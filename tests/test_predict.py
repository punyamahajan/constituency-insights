"""Unit tests for stance prediction and CLI interface.

Uses mocked models and tokenizers to ensure tests run offline without
requiring GPU access or downloading weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from constituency_insights.nlp.predict import (
    DEMO_EXAMPLES,
    StancePrediction,
    load_model_and_tokenizer,
    main,
    predict_stance,
    resolve_model_source,
    run_demonstration,
    validate_checkpoint_path,
)


class MockTokenizer:
    """Mock tokenizer returning deterministic tensor encodings and tracking inputs."""

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
        input_ids = torch.ones((batch_size, length), dtype=torch.long)
        attention_mask = torch.ones((batch_size, length), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class MockModel(nn.Module):
    """Mock classification model returning predefined logits."""

    def __init__(self, logits: list[float], is_fine_tuned: bool = True) -> None:
        super().__init__()
        self.fixed_logits = torch.tensor([logits], dtype=torch.float32)
        self.is_fine_tuned = is_fine_tuned

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        class MockOutput:
            def __init__(self, logits: torch.Tensor) -> None:
                self.logits = logits

        return MockOutput(self.fixed_logits)


def test_predict_stance_success() -> None:
    """Verify paired tokenization, argmax prediction, and probabilities calculation."""
    # Logits: index 0 (AGAINST)=1.0, index 1 (FAVOR)=5.0, index 2 (NONE)=0.5 -> FAVOR is highest
    mock_model = MockModel([1.0, 5.0, 0.5])
    mock_tokenizer = MockTokenizer()

    prediction = predict_stance(
        tweet_text="Hillary is our best choice.",
        target="Hillary Clinton",
        model=mock_model,
        tokenizer=mock_tokenizer,
        device="cpu",
    )

    assert isinstance(prediction, StancePrediction)
    assert prediction.predicted_stance == "FAVOR"
    assert prediction.tweet_text == "Hillary is our best choice."
    assert prediction.target == "Hillary Clinton"
    assert prediction.is_fine_tuned is True
    assert prediction.model_source == "fine_tuned_checkpoint"

    # Probabilities must sum to ~1.0
    probs = prediction.probabilities
    assert set(probs.keys()) == {"AGAINST", "FAVOR", "NONE"}
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)
    assert probs["FAVOR"] > probs["AGAINST"]
    assert probs["FAVOR"] > probs["NONE"]

    # Verify paired-text input tokenization
    assert mock_tokenizer.recorded_inputs[0] == ["Hillary is our best choice."]
    assert mock_tokenizer.recorded_text_pairs[0] == ["Hillary Clinton"]


def test_predict_stance_against_prediction() -> None:
    """Verify prediction when AGAINST logit is highest."""
    mock_model = MockModel([6.0, 1.0, 0.0], is_fine_tuned=False)
    mock_tokenizer = MockTokenizer()

    prediction = predict_stance(
        tweet_text="Oppose this bill completely.",
        target="Legalization of Abortion",
        model=mock_model,
        tokenizer=mock_tokenizer,
        device="cpu",
    )

    assert prediction.predicted_stance == "AGAINST"
    assert prediction.is_fine_tuned is False
    assert prediction.model_source == "cached_base_model"
    assert prediction.probabilities["AGAINST"] > 0.90


def test_predict_stance_empty_inputs_rejected() -> None:
    """Verify ValueError is raised on empty or whitespace-only inputs."""
    mock_model = MockModel([1.0, 1.0, 1.0])
    mock_tokenizer = MockTokenizer()

    with pytest.raises(ValueError, match="Tweet text must be a non-empty string"):
        predict_stance("", "target", model=mock_model, tokenizer=mock_tokenizer)

    with pytest.raises(ValueError, match="Tweet text must be a non-empty string"):
        predict_stance("   \n\t  ", "target", model=mock_model, tokenizer=mock_tokenizer)

    with pytest.raises(ValueError, match="Target entity must be a non-empty string"):
        predict_stance("some tweet", "", model=mock_model, tokenizer=mock_tokenizer)

    with pytest.raises(ValueError, match="Target entity must be a non-empty string"):
        predict_stance("some tweet", "   ", model=mock_model, tokenizer=mock_tokenizer)


def test_resolve_model_source_default_fallback_warning(tmp_path: Path) -> None:
    """Verify resolve_model_source falls back to base model with warning when default dir has no weights."""
    empty_dir = tmp_path / "empty_target_dir"
    empty_dir.mkdir()
    with patch("constituency_insights.nlp.predict.DEFAULT_CHECKPOINT_DIR", empty_dir):
        source, is_fine_tuned, warning = resolve_model_source(None)
        assert is_fine_tuned is False
        assert source == "distilbert/distilbert-base-uncased"
        assert warning is not None
        assert "unavailable" in warning
        assert "untrained classification head" in warning


def test_resolve_model_source_default_detects_fine_tuned_weights() -> None:
    """Verify resolve_model_source detects fine-tuned checkpoint when weights exist in default dir."""
    source, is_fine_tuned, warning = resolve_model_source(None)
    assert is_fine_tuned is True
    assert Path(source).resolve() == Path("experiments/stance_distilbert_target").resolve()
    assert warning is None


def test_resolve_model_source_with_weights(tmp_path: Path) -> None:
    """Verify resolve_model_source recognizes fine-tuned directory when weights exist."""
    (tmp_path / "model.safetensors").write_bytes(b"dummy")
    source, is_fine_tuned, warning = resolve_model_source(tmp_path)
    assert is_fine_tuned is True
    assert source == tmp_path
    assert warning is None


def test_resolve_model_source_explicit_missing_raises(tmp_path: Path) -> None:
    """Verify resolve_model_source raises FileNotFoundError when explicit path is missing weights."""
    empty_dir = tmp_path / "empty_checkpoint"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="does not contain model weights"):
        resolve_model_source(empty_dir)


def test_validate_checkpoint_missing_directory() -> None:
    """Verify FileNotFoundError when directory does not exist on disk."""
    with pytest.raises(FileNotFoundError, match="does not exist on disk"):
        validate_checkpoint_path("experiments/nonexistent_model_checkpoint_dir")


def test_validate_checkpoint_missing_weights_in_dir(tmp_path: Path) -> None:
    """Verify FileNotFoundError when directory exists but lacks weight files."""
    (tmp_path / "run_metadata.json").write_text("{}", encoding="utf-8")
    (tmp_path / "validation_metrics.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="does not contain model weights"):
        validate_checkpoint_path(tmp_path)


def test_validate_checkpoint_with_weights(tmp_path: Path) -> None:
    """Verify validation succeeds when model.safetensors or pytorch_model.bin exists."""
    (tmp_path / "model.safetensors").write_bytes(b"dummy")
    result = validate_checkpoint_path(tmp_path)
    assert result == tmp_path


def test_validate_checkpoint_from_config_json(tmp_path: Path) -> None:
    """Verify that a config JSON resolves to its model_name_or_path."""
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps({"model_name_or_path": "distilbert/distilbert-base-uncased"}),
        encoding="utf-8",
    )
    result = validate_checkpoint_path(config_file)
    assert result == "distilbert/distilbert-base-uncased"


def test_validate_checkpoint_empty_rejected() -> None:
    """Verify ValueError on empty checkpoint string."""
    with pytest.raises(ValueError, match="non-empty string"):
        validate_checkpoint_path("")
    with pytest.raises(ValueError, match="non-empty string"):
        validate_checkpoint_path("   ")


def test_cli_main_success_formatted(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI prints formatted text output with model provenance."""
    fake_prediction = StancePrediction(
        tweet_text="Sample tweet",
        target="Sample target",
        predicted_stance="FAVOR",
        probabilities={"AGAINST": 0.1, "FAVOR": 0.8, "NONE": 0.1},
        model_source="fine_tuned_checkpoint",
        model_identifier="experiments/stance_distilbert_target",
        is_fine_tuned=True,
    )
    with patch("constituency_insights.nlp.predict.predict_stance", return_value=fake_prediction):
        exit_code = main(["--tweet", "Sample tweet", "--target", "Sample target"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Prediction Result:" in captured.out
        assert "Model Source:     fine_tuned_checkpoint" in captured.out
        assert "Fine-Tuned:       Yes" in captured.out
        assert "Predicted Stance: FAVOR" in captured.out
        assert "Sample target" in captured.out


def test_cli_main_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI prints valid JSON with provenance when --json flag is passed."""
    fake_prediction = StancePrediction(
        tweet_text="Sample tweet",
        target="Sample target",
        predicted_stance="FAVOR",
        probabilities={"AGAINST": 0.1, "FAVOR": 0.8, "NONE": 0.1},
        model_source="cached_base_model",
        model_identifier="distilbert/distilbert-base-uncased",
        is_fine_tuned=False,
        warning_message="Untrained classification head.",
    )
    with patch("constituency_insights.nlp.predict.predict_stance", return_value=fake_prediction):
        exit_code = main(["--tweet", "Sample tweet", "--target", "Sample target", "--json"])
        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["predicted_stance"] == "FAVOR"
        assert data["is_fine_tuned"] is False
        assert data["model_source"] == "cached_base_model"
        assert data["probabilities"]["FAVOR"] == 0.8


def test_cli_main_warns_on_base_model(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI writes warning to stderr when using base model."""
    fake_prediction = StancePrediction(
        tweet_text="Sample tweet",
        target="Sample target",
        predicted_stance="NONE",
        probabilities={"AGAINST": 0.33, "FAVOR": 0.33, "NONE": 0.34},
        model_source="cached_base_model",
        model_identifier="distilbert/distilbert-base-uncased",
        is_fine_tuned=False,
        warning_message="Using base pretrained model with untrained head.",
    )
    with patch("constituency_insights.nlp.predict.predict_stance", return_value=fake_prediction):
        exit_code = main(["--tweet", "Sample tweet", "--target", "Sample target"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "WARNING: Using base pretrained model" in captured.err
        assert "Fine-Tuned:       No (untrained classification head)" in captured.out


def test_cli_main_handles_explicit_missing_checkpoint_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI handles missing checkpoint error cleanly and exits with code 1."""
    empty_dir = tmp_path / "empty_checkpoint"
    empty_dir.mkdir()
    exit_code = main(
        [
            "--tweet", "Sample tweet",
            "--target", "Sample target",
            "--checkpoint-path", str(empty_dir),
        ]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error:" in captured.err
    assert "does not contain model weights" in captured.err


def test_cli_main_missing_required_args_returns_2(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI returns code 2 when required arguments are missing."""
    exit_code = main([])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "Error: Both --tweet/-t and --target/-g are required" in captured.err


def test_cli_main_demo_mode_formatted(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify --demo executes all illustrative examples with disclaimers."""
    fake_pred = StancePrediction(
        tweet_text="Demo tweet",
        target="Demo target",
        predicted_stance="FAVOR",
        probabilities={"AGAINST": 0.2, "FAVOR": 0.6, "NONE": 0.2},
        model_source="cached_base_model",
        model_identifier="distilbert/distilbert-base-uncased",
        is_fine_tuned=False,
    )
    with patch("constituency_insights.nlp.predict.predict_stance", return_value=fake_pred):
        exit_code = main(["--demo"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "CONSTITUENCY INSIGHTS: STANCE CLASSIFICATION DEMONSTRATION" in captured.out
        assert "NOT benchmark evaluations" in captured.out
        assert "Demonstration completed successfully." in captured.out
        for item in DEMO_EXAMPLES:
            assert item["label"] in captured.out


def test_cli_main_demo_mode_json(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify --demo --json outputs valid JSON structure with all examples."""
    fake_pred = StancePrediction(
        tweet_text="Demo tweet",
        target="Demo target",
        predicted_stance="FAVOR",
        probabilities={"AGAINST": 0.2, "FAVOR": 0.6, "NONE": 0.2},
        model_source="cached_base_model",
        model_identifier="distilbert/distilbert-base-uncased",
        is_fine_tuned=False,
    )
    with patch("constituency_insights.nlp.predict.predict_stance", return_value=fake_pred):
        exit_code = main(["--demo", "--json"])
        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "demonstration_results" in data
        assert len(data["demonstration_results"]) == len(DEMO_EXAMPLES)
