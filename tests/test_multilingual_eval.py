from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch

from constituency_insights.nlp.multilingual_eval import (
    evaluate_multilingual_examples,
    load_multilingual_examples,
)


class TinyTokenizer:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.targets: list[str] = []

    def __call__(self, texts, *, text_pair, **_kwargs):
        self.texts = list(texts)
        self.targets = list(text_pair)
        size = len(texts)
        return {"input_ids": torch.arange(size).reshape(size, 1)}


class TinyModel(torch.nn.Module):
    def forward(self, input_ids: torch.Tensor, **_kwargs: Any):
        logits = torch.zeros((len(input_ids), 3))
        logits[torch.arange(len(input_ids)), input_ids[:, 0] % 3] = 5
        return type("Output", (), {"logits": logits})()


def test_loader_accepts_tiny_user_supplied_fixture_without_modifying_text(tmp_path: Path) -> None:
    path = tmp_path / "examples.csv"
    original = pd.DataFrame({
        "text": ["synthetic one", "synthetic two", "synthetic three"],
        "target": ["target a", "target b", "target c"],
        "label": ["AGAINST", "FAVOR", "NONE"],
    })
    original.to_csv(path, index=False)

    loaded = load_multilingual_examples(path)

    pd.testing.assert_frame_equal(loaded, original)


def test_target_conditioned_evaluation_returns_only_aggregate_metrics(tmp_path: Path) -> None:
    path = tmp_path / "examples.csv"
    pd.DataFrame({
        "text": ["synthetic one", "synthetic two", "synthetic three"],
        "target": ["target a", "target b", "target c"],
        "label": ["AGAINST", "FAVOR", "NONE"],
    }).to_csv(path, index=False)
    frame = load_multilingual_examples(path)
    tokenizer = TinyTokenizer()

    metrics = evaluate_multilingual_examples(frame, TinyModel(), tokenizer)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert tokenizer.texts == frame["text"].tolist()
    assert tokenizer.targets == frame["target"].tolist()
    assert "text" not in metrics and "predictions" not in metrics


@pytest.mark.parametrize("columns", [["text", "label"], ["target", "text", "label"]])
def test_loader_rejects_wrong_schema(tmp_path: Path, columns: list[str]) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame([["x"] * len(columns)], columns=columns).to_csv(path, index=False)
    with pytest.raises(ValueError, match="columns must be exactly"):
        load_multilingual_examples(path)
