"""User-supplied multilingual robustness evaluation without data fabrication."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from peft import AutoPeftModelForSequenceClassification
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from constituency_insights.nlp.distilbert import LABEL_ORDER, StanceTextDataset


REQUIRED_COLUMNS = ("text", "target", "label")


def load_multilingual_examples(path: Path | str) -> pd.DataFrame:
    """Load and validate a user-provided labeled CSV without altering its text."""

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if tuple(frame.columns) != REQUIRED_COLUMNS:
        raise ValueError(f"CSV columns must be exactly {list(REQUIRED_COLUMNS)} in that order.")
    if frame.empty:
        raise ValueError("Multilingual evaluation CSV contains no rows.")
    for column in REQUIRED_COLUMNS:
        empty = frame[column].str.strip().eq("")
        if empty.any():
            raise ValueError(f"Column {column!r} contains an empty value.")
    unexpected = sorted(set(frame["label"]).difference(LABEL_ORDER))
    if unexpected:
        raise ValueError(f"Unsupported labels: {unexpected}")
    return frame


def evaluate_multilingual_examples(
    frame: pd.DataFrame,
    model: Any,
    tokenizer: Any,
    *,
    batch_size: int = 16,
    max_length: int = 128,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Evaluate paired text/target inputs and return aggregate metrics only."""

    target_device = torch.device(device)
    label_to_id = {label: index for index, label in enumerate(LABEL_ORDER)}
    encodings = tokenizer(
        frame["text"].tolist(), text_pair=frame["target"].tolist(),
        padding="max_length", truncation=True, max_length=max_length,
        return_tensors="pt",
    )
    labels = [label_to_id[label] for label in frame["label"]]
    loader = DataLoader(StanceTextDataset(encodings, labels), batch_size=batch_size)
    model.to(target_device)
    model.eval()
    truth: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(target_device) for key, value in batch.items()}
            batch_labels = batch.pop("labels")
            logits = model(**batch).logits
            truth.extend(batch_labels.cpu().tolist())
            predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    return {
        "rows": len(frame),
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, labels=[0, 1, 2], average="macro", zero_division=0)),
        "label_order": list(LABEL_ORDER),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=[0, 1, 2]).astype(int).tolist(),
        "scope_note": "User-provided robustness sample; not evidence of cross-lingual generalization.",
    }


def load_saved_model(checkpoint: Path | str, device: str | torch.device):
    """Load either a full checkpoint or a PEFT adapter from local disk."""

    checkpoint_path = Path(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
    if (checkpoint_path / "adapter_config.json").is_file():
        model = AutoPeftModelForSequenceClassification.from_pretrained(
            checkpoint_path, local_files_only=True
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path, local_files_only=True
        )
    model.to(torch.device(device))
    return model, tokenizer


def evaluate_csv_checkpoint(
    csv_path: Path | str,
    checkpoint: Path | str,
    *,
    batch_size: int = 16,
    max_length: int = 128,
    device: str | None = None,
) -> dict[str, Any]:
    """Load a local checkpoint and user CSV and return privacy-safe metrics."""

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    frame = load_multilingual_examples(csv_path)
    model, tokenizer = load_saved_model(checkpoint, resolved_device)
    return evaluate_multilingual_examples(
        frame, model, tokenizer, batch_size=batch_size,
        max_length=max_length, device=resolved_device,
    )
