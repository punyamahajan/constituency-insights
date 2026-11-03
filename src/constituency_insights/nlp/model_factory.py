"""Backbone and optional LoRA construction for stance classifiers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Sequence

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from constituency_insights.nlp.distilbert import LABEL_ORDER


BACKBONES = {
    "distilbert-base-uncased": "distilbert/distilbert-base-uncased",
    "xlm-roberta-base": "FacebookAI/xlm-roberta-base",
    "muril-base-cased": "google/muril-base-cased",
    "indicbert": "ai4bharat/indic-bert",
}


@dataclass(frozen=True)
class ModelFactoryConfig:
    """Settings that determine the backbone and fine-tuning method."""

    backbone: str
    tuning_method: str = "full"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: tuple[str, ...] = ("query", "key", "value")


def resolve_backbone(backbone: str) -> str:
    """Resolve a supported short name while permitting explicit optional IDs."""

    return BACKBONES.get(backbone, backbone)


def create_model_and_tokenizer(
    config: ModelFactoryConfig,
    *,
    label_order: Sequence[str] = LABEL_ORDER,
    local_files_only: bool = True,
) -> tuple[Any, Any]:
    """Create one target-conditioned-compatible tokenizer and classifier."""

    if config.tuning_method not in {"full", "lora"}:
        raise ValueError(f"Unsupported tuning method: {config.tuning_method!r}")
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    model_id = resolve_backbone(config.backbone)
    label_to_id = {label: index for index, label in enumerate(label_order)}
    id_to_label = {index: label for index, label in enumerate(label_order)}
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, local_files_only=local_files_only, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=len(label_order),
        label2id=label_to_id,
        id2label=id_to_label,
        local_files_only=local_files_only,
    )
    if config.tuning_method == "lora":
        if config.lora_rank < 1 or config.lora_alpha < 1:
            raise ValueError("LoRA rank and alpha must be positive.")
        lora = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.lora_target_modules),
            bias="none",
        )
        model = get_peft_model(model, lora)
    return model, tokenizer


def parameter_counts(model: Any) -> tuple[int, int]:
    """Return trainable and total parameter counts."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return trainable, total
