"""Leakage-safe DistilBERT stance classification experiment.

Reuses the exact in-memory development-training/validation split established for
SemEval-2016 Task 6, Task A. Evaluates text-only input against the fixed
three-class stance labels (AGAINST, FAVOR, NONE).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from constituency_insights.data import (
    TASK_A_TARGETS,
    ValidationSplit,
    load_task_a,
    official_task_a_file_specs,
    prepare_task_a_validation_split,
)


LABEL_ORDER = ("AGAINST", "FAVOR", "NONE")
DEFAULT_MODEL_ID = "distilbert/distilbert-base-uncased"

FALLBACK_TIERS: tuple[tuple[int, int], ...] = (
    (8, 2),   # Tier 1 (Default): 8 device batch, 2 grad accum -> 16 effective
    (4, 4),   # Tier 2: 4 device batch, 4 grad accum -> 16 effective
    (2, 8),   # Tier 3: 2 device batch, 8 grad accum -> 16 effective
    (1, 16),  # Tier 4: 1 device batch, 16 grad accum -> 16 effective
)


def check_cuda_feasibility(
    model: torch.nn.Module,
    sample_batch: Mapping[str, torch.Tensor],
    device: torch.device,
    use_fp16: bool,
    gradient_accumulation_steps: int,
) -> bool:
    """Trial single forward and backward step on target device to verify CUDA memory."""
    try:
        model.train()
        batch = {k: v.to(device) for k, v in sample_batch.items()}
        with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
        loss.backward()
        model.zero_grad(set_to_none=True)
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()
        return True
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" in str(exc).lower():
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.empty_cache()
            return False
        raise exc


@dataclass(frozen=True)
class DistilBertExperimentConfig:
    """Fixed experiment parameters for DistilBERT stance classification."""

    model_name_or_path: str = DEFAULT_MODEL_ID
    random_seed: int = 42
    max_length: int = 128
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_train_epochs: int = 3
    warmup_ratio: float = 0.0
    fp16: bool = True
    label_order: tuple[str, ...] = LABEL_ORDER
    small_support_threshold: int = 10
    target_conditioned: bool = False
    save_predictions: bool = False

    @property
    def effective_batch_size(self) -> int:
        """Effective batch size after gradient accumulation."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    @classmethod
    def from_json(cls, path: Path | str) -> DistilBertExperimentConfig:
        """Load configuration from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        label_order = (
            tuple(data["label_order"])
            if "label_order" in data
            else LABEL_ORDER
        )
        return cls(
            model_name_or_path=data.get("model_name_or_path", DEFAULT_MODEL_ID),
            random_seed=data.get("random_seed", 42),
            max_length=data.get("max_length", 128),
            per_device_train_batch_size=data.get("per_device_train_batch_size", 8),
            per_device_eval_batch_size=data.get("per_device_eval_batch_size", 8),
            gradient_accumulation_steps=data.get("gradient_accumulation_steps", 2),
            learning_rate=data.get("learning_rate", 2e-5),
            weight_decay=data.get("weight_decay", 0.01),
            num_train_epochs=data.get("num_train_epochs", 3),
            warmup_ratio=data.get("warmup_ratio", 0.0),
            fp16=data.get("fp16", True),
            label_order=label_order,
            small_support_threshold=data.get("small_support_threshold", 10),
            target_conditioned=data.get("target_conditioned", False),
            save_predictions=data.get("save_predictions", False),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        d = asdict(self)
        d["effective_batch_size"] = self.effective_batch_size
        d["label_order"] = list(self.label_order)
        d["target_conditioned"] = self.target_conditioned
        d["save_predictions"] = self.save_predictions
        return d


@dataclass(frozen=True)
class DistilBertRun:
    """Artifacts and validation evaluation outputs from a DistilBERT run."""

    config: DistilBertExperimentConfig
    metrics: dict[str, Any]
    run_metadata: dict[str, Any]
    predictions: list[str]
    validation_predictions: list[dict[str, Any]] | None = None
    model: Any | None = None
    tokenizer: Any | None = None

    def __getitem__(self, item: str) -> Any:
        """Allow dict-style subscripting for attributes and top-level metric keys."""
        if hasattr(self, item):
            return getattr(self, item)
        if item in self.metrics:
            return self.metrics[item]
        raise KeyError(f"DistilBertRun has no attribute or metric key {item!r}.")

    def get(self, item: str, default: Any = None) -> Any:
        """Safe retrieval matching dict.get interface."""
        try:
            return self[item]
        except KeyError:
            return default


class StanceTextDataset(Dataset):
    """PyTorch Dataset wrapping pre-tokenized features and optional labels."""

    def __init__(
        self,
        encodings: Mapping[str, Any],
        labels: Sequence[int] | None = None,
    ) -> None:
        self.encodings = {
            key: val if isinstance(val, torch.Tensor) else torch.tensor(val)
            for key, val in encodings.items()
        }
        self.labels = (
            torch.tensor(labels, dtype=torch.long)
            if labels is not None
            else None
        )

    def __len__(self) -> int:
        return len(next(iter(self.encodings.values())))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: tensor[idx] for key, tensor in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


def set_deterministic_seed(seed: int) -> None:
    """Set random seeds across Python, NumPy, PyTorch, and cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_label_mappings(
    label_order: Sequence[str] = LABEL_ORDER,
) -> tuple[dict[str, int], dict[int, str]]:
    """Return bidirectional string-to-ID and ID-to-string mappings."""
    label_to_id = {label: idx for idx, label in enumerate(label_order)}
    id_to_label = {idx: label for idx, label in enumerate(label_order)}
    return label_to_id, id_to_label


def _validate_partitions(
    development: pd.DataFrame, validation: pd.DataFrame
) -> None:
    """Verify partition isolation, expected sizes, and absence of test leakage."""
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


def _classification_metrics(
    truth: pd.Series | Sequence[str],
    predictions: Sequence[str],
    labels: list[str],
) -> dict[str, Any]:
    """Compute aggregate classification metrics with explicit label ordering."""
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


def _build_evaluation_metrics(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    predictions: Sequence[str],
    config: DistilBertExperimentConfig,
    device: torch.device,
    use_fp16: bool,
    split_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile comprehensive validation metrics matching project schema."""
    labels = list(config.label_order)
    majority_label = str(development["label"].value_counts().idxmax())
    majority_predictions = [majority_label] * len(validation)

    experiment_id = (
        "distilbert_base_uncased_target_conditioned_stance"
        if config.target_conditioned
        else "distilbert_base_uncased_stance"
    )
    setting_note = (
        "Fixed initial DistilBERT target-conditioned (Track B) settings; paired input [CLS] tweet [SEP] target [SEP]; not tuned against validation."
        if config.target_conditioned
        else "Fixed initial DistilBERT settings; not tuned against validation."
    )

    metrics: dict[str, Any] = {
        "experiment": experiment_id,
        "setting_note": setting_note,
        "target_conditioned": config.target_conditioned,
        "model_name_or_path": config.model_name_or_path,
        "random_seed": config.random_seed,
        "max_length": config.max_length,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "effective_batch_size": config.effective_batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "num_train_epochs": config.num_train_epochs,
        "warmup_ratio": config.warmup_ratio,
        "fp16_enabled": use_fp16,
        "device": str(device),
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
    }
    if split_report is not None:
        metrics["split_report"] = split_report

    preds_series = pd.Series(predictions, index=validation.index)
    for target in sorted(validation["target"].unique()):
        mask = validation["target"].eq(target)
        target_metrics = _classification_metrics(
            validation.loc[mask, "label"], preds_series.loc[mask], labels
        )
        small_labels = {
            label: int(target_metrics["per_class"][label]["support"])
            for label in labels
            if int(target_metrics["per_class"][label]["support"])
            < config.small_support_threshold
        }
        target_metrics["small_support_warning"] = (
            {
                "threshold": config.small_support_threshold,
                "labels": small_labels,
                "message": "Per-class metrics with small support are unstable.",
            }
            if small_labels
            else None
        )
        metrics["by_target"][target] = target_metrics

    return metrics


def train_and_evaluate_distilbert(
    split: ValidationSplit,
    config: DistilBertExperimentConfig | None = None,
    *,
    tokenizer: Any = None,
    model: Any = None,
    device: torch.device | str | None = None,
) -> DistilBertRun:
    """Train DistilBERT on development-training and evaluate on validation.

    Accepts mock tokenizer, model, and device objects for synthetic unit testing
    without external network downloads.
    """
    if config is None:
        config = DistilBertExperimentConfig()

    development = split.development_train
    validation = split.validation
    _validate_partitions(development, validation)

    # Set seed for reproducibility
    set_deterministic_seed(config.random_seed)

    # Resolve device and mixed precision
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    use_fp16 = bool(config.fp16 and device.type == "cuda")

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    label_to_id, id_to_label = build_label_mappings(config.label_order)

    # Initialize tokenizer if not provided
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

    train_text_pair = (
        development["target"].tolist() if config.target_conditioned else None
    )
    val_text_pair = (
        validation["target"].tolist() if config.target_conditioned else None
    )

    # Tokenize text inputs (text only for Track A; paired with target for Track B)
    tokenize_train_kwargs: dict[str, Any] = {
        "padding": "max_length",
        "truncation": True,
        "max_length": config.max_length,
        "return_tensors": "pt",
    }
    if train_text_pair is not None:
        tokenize_train_kwargs["text_pair"] = train_text_pair

    tokenize_val_kwargs: dict[str, Any] = {
        "padding": "max_length",
        "truncation": True,
        "max_length": config.max_length,
        "return_tensors": "pt",
    }
    if val_text_pair is not None:
        tokenize_val_kwargs["text_pair"] = val_text_pair

    train_encodings = tokenizer(
        development["text"].tolist(),
        **tokenize_train_kwargs,
    )
    train_labels = [label_to_id[label] for label in development["label"].tolist()]
    train_dataset = StanceTextDataset(train_encodings, train_labels)

    val_encodings = tokenizer(
        validation["text"].tolist(),
        **tokenize_val_kwargs,
    )
    val_dataset = StanceTextDataset(val_encodings)

    # DataLoader with deterministic generator
    data_gen = torch.Generator()
    data_gen.manual_seed(config.random_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        generator=data_gen,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.per_device_eval_batch_size,
        shuffle=False,
    )

    # Initialize model if not provided
    created_model_locally = False
    if model is None:
        created_model_locally = True
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name_or_path,
            num_labels=len(config.label_order),
            id2label=id_to_label,
            label2id=label_to_id,
        )
    model.to(device)

    # CUDA Feasibility Check on initial batch
    sample_batch = next(iter(train_loader))
    feasibility_ok = check_cuda_feasibility(
        model=model,
        sample_batch=sample_batch,
        device=device,
        use_fp16=use_fp16,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    if not feasibility_ok:
        if not created_model_locally or device.type != "cuda":
            raise RuntimeError("CUDA memory feasibility check failed on initial batch.")
        # Execute documented fallback tiers strictly preserving effective batch size
        tier_succeeded = False
        for fallback_batch_size, fallback_grad_accum in FALLBACK_TIERS:
            if (fallback_batch_size, fallback_grad_accum) == (
                config.per_device_train_batch_size,
                config.gradient_accumulation_steps,
            ):
                continue
            del model
            torch.cuda.empty_cache()
            set_deterministic_seed(config.random_seed)
            model = AutoModelForSequenceClassification.from_pretrained(
                config.model_name_or_path,
                num_labels=len(config.label_order),
                id2label=id_to_label,
                label2id=label_to_id,
            )
            model.to(device)
            data_gen = torch.Generator()
            data_gen.manual_seed(config.random_seed)
            train_loader = DataLoader(
                train_dataset,
                batch_size=fallback_batch_size,
                shuffle=True,
                generator=data_gen,
            )
            sample_batch = next(iter(train_loader))
            if check_cuda_feasibility(
                model=model,
                sample_batch=sample_batch,
                device=device,
                use_fp16=use_fp16,
                gradient_accumulation_steps=fallback_grad_accum,
            ):
                config = DistilBertExperimentConfig(
                    model_name_or_path=config.model_name_or_path,
                    random_seed=config.random_seed,
                    max_length=config.max_length,
                    per_device_train_batch_size=fallback_batch_size,
                    per_device_eval_batch_size=config.per_device_eval_batch_size,
                    gradient_accumulation_steps=fallback_grad_accum,
                    learning_rate=config.learning_rate,
                    weight_decay=config.weight_decay,
                    num_train_epochs=config.num_train_epochs,
                    warmup_ratio=config.warmup_ratio,
                    fp16=config.fp16,
                    label_order=config.label_order,
                    small_support_threshold=config.small_support_threshold,
                    target_conditioned=config.target_conditioned,
                )
                tier_succeeded = True
                break
        if not tier_succeeded:
            raise RuntimeError("All CUDA memory fallback tiers exhausted.")

    # Optimizer with weight decay grouping
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=config.learning_rate
    )

    total_update_steps = (
        math.ceil(len(train_loader) / config.gradient_accumulation_steps)
        * config.num_train_epochs
    )
    warmup_steps = int(total_update_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    # Deterministic training loop
    start_time = time.perf_counter()
    model.train()
    for _ in range(config.num_train_epochs):
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
                outputs = model(**batch)
                loss = outputs.loss
                if config.gradient_accumulation_steps > 1:
                    loss = loss / config.gradient_accumulation_steps

            scaler.scale(loss).backward()

            is_accum_step = (step + 1) % config.gradient_accumulation_steps == 0
            is_last_step = (step + 1) == len(train_loader)
            if is_accum_step or is_last_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

    runtime_seconds = float(time.perf_counter() - start_time)
    peak_gpu_memory_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        if torch.cuda.is_available() and device.type == "cuda"
        else None
    )

    # Evaluation loop on validation partition
    model.eval()
    all_preds: list[int] = []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
                outputs = model(**batch)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().tolist())

    predictions = [id_to_label[p] for p in all_preds]

    validation_records: list[dict[str, Any]] = []
    has_source_id = "source_id" in validation.columns
    for (idx, row), pred in zip(validation.iterrows(), predictions):
        record: dict[str, Any] = {
            "row_index": int(idx),
            "target": str(row["target"]),
            "true_label": str(row["label"]),
            "predicted_label": str(pred),
        }
        if has_source_id and pd.notna(row["source_id"]):
            record["source_id"] = str(row["source_id"])
        validation_records.append(record)

    split_report_dict = (
        asdict(split.report) if hasattr(split, "report") and split.report else None
    )
    metrics = _build_evaluation_metrics(
        development=development,
        validation=validation,
        predictions=predictions,
        config=config,
        device=device,
        use_fp16=use_fp16,
        split_report=split_report_dict,
    )
    metrics["runtime_seconds"] = runtime_seconds
    metrics["peak_gpu_memory_mb"] = peak_gpu_memory_mb
    metrics["cuda_feasibility_passed"] = feasibility_ok

    run_metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "detected_gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "fp16_enabled": use_fp16,
        "runtime_seconds": runtime_seconds,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "cuda_feasibility_passed": feasibility_ok,
        "config": config.to_dict(),
    }

    return DistilBertRun(
        config=config,
        metrics=metrics,
        run_metadata=run_metadata,
        predictions=predictions,
        validation_predictions=validation_records,
        model=model,
        tokenizer=tokenizer,
    )


def write_distilbert_outputs(
    run: DistilBertRun,
    output_dir: Path,
    *,
    save_predictions: bool | None = None,
    save_model: bool = True,
) -> None:
    """Save aggregate metrics and run metadata; optionally save privacy-safe row predictions and model weights."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_text = json.dumps(run.metrics, indent=2, sort_keys=True) + "\n"
    metadata_text = json.dumps(run.run_metadata, indent=2, sort_keys=True) + "\n"

    # Privacy safety guard: verify that no tweet text fields are exported
    for forbidden_key in ('"text":', '"text_excerpt":', '"Tweet":'):
        if forbidden_key in metrics_text or forbidden_key in metadata_text:
            raise ValueError(
                f"Privacy violation: Forbidden key {forbidden_key} found in output payload."
            )

    (output_dir / "validation_metrics.json").write_text(
        metrics_text, encoding="utf-8"
    )
    (output_dir / "run_metadata.json").write_text(
        metadata_text, encoding="utf-8"
    )

    should_save_preds = (
        save_predictions
        if save_predictions is not None
        else getattr(run.config, "save_predictions", False)
    )
    if should_save_preds and run.validation_predictions is not None:
        preds_text = json.dumps(run.validation_predictions, indent=2) + "\n"
        for forbidden_key in ('"text":', '"text_excerpt":', '"Tweet":'):
            if forbidden_key in preds_text:
                raise ValueError(
                    f"Privacy violation: Forbidden key {forbidden_key} found in validation predictions."
                )
        (output_dir / "validation_predictions.json").write_text(
            preds_text, encoding="utf-8"
        )

    if save_model and run.model is not None and hasattr(run.model, "save_pretrained"):
        run.model.save_pretrained(output_dir)
    if save_model and run.tokenizer is not None and hasattr(run.tokenizer, "save_pretrained"):
        run.tokenizer.save_pretrained(output_dir)


def run_official_distilbert_experiment(
    *,
    config_path: Path = Path("configs/distilbert_baseline.json"),
    raw_dir: Path = Path("data/raw/stance/semeval2016_task6"),
    output_dir: Path | None = None,
    save_predictions: bool | None = None,
    save_model: bool = True,
) -> DistilBertRun:
    """Run DistilBERT experiment without loading or accessing official test data."""
    config = DistilBertExperimentConfig.from_json(config_path)

    if output_dir is None:
        output_dir = (
            Path("experiments/stance_distilbert_target")
            if config.target_conditioned
            else Path("experiments/stance_distilbert")
        )

    # Ingestion: load official Task A training file spec only
    train_spec, _ = official_task_a_file_specs(raw_dir)
    data = load_task_a([train_spec], expected_targets=TASK_A_TARGETS)

    # Stratified 80/20 partition: 2,331 development-train, 583 validation
    split = prepare_task_a_validation_split(data)

    run = train_and_evaluate_distilbert(split, config=config)
    write_distilbert_outputs(
        run, output_dir, save_predictions=save_predictions, save_model=save_model
    )
    return run

