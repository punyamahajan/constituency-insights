"""Explicit, reproducible PyTorch training for stance classification."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from constituency_insights.data import (
    TASK_A_TARGETS,
    ValidationSplit,
    load_task_a,
    official_task_a_file_specs,
    prepare_task_a_validation_split,
)
from constituency_insights.nlp.distilbert import LABEL_ORDER, StanceTextDataset


IMBALANCE_STRATEGIES = frozenset({"none", "weighted", "focal"})


@dataclass(frozen=True)
class StanceTrainingConfig:
    """Fully resolved settings for one training run."""

    name: str
    seed: int
    tracking_uri: str
    experiment_name: str
    raw_dir: Path
    output_dir: Path
    comparison_report: Path
    model_name_or_path: str
    max_length: int
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    warmup_ratio: float
    max_grad_norm: float
    early_stopping_patience: int
    min_delta: float
    mixed_precision: bool
    imbalance_strategy: str
    focal_gamma: float
    target_conditioned: bool

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / self.name / "best_checkpoint"


@dataclass(frozen=True)
class TrainingResult:
    """Privacy-safe outputs from one tracked training run."""

    run_id: str
    run_name: str
    best_epoch: int
    best_macro_f1: float
    final_accuracy: float
    final_macro_f1: float
    confusion_matrix: list[list[int]]
    checkpoint_dir: Path


def load_training_configs(path: Path | str) -> list[StanceTrainingConfig]:
    """Load shared settings and experiment variants from one YAML file."""

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Training config must contain a YAML mapping.")
    training = payload.get("training")
    experiments = payload.get("experiments")
    if not isinstance(training, dict) or not isinstance(experiments, list):
        raise ValueError("Config requires 'training' mapping and 'experiments' list.")

    required_root = {
        "seed", "tracking_uri", "experiment_name", "raw_dir", "output_dir",
        "comparison_report",
    }
    required_training = {
        "model_name_or_path", "max_length", "train_batch_size", "eval_batch_size",
        "gradient_accumulation_steps", "learning_rate", "weight_decay", "max_epochs",
        "warmup_ratio", "max_grad_norm", "early_stopping_patience", "min_delta",
        "mixed_precision", "focal_gamma", "target_conditioned",
    }
    missing = sorted(required_root.difference(payload))
    missing += [f"training.{key}" for key in sorted(required_training.difference(training))]
    if missing:
        raise ValueError(f"Missing required YAML settings: {missing}")

    configs: list[StanceTrainingConfig] = []
    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise ValueError("Each experiment must be a YAML mapping.")
        strategy = str(experiment.get("imbalance_strategy", ""))
        if strategy not in IMBALANCE_STRATEGIES:
            raise ValueError(f"Unsupported imbalance strategy: {strategy!r}")
        configs.append(
            StanceTrainingConfig(
                name=str(experiment["name"]),
                seed=int(payload["seed"]),
                tracking_uri=str(payload["tracking_uri"]),
                experiment_name=str(payload["experiment_name"]),
                raw_dir=Path(payload["raw_dir"]),
                output_dir=Path(payload["output_dir"]),
                comparison_report=Path(payload["comparison_report"]),
                model_name_or_path=str(training["model_name_or_path"]),
                max_length=int(training["max_length"]),
                train_batch_size=int(training["train_batch_size"]),
                eval_batch_size=int(training["eval_batch_size"]),
                gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
                learning_rate=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
                max_epochs=int(training["max_epochs"]),
                warmup_ratio=float(training["warmup_ratio"]),
                max_grad_norm=float(training["max_grad_norm"]),
                early_stopping_patience=int(training["early_stopping_patience"]),
                min_delta=float(training["min_delta"]),
                mixed_precision=bool(training["mixed_precision"]),
                imbalance_strategy=strategy,
                focal_gamma=float(training["focal_gamma"]),
                target_conditioned=bool(training["target_conditioned"]),
            )
        )
    if not configs:
        raise ValueError("At least one experiment must be configured.")
    return configs


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and request deterministic kernels."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float = 2.0,
    class_weights: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute multi-class focal loss from logits without external packages."""

    if gamma < 0:
        raise ValueError("gamma must be non-negative.")
    log_probabilities = F.log_softmax(logits, dim=-1)
    rows = torch.arange(targets.shape[0], device=targets.device)
    target_log_probabilities = log_probabilities[rows, targets]
    target_probabilities = target_log_probabilities.exp()
    losses = -((1.0 - target_probabilities) ** gamma) * target_log_probabilities
    if class_weights is not None:
        losses = losses * class_weights.to(logits.device)[targets]
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    raise ValueError(f"Unsupported reduction: {reduction!r}")


def compute_class_weights(labels: Sequence[int], num_classes: int) -> torch.Tensor:
    """Return balanced inverse-frequency weights with mean weight one."""

    counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float()
    if torch.any(counts == 0):
        raise ValueError("Cannot compute class weights when a class is absent.")
    weights = counts.sum() / (num_classes * counts)
    return weights / weights.mean()


def _loss_for_strategy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    strategy: str,
    class_weights: torch.Tensor | None,
    focal_gamma: float,
) -> torch.Tensor:
    if strategy == "none":
        return F.cross_entropy(logits, labels)
    if strategy == "weighted":
        return F.cross_entropy(logits, labels, weight=class_weights)
    if strategy == "focal":
        return focal_loss(logits, labels, gamma=focal_gamma, class_weights=class_weights)
    raise ValueError(f"Unsupported imbalance strategy: {strategy!r}")


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    device: torch.device,
    strategy: str,
    class_weights: torch.Tensor | None,
    focal_gamma: float,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    mixed_precision: bool,
) -> float:
    """Train one epoch with accumulation, clipping, and optional CUDA AMP."""

    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive.")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision and device.type == "cuda")
    total_loss = 0.0
    for step, batch in enumerate(loader):
        batch = {key: value.to(device) for key, value in batch.items()}
        labels = batch.pop("labels")
        with _autocast_context(device, mixed_precision):
            logits = model(**batch).logits
            raw_loss = _loss_for_strategy(
                logits, labels, strategy, class_weights, focal_gamma
            )
            loss = raw_loss / gradient_accumulation_steps
        scaler.scale(loss).backward()
        total_loss += float(raw_loss.detach().cpu())
        should_step = (step + 1) % gradient_accumulation_steps == 0 or step + 1 == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
    return total_loss / max(1, len(loader))


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    strategy: str,
    class_weights: torch.Tensor | None,
    focal_gamma: float,
    mixed_precision: bool,
) -> dict[str, Any]:
    """Evaluate without retaining or serializing input text."""

    model.eval()
    losses: list[float] = []
    predictions: list[int] = []
    truths: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            with _autocast_context(device, mixed_precision):
                logits = model(**batch).logits
                loss = _loss_for_strategy(
                    logits, labels, strategy, class_weights, focal_gamma
                )
            losses.append(float(loss.cpu()))
            predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            truths.extend(labels.cpu().tolist())
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(truths, predictions)),
        "macro_f1": float(f1_score(truths, predictions, average="macro", zero_division=0)),
        "predictions": predictions,
        "truths": truths,
        "confusion_matrix": confusion_matrix(
            truths, predictions, labels=list(range(len(LABEL_ORDER)))
        ).astype(int).tolist(),
    }


def _build_loaders(
    split: ValidationSplit,
    tokenizer: Any,
    config: StanceTrainingConfig,
) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    label_to_id = {label: index for index, label in enumerate(LABEL_ORDER)}

    def encode(frame: pd.DataFrame) -> StanceTextDataset:
        kwargs: dict[str, Any] = {
            "padding": "max_length", "truncation": True,
            "max_length": config.max_length, "return_tensors": "pt",
        }
        if config.target_conditioned:
            kwargs["text_pair"] = frame["target"].tolist()
        encodings = tokenizer(frame["text"].tolist(), **kwargs)
        labels = [label_to_id[label] for label in frame["label"]]
        return StanceTextDataset(encodings, labels)

    train_labels = [label_to_id[label] for label in split.development_train["label"]]
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        encode(split.development_train), batch_size=config.train_batch_size,
        shuffle=True, generator=generator,
    )
    validation_loader = DataLoader(
        encode(split.validation), batch_size=config.eval_batch_size, shuffle=False,
    )
    return train_loader, validation_loader, compute_class_weights(train_labels, len(LABEL_ORDER))


def _save_checkpoint(model: Any, tokenizer: Any, config: StanceTrainingConfig) -> None:
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.checkpoint_dir)
    tokenizer.save_pretrained(config.checkpoint_dir)
    (config.checkpoint_dir / "training_config.json").write_text(
        json.dumps({**asdict(config), "raw_dir": str(config.raw_dir),
                    "output_dir": str(config.output_dir),
                    "comparison_report": str(config.comparison_report)}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def train_stance_model(
    config: StanceTrainingConfig,
    *,
    split: ValidationSplit | None = None,
    model: Any | None = None,
    tokenizer: Any | None = None,
    device: str | torch.device | None = None,
) -> TrainingResult:
    """Train one configuration using official training rows only and track it."""

    seed_everything(config.seed)
    if split is None:
        train_spec, _unused_test_spec = official_task_a_file_specs(config.raw_dir)
        data = load_task_a([train_spec], expected_targets=TASK_A_TARGETS)
        split = prepare_task_a_validation_split(data, random_seed=config.seed)
    if not split.development_train["split"].eq("train").all() or not split.validation["split"].eq("train").all():
        raise ValueError("Training accepts only partitions derived from official training rows.")

    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.model_name_or_path, local_files_only=True)
    model = model or AutoModelForSequenceClassification.from_pretrained(
        config.model_name_or_path, num_labels=len(LABEL_ORDER),
        id2label={i: label for i, label in enumerate(LABEL_ORDER)},
        label2id={label: i for i, label in enumerate(LABEL_ORDER)},
        local_files_only=True,
    )
    model.to(target_device)
    train_loader, validation_loader, class_weights = _build_loaders(split, tokenizer, config)
    class_weights = class_weights.to(target_device)
    loss_weights = class_weights if config.imbalance_strategy in {"weighted", "focal"} else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    total_updates = updates_per_epoch * config.max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_updates * config.warmup_ratio), total_updates
    )

    if "://" not in config.tracking_uri:
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    best_f1 = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    best_metrics: dict[str, Any] = {}
    with mlflow.start_run(run_name=config.name) as active_run:
        mlflow.log_params({
            **{key: value for key, value in asdict(config).items() if isinstance(value, (str, int, float, bool))},
            "device": str(target_device),
            "effective_batch_size": config.train_batch_size * config.gradient_accumulation_steps,
        })
        for epoch in range(1, config.max_epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, device=target_device,
                strategy=config.imbalance_strategy, class_weights=loss_weights,
                focal_gamma=config.focal_gamma,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                max_grad_norm=config.max_grad_norm, mixed_precision=config.mixed_precision,
            )
            metrics = evaluate(
                model, validation_loader, device=target_device,
                strategy=config.imbalance_strategy, class_weights=loss_weights,
                focal_gamma=config.focal_gamma, mixed_precision=config.mixed_precision,
            )
            mlflow.log_metrics({
                "train_loss": train_loss, "val_loss": metrics["loss"],
                "val_accuracy": metrics["accuracy"], "val_macro_f1": metrics["macro_f1"],
            }, step=epoch)
            if metrics["macro_f1"] > best_f1 + config.min_delta:
                best_f1 = metrics["macro_f1"]
                best_epoch = epoch
                best_metrics = metrics
                epochs_without_improvement = 0
                _save_checkpoint(model, tokenizer, config)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.early_stopping_patience:
                    break

        artifact_dir = config.output_dir / config.name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        confusion_path = artifact_dir / "confusion_matrix.json"
        confusion_path.write_text(
            json.dumps({"label_order": list(LABEL_ORDER), "matrix": best_metrics["confusion_matrix"]}, indent=2) + "\n",
            encoding="utf-8",
        )
        mlflow.log_artifact(str(confusion_path), artifact_path="evaluation")
        mlflow.log_metrics({
            "best_epoch": float(best_epoch),
            "best_val_macro_f1": best_metrics["macro_f1"],
            "best_val_accuracy": best_metrics["accuracy"],
        })
        mlflow.set_tag("data_scope", "official_train_derived_validation_only")
        run_id = active_run.info.run_id

    return TrainingResult(
        run_id=run_id, run_name=config.name, best_epoch=best_epoch,
        best_macro_f1=best_metrics["macro_f1"], final_accuracy=best_metrics["accuracy"],
        final_macro_f1=best_metrics["macro_f1"],
        confusion_matrix=best_metrics["confusion_matrix"], checkpoint_dir=config.checkpoint_dir,
    )


def generate_comparison_from_mlflow(
    configs: Sequence[StanceTrainingConfig],
) -> Path:
    """Generate a Markdown comparison table exclusively from tracked run data."""

    if not configs:
        raise ValueError("At least one training config is required.")
    config = configs[0]
    if any(item.tracking_uri != config.tracking_uri for item in configs):
        raise ValueError("Comparison runs must use one MLflow tracking URI.")
    if any(item.experiment_name != config.experiment_name for item in configs):
        raise ValueError("Comparison runs must use one MLflow experiment.")
    if "://" not in config.tracking_uri:
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(config.tracking_uri)
    experiment = mlflow.get_experiment_by_name(config.experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment not found: {config.experiment_name}")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
    )
    latest: dict[str, Mapping[str, Any]] = {}
    expected_names = {item.name for item in configs}
    for _, row in runs.iterrows():
        name = row.get("tags.mlflow.runName")
        if name in expected_names and name not in latest:
            latest[str(name)] = row
    missing = sorted(expected_names.difference(latest))
    if missing:
        raise RuntimeError(f"Missing completed MLflow runs: {missing}")

    lines = [
        "# Class-Imbalance Strategy Comparison", "",
        "Generated from completed MLflow runs on the fixed validation partition derived only from the official training split.", "",
        "| Strategy | MLflow run ID | Best epoch | Validation accuracy | Validation macro-F1 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for name in sorted(expected_names):
        row = latest[name]
        strategy = str(row["params.imbalance_strategy"])
        lines.append(
            f"| {strategy} | `{row['run_id']}` | {int(float(row['metrics.best_epoch']))} | "
            f"{float(row['metrics.best_val_accuracy']):.6f} | {float(row['metrics.best_val_macro_f1']):.6f} |"
        )
    lines += ["", "The table is generated by `generate_comparison_from_mlflow`; values are not hand-entered.", ""]
    config.comparison_report.parent.mkdir(parents=True, exist_ok=True)
    config.comparison_report.write_text("\n".join(lines), encoding="utf-8")
    return config.comparison_report
