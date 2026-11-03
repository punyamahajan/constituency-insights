"""Tracked full fine-tuning and LoRA comparison using the shared training loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Sequence

import mlflow
from mlflow import MlflowClient
import pandas as pd
import psutil
import torch
import yaml
from transformers.optimization import get_linear_schedule_with_warmup

from constituency_insights.data import (
    TASK_A_TARGETS,
    ValidationSplit,
    load_task_a,
    official_task_a_file_specs,
    prepare_task_a_validation_split,
)
from constituency_insights.nlp.comparison import compare_validation_predictions
from constituency_insights.nlp.distilbert import LABEL_ORDER
from constituency_insights.nlp.model_factory import (
    ModelFactoryConfig,
    create_model_and_tokenizer,
    parameter_counts,
)
from constituency_insights.nlp.training import (
    StanceTrainingConfig,
    _build_loaders,
    evaluate,
    seed_everything,
    train_one_epoch,
)


@dataclass(frozen=True)
class FinetuningConfig:
    """One fully resolved full/LoRA comparison run."""

    name: str
    backbone: str
    tuning_method: str
    seed: int
    tracking_uri: str
    experiment_name: str
    raw_dir: Path
    output_dir: Path
    comparison_report: Path
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
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: tuple[str, ...] = ("query", "key", "value")

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "finetuning" / self.name / "best_checkpoint"

    @property
    def prediction_path(self) -> Path:
        return self.output_dir / "finetuning" / self.name / "validation_predictions.json"


@dataclass(frozen=True)
class FinetuningResult:
    run_id: str
    config: FinetuningConfig
    best_macro_f1: float
    predictions: list[dict[str, Any]]


class _MemoryMonitor:
    def __init__(self) -> None:
        self._process = psutil.Process()
        self.peak_bytes = self._process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(0.02):
            self.peak_bytes = max(self.peak_bytes, self._process.memory_info().rss)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_args):
        self._stop.set()
        self._thread.join()
        self.peak_bytes = max(self.peak_bytes, self._process.memory_info().rss)


def load_finetuning_configs(path: Path | str) -> list[FinetuningConfig]:
    """Load comparison variants from the repository's single experiment YAML."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    training = payload["training"]
    section = payload["finetuning_comparison"]
    configs: list[FinetuningConfig] = []
    for experiment in section["experiments"]:
        configs.append(FinetuningConfig(
            name=str(experiment["name"]), backbone=str(experiment["backbone"]),
            tuning_method=str(experiment["tuning_method"]), seed=int(payload["seed"]),
            tracking_uri=str(payload["tracking_uri"]),
            experiment_name=str(section["experiment_name"]), raw_dir=Path(payload["raw_dir"]),
            output_dir=Path(payload["output_dir"]),
            comparison_report=Path(section["comparison_report"]),
            max_length=int(training["max_length"]), train_batch_size=int(training["train_batch_size"]),
            eval_batch_size=int(training["eval_batch_size"]),
            gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
            learning_rate=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]),
            max_epochs=int(training["max_epochs"]), warmup_ratio=float(training["warmup_ratio"]),
            max_grad_norm=float(training["max_grad_norm"]),
            early_stopping_patience=int(training["early_stopping_patience"]),
            min_delta=float(training["min_delta"]), mixed_precision=bool(training["mixed_precision"]),
            imbalance_strategy=str(experiment["imbalance_strategy"]),
            focal_gamma=float(training["focal_gamma"]),
            target_conditioned=bool(training["target_conditioned"]),
            lora_rank=int(experiment.get("lora_rank", 8)),
            lora_alpha=int(experiment.get("lora_alpha", 16)),
            lora_dropout=float(experiment.get("lora_dropout", 0.1)),
            lora_target_modules=tuple(experiment.get("lora_target_modules", ["query", "key", "value"])),
        ))
    return configs


def _loader_config(config: FinetuningConfig) -> StanceTrainingConfig:
    """Adapt shared loader settings without creating a second input pipeline."""

    return StanceTrainingConfig(
        name=config.name, seed=config.seed, tracking_uri=config.tracking_uri,
        experiment_name=config.experiment_name, raw_dir=config.raw_dir,
        output_dir=config.output_dir, comparison_report=config.comparison_report,
        model_name_or_path=config.backbone, max_length=config.max_length,
        train_batch_size=config.train_batch_size, eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate, weight_decay=config.weight_decay,
        max_epochs=config.max_epochs, warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        early_stopping_patience=config.early_stopping_patience, min_delta=config.min_delta,
        mixed_precision=config.mixed_precision, imbalance_strategy=config.imbalance_strategy,
        focal_gamma=config.focal_gamma, target_conditioned=config.target_conditioned,
    )


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _save_best(model: Any, tokenizer: Any, config: FinetuningConfig) -> None:
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.checkpoint_dir)
    tokenizer.save_pretrained(config.checkpoint_dir)


def train_finetuning_run(
    config: FinetuningConfig,
    *,
    split: ValidationSplit | None = None,
    device: str | torch.device | None = None,
) -> FinetuningResult:
    """Train one factory model and log resource/performance metrics to MLflow."""

    seed_everything(config.seed)
    if split is None:
        train_spec, _unused_test_spec = official_task_a_file_specs(config.raw_dir)
        data = load_task_a([train_spec], expected_targets=TASK_A_TARGETS)
        split = prepare_task_a_validation_split(data, random_seed=config.seed)
    if not split.development_train["split"].eq("train").all() or not split.validation["split"].eq("train").all():
        raise ValueError("Fine-tuning accepts only official-training-derived partitions.")

    model, tokenizer = create_model_and_tokenizer(ModelFactoryConfig(
        backbone=config.backbone, tuning_method=config.tuning_method,
        lora_rank=config.lora_rank, lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout, lora_target_modules=config.lora_target_modules,
    ))
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(target_device)
    trainable, total = parameter_counts(model)
    train_loader, validation_loader, class_weights = _build_loaders(split, tokenizer, _loader_config(config))
    class_weights = class_weights.to(target_device)
    loss_weights = class_weights if config.imbalance_strategy in {"weighted", "focal"} else None
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(steps_per_epoch * config.max_epochs * config.warmup_ratio),
        steps_per_epoch * config.max_epochs,
    )
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    best_f1 = -math.inf
    best_epoch = 0
    best_metrics: dict[str, Any] = {}
    best_predictions: list[int] = []
    stale_epochs = 0
    epoch_times: list[float] = []
    with _MemoryMonitor() as memory, mlflow.start_run(run_name=config.name) as run:
        mlflow.log_params({
            **{key: value for key, value in asdict(config).items() if isinstance(value, (str, int, float, bool))},
            "lora_target_modules": ",".join(config.lora_target_modules),
            "trainable_parameters": trainable, "total_parameters": total,
        })
        for epoch in range(1, config.max_epochs + 1):
            started = time.perf_counter()
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
            elapsed = time.perf_counter() - started
            epoch_times.append(elapsed)
            mlflow.log_metrics({
                "train_loss": train_loss, "val_loss": metrics["loss"],
                "val_accuracy": metrics["accuracy"], "val_macro_f1": metrics["macro_f1"],
                "epoch_wall_seconds": elapsed,
            }, step=epoch)
            if metrics["macro_f1"] > best_f1 + config.min_delta:
                best_f1, best_epoch, best_metrics = metrics["macro_f1"], epoch, metrics
                best_predictions = list(metrics["predictions"])
                stale_epochs = 0
                _save_best(model, tokenizer, config)
            else:
                stale_epochs += 1
                if stale_epochs >= config.early_stopping_patience:
                    break

        labels = list(LABEL_ORDER)
        prediction_records = [
            {"row_index": int(index), "target": str(row["target"]),
             "true_label": str(row["label"]), "predicted_label": labels[prediction]}
            for (index, row), prediction in zip(split.validation.iterrows(), best_predictions)
        ]
        config.prediction_path.parent.mkdir(parents=True, exist_ok=True)
        config.prediction_path.write_text(json.dumps(prediction_records, indent=2) + "\n", encoding="utf-8")
        confusion_path = config.prediction_path.parent / "confusion_matrix.json"
        confusion_path.write_text(json.dumps({"label_order": labels, "matrix": best_metrics["confusion_matrix"]}, indent=2) + "\n", encoding="utf-8")
        checkpoint_bytes = _directory_size(config.checkpoint_dir)
        gpu_peak = torch.cuda.max_memory_allocated(target_device) if target_device.type == "cuda" else 0
        mlflow.log_metrics({
            "best_epoch": best_epoch, "best_val_accuracy": best_metrics["accuracy"],
            "best_val_macro_f1": best_metrics["macro_f1"],
            "mean_epoch_wall_seconds": sum(epoch_times) / len(epoch_times),
            "peak_gpu_memory_mb": gpu_peak / (1024 ** 2),
            "peak_cpu_memory_mb": memory.peak_bytes / (1024 ** 2),
            "checkpoint_size_mb": checkpoint_bytes / (1024 ** 2),
            "trainable_parameters": trainable,
        })
        mlflow.log_artifact(str(confusion_path), artifact_path="evaluation")
        mlflow.set_tags({"data_scope": "official_train_derived_validation_only", "backbone": config.backbone, "tuning_method": config.tuning_method})
        run_id = run.info.run_id
    return FinetuningResult(run_id, config, best_f1, prediction_records)


def generate_finetuning_report(
    configs: Sequence[FinetuningConfig],
    *,
    baseline_predictions_path: Path = Path("experiments/stance_distilbert_target/validation_predictions.json"),
) -> Path:
    """Generate the table from MLflow and paired test from saved predictions."""

    if not configs:
        raise ValueError("No fine-tuning configs supplied.")
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(configs[0].tracking_uri)
    experiment = mlflow.get_experiment_by_name(configs[0].experiment_name)
    if experiment is None:
        raise RuntimeError("Fine-tuning MLflow experiment does not exist.")
    runs = mlflow.search_runs([experiment.experiment_id], filter_string="attributes.status = 'FINISHED'", order_by=["attributes.start_time DESC"])
    latest: dict[str, pd.Series] = {}
    for _, row in runs.iterrows():
        name = str(row.get("tags.mlflow.runName"))
        if name in {config.name for config in configs} and name not in latest:
            latest[name] = row
    missing = {config.name for config in configs}.difference(latest)
    if missing:
        raise RuntimeError(f"Missing MLflow runs: {sorted(missing)}")

    new_names = [config.name for config in configs if config.backbone == "xlm-roberta-base"]
    best_name = max(new_names, key=lambda name: float(latest[name]["metrics.best_val_macro_f1"]))
    best_config = next(config for config in configs if config.name == best_name)
    baseline = json.loads(baseline_predictions_path.read_text(encoding="utf-8"))
    candidate = json.loads(best_config.prediction_path.read_text(encoding="utf-8"))
    paired = compare_validation_predictions(baseline, candidate, candidate_name=best_name)
    mcnemar = paired["mcnemar_test"]

    lines = [
        "# Full Fine-Tuning and LoRA Comparison", "",
        "Generated from MLflow metrics on the fixed validation partition derived exclusively from the official SemEval training split.", "",
        "SemEval-2016 Task 6 is English-only. These experiments do not provide evidence of cross-lingual transfer to Hindi, Tamil, code-mixed text, or Indian constituency discourse.", "",
        "| Model | Method | Accuracy | Macro-F1 | Trainable parameters | Peak GPU MB | Peak CPU MB | Seconds/epoch | Checkpoint MB |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for config in configs:
        row = latest[config.name]
        lines.append(
            f"| {config.backbone} | {config.tuning_method} | {float(row['metrics.best_val_accuracy']):.6f} | "
            f"{float(row['metrics.best_val_macro_f1']):.6f} | {int(float(row['metrics.trainable_parameters'])):,} | "
            f"{float(row['metrics.peak_gpu_memory_mb']):.2f} | {float(row['metrics.peak_cpu_memory_mb']):.2f} | "
            f"{float(row['metrics.mean_epoch_wall_seconds']):.2f} | {float(row['metrics.checkpoint_size_mb']):.2f} |"
        )
    lines += [
        "", "## Paired McNemar comparison", "",
        f"Best new model: `{best_name}`. Compared with the existing target-conditioned DistilBERT predictions on the identical validation rows.", "",
        f"- Baseline-only correct: {mcnemar['baseline_only_correct']}",
        f"- New-model-only correct: {mcnemar['candidate_only_correct']}",
        f"- Continuity-corrected statistic: {mcnemar['statistic_continuity_corrected']:.6f}",
        f"- Continuity-corrected p-value: {mcnemar['p_value_continuity_corrected']:.6g}",
        "", "## Analysis", "",
        "**TODO (author): Add interpretation of performance, efficiency, and practical trade-offs here.**", "",
    ]
    report = configs[0].comparison_report
    report.write_text("\n".join(lines), encoding="utf-8")
    return report
