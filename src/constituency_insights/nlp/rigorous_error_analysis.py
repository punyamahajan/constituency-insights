"""Rigorous validation-only error and calibration analysis."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from constituency_insights.data import (
    TASK_A_TARGETS,
    load_task_a,
    official_task_a_file_specs,
    prepare_task_a_validation_split,
)
from constituency_insights.nlp.distilbert import LABEL_ORDER, StanceTextDataset


_NEGATION = re.compile(r"(?i)\b(?:no|not|never|none|neither|nor|cannot|can't|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't|without)\b")
_HASHTAG = re.compile(r"(?<!\w)#[A-Za-z0-9_]+")
_MENTION = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")


@dataclass(frozen=True)
class ErrorAnalysisConfig:
    seed: int
    raw_dir: Path
    checkpoint: Path
    local_error_cases: Path
    report: Path
    bootstrap_resamples: int
    confidence_level: float
    calibration_bins: int
    crossfit_folds: int
    inference_batch_size: int
    max_length: int
    very_short_token_threshold: int


def load_error_analysis_config(path: Path | str) -> ErrorAnalysisConfig:
    """Load analysis settings and seed from the repository's single YAML file."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    section = payload["error_analysis"]
    return ErrorAnalysisConfig(
        seed=int(payload["seed"]), raw_dir=Path(payload["raw_dir"]),
        checkpoint=Path(section["checkpoint"]),
        local_error_cases=Path(section["local_error_cases"]),
        report=Path(section["report"]),
        bootstrap_resamples=int(section["bootstrap_resamples"]),
        confidence_level=float(section["confidence_level"]),
        calibration_bins=int(section["calibration_bins"]),
        crossfit_folds=int(section["crossfit_folds"]),
        inference_batch_size=int(section["inference_batch_size"]),
        max_length=int(section["max_length"]),
        very_short_token_threshold=int(section["very_short_token_threshold"]),
    )


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Stable row-wise softmax with a positive temperature."""

    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    scaled = logits / temperature
    return np.exp(scaled - logsumexp(scaled, axis=1, keepdims=True))


def calibration_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bins: int,
) -> dict[str, Any]:
    """Compute top-label reliability data, ECE, and multiclass Brier score."""

    if bins < 2:
        raise ValueError("At least two calibration bins are required.")
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    reliability: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        count = int(mask.sum())
        mean_confidence = float(confidence[mask].mean()) if count else None
        accuracy = float(correct[mask].mean()) if count else None
        if count:
            ece += count / len(labels) * abs(float(mean_confidence) - float(accuracy))
        reliability.append({
            "bin": index + 1, "lower": float(lower), "upper": float(upper),
            "count": count, "mean_confidence": mean_confidence, "accuracy": accuracy,
        })
    one_hot = np.eye(probabilities.shape[1])[labels]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    return {"ece": float(ece), "brier_score": brier, "reliability": reliability}


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit one positive scalar temperature by minimizing multiclass NLL."""

    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        scaled = logits / temperature
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(labels)), labels]))

    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    if not result.success:
        raise RuntimeError("Temperature optimization failed.")
    return float(math.exp(result.x))


def cross_fitted_temperature_scaling(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    """Fit temperatures on K-1 folds and calibrate each held-out fold."""

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    probabilities = np.empty_like(logits, dtype=float)
    temperatures: list[float] = []
    for train_indices, heldout_indices in splitter.split(logits, labels):
        temperature = fit_temperature(logits[train_indices], labels[train_indices])
        probabilities[heldout_indices] = softmax(logits[heldout_indices], temperature)
        temperatures.append(temperature)
    return probabilities, temperatures


def heuristic_tags(text: str, *, very_short_threshold: int) -> tuple[str, ...]:
    """Assign deterministic surface-form tags without semantic interpretation."""

    tags: list[str] = []
    if _NEGATION.search(text):
        tags.append("contains_negation")
    if _HASHTAG.search(text):
        tags.append("contains_hashtag")
    if len(text.split()) <= very_short_threshold:
        tags.append("very_short")
    if "?" in text:
        tags.append("contains_question_mark")
    if _MENTION.search(text):
        tags.append("contains_mention")
    return tuple(tags)


def tag_error_rates(
    texts: Sequence[str],
    truth: np.ndarray,
    predictions: np.ndarray,
    *,
    very_short_threshold: int,
) -> tuple[dict[str, dict[str, Any]], list[tuple[str, ...]]]:
    """Compute conditional error rates for each heuristic tag."""

    row_tags = [
        heuristic_tags(text, very_short_threshold=very_short_threshold) for text in texts
    ]
    errors = truth != predictions
    names = (
        "contains_negation", "contains_hashtag", "very_short",
        "contains_question_mark", "contains_mention",
    )
    report: dict[str, dict[str, Any]] = {}
    for name in names:
        mask = np.array([name in tags for tags in row_tags])
        count = int(mask.sum())
        error_count = int(errors[mask].sum())
        report[name] = {
            "count": count, "errors": error_count,
            "error_rate": error_count / count if count else None,
        }
    return report, row_tags


def _metric_vector(
    truth: np.ndarray, predictions: np.ndarray, labels: Sequence[int]
) -> np.ndarray:
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predictions, labels=list(labels), zero_division=0
    )
    return np.column_stack((precision, recall, f1))


def metrics_with_bootstrap(
    truth: np.ndarray,
    predictions: np.ndarray,
    *,
    label_names: Sequence[str],
    resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, dict[str, dict[str, float]]]:
    """Return per-class metrics and seeded percentile bootstrap intervals."""

    labels = list(range(len(label_names)))
    point = _metric_vector(truth, predictions, labels)
    rng = np.random.default_rng(seed)
    samples = np.empty((resamples, len(labels), 3), dtype=float)
    for sample_index in range(resamples):
        indices = rng.integers(0, len(truth), size=len(truth))
        samples[sample_index] = _metric_vector(truth[indices], predictions[indices], labels)
    alpha = (1.0 - confidence_level) / 2.0
    lower = np.quantile(samples, alpha, axis=0)
    upper = np.quantile(samples, 1.0 - alpha, axis=0)
    metric_names = ("precision", "recall", "f1")
    return {
        label_name: {
            metric: {"value": float(point[i, j]), "lower": float(lower[i, j]), "upper": float(upper[i, j])}
            for j, metric in enumerate(metric_names)
        }
        for i, label_name in enumerate(label_names)
    }


def infer_validation(
    config: ErrorAnalysisConfig,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load official training only, recreate validation, and return logits."""

    train_spec, _unused_test_spec = official_task_a_file_specs(config.raw_dir)
    data = load_task_a([train_spec], expected_targets=TASK_A_TARGETS)
    split = prepare_task_a_validation_split(data, random_seed=config.seed)
    validation = split.validation
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.checkpoint, local_files_only=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    encodings = tokenizer(
        validation["text"].tolist(), text_pair=validation["target"].tolist(),
        padding="max_length", truncation=True, max_length=config.max_length,
        return_tensors="pt",
    )
    loader = DataLoader(StanceTextDataset(encodings), batch_size=config.inference_batch_size)
    logits: list[list[float]] = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(**{key: value.to(device) for key, value in batch.items()})
            logits.extend(outputs.logits.float().cpu().tolist())
    return validation, np.asarray(logits)


def _format_interval(metric: dict[str, float]) -> str:
    return f"{metric['value']:.4f} [{metric['lower']:.4f}, {metric['upper']:.4f}]"


def run_error_analysis(config: ErrorAnalysisConfig) -> Path:
    """Generate the ignored local cases and privacy-safe committed report."""

    validation, logits = infer_validation(config)
    label_to_id = {label: index for index, label in enumerate(LABEL_ORDER)}
    truth = np.array([label_to_id[label] for label in validation["label"]])
    probabilities = softmax(logits)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)

    overall = metrics_with_bootstrap(
        truth, predictions, label_names=LABEL_ORDER,
        resamples=config.bootstrap_resamples,
        confidence_level=config.confidence_level, seed=config.seed,
    )
    by_target: dict[str, Any] = {}
    for offset, target in enumerate(sorted(validation["target"].unique())):
        mask = validation["target"].eq(target).to_numpy()
        by_target[target] = metrics_with_bootstrap(
            truth[mask], predictions[mask], label_names=LABEL_ORDER,
            resamples=config.bootstrap_resamples,
            confidence_level=config.confidence_level, seed=config.seed + offset + 1,
        )

    tags, row_tags = tag_error_rates(
        validation["text"].tolist(), truth, predictions,
        very_short_threshold=config.very_short_token_threshold,
    )
    wrong_positions = np.flatnonzero(truth != predictions)
    ranked_positions = wrong_positions[np.argsort(-confidence[wrong_positions])][:50]
    local_rows: list[dict[str, Any]] = []
    safe_rows: list[dict[str, Any]] = []
    for position in ranked_positions:
        index = validation.index[position]
        common = {
            "source_row_index": int(index),
            "target": str(validation.iloc[position]["target"]),
            "true_label": LABEL_ORDER[truth[position]],
            "predicted_label": LABEL_ORDER[predictions[position]],
            "confidence": float(confidence[position]),
        }
        safe_rows.append(common)
        local_rows.append({
            **common, "text": str(validation.iloc[position]["text"]),
            "heuristic_tags": ",".join(row_tags[position]),
        })
    config.local_error_cases.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(local_rows).to_csv(config.local_error_cases, index=False, encoding="utf-8")

    uncalibrated = calibration_metrics(
        probabilities, truth, bins=config.calibration_bins
    )
    fitted_temperature = fit_temperature(logits, truth)
    same_split = calibration_metrics(
        softmax(logits, fitted_temperature), truth, bins=config.calibration_bins
    )
    crossfit_probabilities, fold_temperatures = cross_fitted_temperature_scaling(
        logits, truth, folds=config.crossfit_folds, seed=config.seed
    )
    crossfit = calibration_metrics(
        crossfit_probabilities, truth, bins=config.calibration_bins
    )

    lines = [
        "# Validation Error and Calibration Analysis", "",
        "Generated from the fixed validation partition derived only from the official SemEval training split. The official test split was not accessed.", "",
        f"All confidence intervals below are seeded {int(config.confidence_level * 100)}% percentile bootstrap intervals from {config.bootstrap_resamples:,} resamples.", "",
        "## Per-class metrics", "",
        "| Class | Precision [CI] | Recall [CI] | F1 [CI] |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label in LABEL_ORDER:
        lines.append(f"| {label} | {_format_interval(overall[label]['precision'])} | {_format_interval(overall[label]['recall'])} | {_format_interval(overall[label]['f1'])} |")
    lines += ["", "## Per-target metrics", "", "| Target | Class | Precision [CI] | Recall [CI] | F1 [CI] |", "| --- | --- | ---: | ---: | ---: |"]
    for target, target_metrics in by_target.items():
        for label in LABEL_ORDER:
            lines.append(f"| {target} | {label} | {_format_interval(target_metrics[label]['precision'])} | {_format_interval(target_metrics[label]['recall'])} | {_format_interval(target_metrics[label]['f1'])} |")
    lines += ["", "## Highest-confidence errors", "", "Raw text for these cases is stored only in the Git-ignored local file `outputs/local/error_cases.csv`.", "", "| Index | Target | True label | Predicted label | Confidence |", "| ---: | --- | --- | --- | ---: |"]
    for row in safe_rows:
        lines.append(f"| {row['source_row_index']} | {row['target']} | {row['true_label']} | {row['predicted_label']} | {row['confidence']:.6f} |")
    lines += ["", "## Heuristic tag error rates", "", "Tags are overlapping surface-form indicators; rates are conditional on a row having the tag.", "", "| Tag | Tagged rows | Errors | Error rate |", "| --- | ---: | ---: | ---: |"]
    for tag, values in tags.items():
        rate = "N/A" if values["error_rate"] is None else f"{values['error_rate']:.4f}"
        lines.append(f"| {tag} | {values['count']} | {values['errors']} | {rate} |")
    lines += [
        "", "## Calibration", "",
        f"- Uncalibrated ECE: {uncalibrated['ece']:.6f}",
        f"- Uncalibrated multiclass Brier score: {uncalibrated['brier_score']:.6f}",
        f"- Same-split fitted temperature: {fitted_temperature:.6f}",
        f"- Same-split temperature-scaled ECE: {same_split['ece']:.6f}",
        f"- Same-split temperature-scaled Brier score: {same_split['brier_score']:.6f}",
        "", "> **Warning:** The same-split temperature is fitted and evaluated on this validation set, so its calibration estimates are optimistically biased and must not be treated as held-out evidence.",
        "", f"The preferred estimate uses {config.crossfit_folds}-fold cross-fitting: each row is calibrated by a temperature fitted without that row's fold.", "",
        f"- Cross-fitted temperatures: {', '.join(f'{value:.6f}' for value in fold_temperatures)}",
        f"- Cross-fitted ECE: {crossfit['ece']:.6f}",
        f"- Cross-fitted multiclass Brier score: {crossfit['brier_score']:.6f}",
        "", "### Reliability diagram data", "",
        "| Variant | Bin | Range | Count | Mean confidence | Accuracy |", "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for variant, metrics in (("uncalibrated", uncalibrated), ("same_split_temperature", same_split), ("cross_fitted_temperature", crossfit)):
        for row in metrics["reliability"]:
            mean_confidence = "N/A" if row["mean_confidence"] is None else f"{row['mean_confidence']:.6f}"
            accuracy = "N/A" if row["accuracy"] is None else f"{row['accuracy']:.6f}"
            lines.append(f"| {variant} | {row['bin']} | [{row['lower']:.1f}, {row['upper']:.1f}] | {row['count']} | {mean_confidence} | {accuracy} |")
    lines += ["", "## Manual qualitative notes", "", "**TODO (author): Add qualitative observations after reviewing the local error-case CSV.**", ""]
    config.report.parent.mkdir(parents=True, exist_ok=True)
    config.report.write_text("\n".join(lines), encoding="utf-8")
    return config.report
