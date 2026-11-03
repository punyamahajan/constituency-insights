"""One-shot official SemEval test evaluation with immutable preflight gates.

This is the only repository entry point authorized to load the official test
split. It performs no tuning, threshold selection, or checkpoint comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
import mlflow
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from constituency_insights.data.semeval2016 import (
    TASK_A_TARGETS, load_task_a, official_task_a_file_specs,
)
from constituency_insights.data.splitting import prepare_task_a_validation_split
from constituency_insights.nlp.baseline import LABEL_ORDER, build_baseline_pipeline
from constituency_insights.nlp.comparison import compute_mcnemar_test
from constituency_insights.nlp.distilbert import StanceTextDataset
from constituency_insights.nlp.training import seed_everything


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_preflight(config_path: Path, supplied_hash: str, force: bool
                       ) -> tuple[dict[str, Any], str | None]:
    """Validate every irreversible-evaluation gate before any dataset access."""
    if not supplied_hash or len(supplied_hash) != 64:
        raise ValueError("--checkpoint-hash must be a 64-character SHA-256 digest")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("frozen") is not True:
        raise ValueError("Model config must explicitly contain frozen: true")
    required = {
        "selection_document", "checkpoint", "weights_file",
        "expected_checkpoint_sha256", "raw_dir", "lock_file", "report",
        "output_dir", "target_conditioned", "max_length", "batch_size",
        "bootstrap_resamples", "confidence_level", "seed", "model_name",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Frozen config is missing fields: {missing}")
    selection = Path(config["selection_document"])
    if not selection.is_file():
        raise FileNotFoundError(
            "Validation-only model selection document must exist before evaluation: "
            f"{selection}"
        )
    expected = str(config["expected_checkpoint_sha256"]).lower()
    supplied = supplied_hash.lower()
    if supplied != expected:
        raise ValueError("Provided checkpoint hash does not match the frozen config")
    weights = Path(config["checkpoint"]) / str(config["weights_file"])
    if not weights.is_file():
        raise FileNotFoundError(f"Frozen checkpoint weights are missing: {weights}")
    actual = sha256_file(weights)
    if actual != expected:
        raise ValueError("Actual checkpoint weights hash does not match the frozen hash")
    lock = Path(config["lock_file"])
    warning = None
    if lock.exists() and not force:
        raise RuntimeError(
            f"Official test evaluation is locked by {lock}. A second run requires --force."
        )
    if lock.exists() and force:
        warning = (
            "WARNING: --force overrode an existing final-test lock. This rerun must "
            "not be used for model selection, tuning, or cherry-picking."
        )
    return config, warning


def _predict_transformer(test: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    checkpoint = Path(config["checkpoint"])
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, local_files_only=True
    )
    kwargs: dict[str, Any] = {
        "padding": "max_length", "truncation": True,
        "max_length": int(config["max_length"]), "return_tensors": "pt",
    }
    if bool(config["target_conditioned"]):
        kwargs["text_pair"] = test["target"].tolist()
    encodings = tokenizer(test["text"].tolist(), **kwargs)
    loader = DataLoader(StanceTextDataset(encodings),
                        batch_size=int(config["batch_size"]), shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    predictions: list[str] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(**{key: value.to(device) for key, value in batch.items()}).logits
            ids = logits.argmax(dim=-1).cpu().tolist()
            predictions.extend(str(model.config.id2label[index]) for index in ids)
    return predictions


def classification_metrics(truth: Sequence[str], predictions: Sequence[str]
                           ) -> dict[str, Any]:
    report = classification_report(
        truth, predictions, labels=list(LABEL_ORDER), output_dict=True,
        zero_division=0,
    )
    per_class = {
        label: {"precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"])}
        for label in LABEL_ORDER
    }
    return {
        "rows": len(truth),
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(
            truth, predictions, labels=list(LABEL_ORDER), average="macro",
            zero_division=0,
        )),
        "official_f1_favor_against": float(
            (per_class["FAVOR"]["f1"] + per_class["AGAINST"]["f1"]) / 2
        ),
        "per_class": per_class,
    }


def bootstrap_confidence_intervals(
    truth: Sequence[str], predictions: Sequence[str], seed: int,
    resamples: int, confidence: float,
) -> dict[str, list[float]]:
    truth_array, prediction_array = np.asarray(truth), np.asarray(predictions)
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "accuracy": [], "macro_f1": [], "official_f1_favor_against": [],
        **{f"f1_{label.lower()}": [] for label in LABEL_ORDER},
    }
    for _ in range(resamples):
        indices = rng.integers(0, len(truth_array), len(truth_array))
        metrics = classification_metrics(truth_array[indices], prediction_array[indices])
        values["accuracy"].append(metrics["accuracy"])
        values["macro_f1"].append(metrics["macro_f1"])
        values["official_f1_favor_against"].append(
            metrics["official_f1_favor_against"]
        )
        for label in LABEL_ORDER:
            values[f"f1_{label.lower()}"].append(metrics["per_class"][label]["f1"])
    alpha = (1 - confidence) / 2
    return {name: [float(np.quantile(series, alpha)),
                   float(np.quantile(series, 1 - alpha))]
            for name, series in values.items()}


def evaluate_predictions(test: pd.DataFrame, predictions: Sequence[str],
                         config: dict[str, Any], seed_offset: int = 0
                         ) -> dict[str, Any]:
    result = classification_metrics(test.label, predictions)
    result["confidence_intervals"] = bootstrap_confidence_intervals(
        test.label, predictions, int(config["seed"]) + seed_offset,
        int(config["bootstrap_resamples"]), float(config["confidence_level"]),
    )
    result["by_target"] = {}
    for offset, target in enumerate(sorted(test.target.unique()), start=1):
        mask = test.target.eq(target).to_numpy()
        target_result = classification_metrics(
            test.loc[mask, "label"], np.asarray(predictions)[mask]
        )
        target_result["confidence_intervals"] = bootstrap_confidence_intervals(
            test.loc[mask, "label"], np.asarray(predictions)[mask],
            int(config["seed"]) + seed_offset + offset,
            int(config["bootstrap_resamples"]), float(config["confidence_level"]),
        )
        result["by_target"][target] = target_result
    return result


def generate_report(result: dict[str, Any], path: Path) -> None:
    warning = result.get("warning")
    lines = ["# Final official SemEval test results", "",
             "> Generated by `scripts/final_test_eval.py` after validation-only model selection and checkpoint-hash verification.", ""]
    if warning:
        lines += [f"> **{warning}**", ""]
    lines += ["This is the one-shot official SemEval-2016 Task 6A test evaluation. No post-test tuning or checkpoint selection is permitted.", "",
              "## Overall", "", "| Model | Accuracy | 95% CI | Macro-F1 | 95% CI | Official F1 (FAVOR/AGAINST) | 95% CI |", "|---|---:|---:|---:|---:|---:|---:|"]
    for key, label in (("final_model", "Frozen DistilBERT"), ("tfidf_baseline", "TF-IDF + logistic regression")):
        metrics = result[key]; ci = metrics["confidence_intervals"]
        lines.append(
            f"| {label} | {metrics['accuracy']:.4f} | [{ci['accuracy'][0]:.4f}, {ci['accuracy'][1]:.4f}] | "
            f"{metrics['macro_f1']:.4f} | [{ci['macro_f1'][0]:.4f}, {ci['macro_f1'][1]:.4f}] | "
            f"{metrics['official_f1_favor_against']:.4f} | [{ci['official_f1_favor_against'][0]:.4f}, {ci['official_f1_favor_against'][1]:.4f}] |"
        )
    for key, label in (("final_model", "Frozen DistilBERT"), ("tfidf_baseline", "TF-IDF baseline")):
        lines += ["", f"## {label}: per-class metrics", "", "| Class | Precision | Recall | F1 | F1 95% CI | Support |", "|---|---:|---:|---:|---:|---:|"]
        metrics = result[key]
        for class_label in LABEL_ORDER:
            row = metrics["per_class"][class_label]
            ci = metrics["confidence_intervals"][f"f1_{class_label.lower()}"]
            lines.append(f"| {class_label} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {row['support']} |")
        lines += ["", f"## {label}: per-target metrics", "", "| Target | Rows | Accuracy (95% CI) | Macro-F1 (95% CI) | Official F1 (95% CI) |", "|---|---:|---:|---:|---:|"]
        for target, row in metrics["by_target"].items():
            ci = row["confidence_intervals"]
            lines.append(
                f"| {target} | {row['rows']} | {row['accuracy']:.4f} "
                f"[{ci['accuracy'][0]:.4f}, {ci['accuracy'][1]:.4f}] | "
                f"{row['macro_f1']:.4f} [{ci['macro_f1'][0]:.4f}, {ci['macro_f1'][1]:.4f}] | "
                f"{row['official_f1_favor_against']:.4f} "
                f"[{ci['official_f1_favor_against'][0]:.4f}, {ci['official_f1_favor_against'][1]:.4f}] |"
            )
    test = result["mcnemar"]
    lines += ["", "## Paired McNemar test", "",
              f"Baseline-only correct: `{test['baseline_only_correct']}`; final-model-only correct: `{test['candidate_only_correct']}`; exact two-sided p-value: `{test['exact_binomial_p_value']:.6f}`.", "",
              "McNemar's test measures paired correctness disagreement on this fixed test sample; it does not establish generalization beyond the SemEval domain.", "",
              "## Protocol notes", "",
              "The frozen transformer checkpoint was used without retraining. The fixed TF-IDF pipeline was fitted only on the development-training portion of the official training split, recreating its validation-era protocol. Test labels were used only for this final scoring pass. Predictions stored locally contain no tweet text."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def log_aggregate_result(result: dict[str, Any], config: dict[str, Any],
                         results_path: Path) -> None:
    """Track only aggregate/label-safe artifacts; never log input text."""
    if not config.get("tracking_uri") or not config.get("experiment_name"):
        return
    Path("mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(str(config["tracking_uri"]))
    mlflow.set_experiment(str(config["experiment_name"]))
    with mlflow.start_run(run_name="forced-rerun" if result["forced_rerun"] else "one-shot-final"):
        mlflow.log_params({
            "model_name": result["model_name"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "forced_rerun": result["forced_rerun"],
        })
        for prefix in ("final_model", "tfidf_baseline"):
            for metric in ("accuracy", "macro_f1", "official_f1_favor_against"):
                mlflow.log_metric(f"{prefix}_{metric}", result[prefix][metric])
        mlflow.log_metric("mcnemar_exact_p", result["mcnemar"]["exact_binomial_p_value"])
        mlflow.log_artifact(str(results_path))
        mlflow.log_artifact(str(config["report"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-hash", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config, warning = validate_preflight(args.config, args.checkpoint_hash, args.force)
    if warning:
        print(warning, file=sys.stderr)
    seed_everything(int(config["seed"]))

    # Authorized test access begins only after every preflight check above.
    train_spec, test_spec = official_task_a_file_specs(Path(config["raw_dir"]))
    train = load_task_a([train_spec], expected_targets=TASK_A_TARGETS)
    test = load_task_a([test_spec], expected_targets=TASK_A_TARGETS)
    split = prepare_task_a_validation_split(train)

    baseline = build_baseline_pipeline()
    baseline.fit(split.development_train.text, split.development_train.label)
    baseline_predictions = baseline.predict(test.text).tolist()
    final_predictions = _predict_transformer(test, config)

    result = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "warning": warning,
        "forced_rerun": bool(args.force),
        "model_name": config["model_name"],
        "checkpoint_sha256": args.checkpoint_hash.lower(),
        "final_model": evaluate_predictions(test, final_predictions, config),
        "tfidf_baseline": evaluate_predictions(test, baseline_predictions, config, 10000),
        "mcnemar": compute_mcnemar_test(
            (np.asarray(baseline_predictions) == test.label.to_numpy()).tolist(),
            (np.asarray(final_predictions) == test.label.to_numpy()).tolist(),
        ).to_dict(),
    }
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions.json").write_text(json.dumps({
        "rows": [{"row_index": int(index), "target": str(target),
                  "true_label": str(truth), "final_prediction": str(final),
                  "baseline_prediction": str(baseline_prediction)}
                 for index, (target, truth, final, baseline_prediction) in enumerate(zip(
                     test.target, test.label, final_predictions, baseline_predictions
                 ))]
    }, indent=2) + "\n", encoding="utf-8")
    results_path = output / "results.json"
    results_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    generate_report(result, Path(config["report"]))
    log_aggregate_result(result, config, results_path)
    lock = Path(config["lock_file"])
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({
        "evaluated_at_utc": result["evaluated_at_utc"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "forced_rerun": result["forced_rerun"], "warning": warning,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Final evaluation complete. Lock written to {lock}")


if __name__ == "__main__":
    main()
