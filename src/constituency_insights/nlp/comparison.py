"""Leakage-safe validation prediction generation and paired model comparison.

Generates row-level validation predictions with class probabilities for the
classical TF-IDF + Logistic Regression baseline and the fine-tuned DistilBERT
target-conditioned model (Track B). Computes paired error overlap and McNemar's
hypothesis test without touching the official test set or serializing raw tweet text.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
import torch

from constituency_insights.data import (
    TASK_A_TARGETS,
    ValidationSplit,
    load_task_a,
    official_task_a_file_specs,
    prepare_task_a_validation_split,
)
from constituency_insights.nlp.baseline import (
    LABEL_ORDER,
    build_baseline_pipeline,
)
from constituency_insights.nlp.distilbert import (
    StanceTextDataset,
    build_label_mappings,
)
from constituency_insights.nlp.error_analysis import extract_directional_errors
from constituency_insights.nlp.predict import load_model_and_tokenizer


@dataclass(frozen=True)
class McNemarResult:
    """Statistical summary of paired dichotomous model evaluation."""

    both_correct: int
    baseline_only_correct: int
    candidate_only_correct: int
    both_incorrect: int
    total_instances: int
    discordant_pairs: int
    statistic_continuity_corrected: float
    p_value_continuity_corrected: float
    statistic_uncorrected: float
    p_value_uncorrected: float
    exact_binomial_p_value: float
    is_statistically_significant_05: bool
    is_statistically_significant_01: bool
    assumptions_and_limitations: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return asdict(self)


def compute_mcnemar_test(
    baseline_correct: Sequence[bool],
    candidate_correct: Sequence[bool],
) -> McNemarResult:
    """Compute McNemar's test for paired classification predictions.

    Args:
        baseline_correct: Boolean sequence indicating if baseline predicted correctly.
        candidate_correct: Boolean sequence indicating if candidate model predicted correctly.

    Returns:
        McNemarResult with contingency counts, corrected and uncorrected chi-squared
        statistics, exact binomial p-value, and explicit assumption documentation.

    Raises:
        ValueError: If prediction sequences differ in length or are empty.
    """
    if len(baseline_correct) != len(candidate_correct):
        raise ValueError(
            f"Length mismatch: baseline ({len(baseline_correct)}) vs "
            f"candidate ({len(candidate_correct)})."
        )
    if not baseline_correct:
        raise ValueError("Cannot compute McNemar's test on empty prediction sequences.")

    b_corr = list(baseline_correct)
    c_corr = list(candidate_correct)

    both_correct = sum(b and c for b, c in zip(b_corr, c_corr))
    baseline_only = sum(b and not c for b, c in zip(b_corr, c_corr))
    candidate_only = sum(not b and c for b, c in zip(b_corr, c_corr))
    both_incorrect = sum(not b and not c for b, c in zip(b_corr, c_corr))
    total = len(b_corr)
    discordant = baseline_only + candidate_only

    if discordant == 0:
        stat_corrected = 0.0
        p_val_corrected = 1.0
        stat_uncorrected = 0.0
        p_val_uncorrected = 1.0
        exact_p = 1.0
    else:
        # Edwards continuity correction: (|b - c| - 1)^2 / (b + c)
        diff = abs(baseline_only - candidate_only)
        corrected_numerator = max(0.0, float(diff - 1)) ** 2
        stat_corrected = float(corrected_numerator / discordant)
        p_val_corrected = float(1.0 - chi2.cdf(stat_corrected, df=1))

        uncorrected_numerator = float(diff**2)
        stat_uncorrected = float(uncorrected_numerator / discordant)
        p_val_uncorrected = float(1.0 - chi2.cdf(stat_uncorrected, df=1))

        # Two-sided exact binomial test on discordant pairs
        binom_res = binomtest(
            k=min(baseline_only, candidate_only),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        )
        exact_p = float(binom_res.pvalue)

    assumptions = {
        "paired_design": (
            "Assumes both models were evaluated on identical instances in the same order."
        ),
        "dichotomous_outcomes": (
            "Evaluates binary correctness (correct vs. incorrect) on each instance."
        ),
        "null_hypothesis": (
            "H0: Discordant probabilities are symmetric (P(baseline correct, candidate incorrect) = "
            "P(candidate correct, baseline incorrect))."
        ),
        "continuity_correction": (
            "Edwards continuity correction (|b - c| - 1)^2 / (b + c) adjusts for discrete sample counts."
        ),
        "generalization_limitation": (
            "Statistical significance on this 583-row validation partition indicates genuine "
            "discordance on this specific sample (US political Twitter discourse, 2015-2016). "
            "It does NOT prove generalization across domains, across random splits, or to multilingual "
            "Indian constituency contexts."
        ),
    }

    return McNemarResult(
        both_correct=both_correct,
        baseline_only_correct=baseline_only,
        candidate_only_correct=candidate_only,
        both_incorrect=both_incorrect,
        total_instances=total,
        discordant_pairs=discordant,
        statistic_continuity_corrected=round(stat_corrected, 6),
        p_value_continuity_corrected=float(p_val_corrected),
        statistic_uncorrected=round(stat_uncorrected, 6),
        p_value_uncorrected=float(p_val_uncorrected),
        exact_binomial_p_value=float(exact_p),
        is_statistically_significant_05=bool(p_val_corrected < 0.05),
        is_statistically_significant_01=bool(p_val_corrected < 0.01),
        assumptions_and_limitations=assumptions,
    )


def generate_baseline_validation_predictions(
    split: ValidationSplit,
    *,
    pipeline: Pipeline | None = None,
    label_order: Sequence[str] = LABEL_ORDER,
) -> list[dict[str, Any]]:
    """Fit baseline on development-train and generate row-level validation predictions with probabilities.

    Args:
        split: In-memory ValidationSplit containing development_train and validation partitions.
        pipeline: Optional pre-fitted or pre-configured Pipeline. If None, builds standard baseline.
        label_order: Explicit class label order.

    Returns:
        List of validation prediction records containing row_index, target, true_label,
        predicted_label, and class probabilities dictionary. Raw tweet text is never stored.
    """
    development = split.development_train
    validation = split.validation

    if pipeline is None:
        pipeline = build_baseline_pipeline()
        pipeline.fit(development["text"], development["label"])

    preds = pipeline.predict(validation["text"])
    probs = pipeline.predict_proba(validation["text"])

    classifier = pipeline.named_steps["classifier"]
    class_indices = {cls_name: i for i, cls_name in enumerate(classifier.classes_)}

    records: list[dict[str, Any]] = []
    has_source_id = "source_id" in validation.columns

    for (row_idx, row), pred, prob_row in zip(
        validation.iterrows(), preds, probs
    ):
        prob_dict = {
            label: round(float(prob_row[class_indices[label]]), 6)
            for label in label_order
            if label in class_indices
        }
        record: dict[str, Any] = {
            "row_index": int(row_idx),
            "target": str(row["target"]),
            "true_label": str(row["label"]),
            "predicted_label": str(pred),
            "probabilities": prob_dict,
        }
        if has_source_id and pd.notna(row["source_id"]):
            record["source_id"] = str(row["source_id"])
        records.append(record)

    _verify_no_tweet_text(records)
    return records


def generate_distilbert_validation_predictions(
    split: ValidationSplit,
    *,
    model_path: Path | str = Path("experiments/stance_distilbert_target"),
    model: Any | None = None,
    tokenizer: Any | None = None,
    device: str | torch.device | None = None,
    batch_size: int = 16,
    max_length: int = 128,
    label_order: Sequence[str] = LABEL_ORDER,
) -> list[dict[str, Any]]:
    """Generate row-level validation predictions with probabilities using fine-tuned DistilBERT.

    Args:
        split: In-memory ValidationSplit containing development_train and validation partitions.
        model_path: Checkpoint directory containing model weights and configuration.
        model: Optional pre-loaded PyTorch model (useful for unit testing).
        tokenizer: Optional pre-loaded Hugging Face tokenizer.
        device: Device to run inference on ('cpu', 'cuda', or torch.device).
        batch_size: Batch size for validation inference.
        max_length: Max token length for paired encoding.
        label_order: Explicit class label order.

    Returns:
        List of validation prediction records containing row_index, target, true_label,
        predicted_label, and class probabilities dictionary. Raw tweet text is never stored.
    """
    validation = split.validation

    if model is None or tokenizer is None:
        loaded_model, loaded_tokenizer, resolved_device = load_model_and_tokenizer(
            model_path,
            device=device,
            label_order=label_order,
        )
        if model is None:
            model = loaded_model
        if tokenizer is None:
            tokenizer = loaded_tokenizer
    else:
        if device is None:
            try:
                resolved_device = next(model.parameters()).device
            except (AttributeError, StopIteration):
                resolved_device = torch.device("cpu")
        else:
            resolved_device = torch.device(device)

    # Paired tokenization matching Track B protocol: [CLS] text [SEP] target [SEP]
    encodings = tokenizer(
        validation["text"].tolist(),
        text_pair=validation["target"].tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    dataset = StanceTextDataset(encodings)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    if hasattr(model, "eval"):
        model.eval()

    all_probs: list[list[float]] = []
    with torch.no_grad():
        for batch in data_loader:
            batch_tensors = {k: v.to(resolved_device) for k, v in batch.items()}
            outputs = model(**batch_tensors)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            probs = torch.softmax(logits, dim=-1)
            all_probs.extend(probs.cpu().tolist())

    _, id_to_label = build_label_mappings(label_order)
    predicted_labels = [
        id_to_label[int(np.argmax(p))] for p in all_probs
    ]

    records: list[dict[str, Any]] = []
    has_source_id = "source_id" in validation.columns

    for (row_idx, row), pred, prob_list in zip(
        validation.iterrows(), predicted_labels, all_probs
    ):
        prob_dict = {
            label: round(float(prob), 6)
            for label, prob in zip(label_order, prob_list)
        }
        record = {
            "row_index": int(row_idx),
            "target": str(row["target"]),
            "true_label": str(row["label"]),
            "predicted_label": str(pred),
            "probabilities": prob_dict,
        }
        if has_source_id and pd.notna(row["source_id"]):
            record["source_id"] = str(row["source_id"])
        records.append(record)

    _verify_no_tweet_text(records)
    return records


def _classification_metrics_from_predictions(
    truth: Sequence[str],
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


def compare_validation_predictions(
    baseline_predictions: Sequence[dict[str, Any]],
    candidate_predictions: Sequence[dict[str, Any]],
    *,
    label_order: Sequence[str] = LABEL_ORDER,
    baseline_name: str = "TF-IDF + Logistic Regression",
    candidate_name: str = "DistilBERT (Track B, Target-Conditioned)",
) -> dict[str, Any]:
    """Perform comprehensive paired comparison between baseline and candidate model.

    Args:
        baseline_predictions: List of row-level prediction dictionaries for baseline.
        candidate_predictions: List of row-level prediction dictionaries for candidate.
        label_order: Stance class label ordering.
        baseline_name: Descriptive name for baseline model.
        candidate_name: Descriptive name for candidate model.

    Returns:
        Structured dictionary containing metrics, per-target breakdown, error overlap
        contingency, McNemar test results, and directional error transition comparison.

    Raises:
        ValueError: If prediction sets differ in length or row alignment.
    """
    if len(baseline_predictions) != len(candidate_predictions):
        raise ValueError(
            f"Prediction length mismatch: baseline has {len(baseline_predictions)} rows, "
            f"candidate has {len(candidate_predictions)} rows."
        )
    if not baseline_predictions:
        raise ValueError("Cannot compare empty prediction lists.")

    labels = list(label_order)
    n_rows = len(baseline_predictions)

    # Strict alignment validation
    for i, (b_row, c_row) in enumerate(zip(baseline_predictions, candidate_predictions)):
        if b_row["row_index"] != c_row["row_index"]:
            raise ValueError(
                f"Row index mismatch at position {i}: baseline has {b_row['row_index']}, "
                f"candidate has {c_row['row_index']}."
            )
        if b_row["true_label"] != c_row["true_label"]:
            raise ValueError(
                f"True label mismatch at position {i} (row {b_row['row_index']}): "
                f"baseline has '{b_row['true_label']}', candidate has '{c_row['true_label']}'."
            )
        if b_row["target"] != c_row["target"]:
            raise ValueError(
                f"Target mismatch at position {i} (row {b_row['row_index']}): "
                f"baseline has '{b_row['target']}', candidate has '{c_row['target']}'."
            )

    true_labels = [row["true_label"] for row in baseline_predictions]
    b_preds = [row["predicted_label"] for row in baseline_predictions]
    c_preds = [row["predicted_label"] for row in candidate_predictions]
    targets = [row["target"] for row in baseline_predictions]

    # Overall metrics
    b_metrics = _classification_metrics_from_predictions(true_labels, b_preds, labels)
    c_metrics = _classification_metrics_from_predictions(true_labels, c_preds, labels)

    b_correct = [p == y for p, y in zip(b_preds, true_labels)]
    c_correct = [p == y for p, y in zip(c_preds, true_labels)]

    mcnemar = compute_mcnemar_test(b_correct, c_correct)

    # Per-target paired breakdown
    unique_targets = sorted(set(targets))
    by_target: dict[str, Any] = {}
    for target in unique_targets:
        indices = [i for i, t in enumerate(targets) if t == target]
        t_truth = [true_labels[i] for i in indices]
        t_b_preds = [b_preds[i] for i in indices]
        t_c_preds = [c_preds[i] for i in indices]

        t_b_correct = [b_correct[i] for i in indices]
        t_c_correct = [c_correct[i] for i in indices]

        t_both_correct = sum(b and c for b, c in zip(t_b_correct, t_c_correct))
        t_b_only = sum(b and not c for b, c in zip(t_b_correct, t_c_correct))
        t_c_only = sum(not b and c for b, c in zip(t_b_correct, t_c_correct))
        t_both_incorrect = sum(not b and not c for b, c in zip(t_b_correct, t_c_correct))

        by_target[target] = {
            "rows": len(indices),
            "baseline": _classification_metrics_from_predictions(t_truth, t_b_preds, labels),
            "candidate": _classification_metrics_from_predictions(t_truth, t_c_preds, labels),
            "error_overlap": {
                "both_correct": t_both_correct,
                "baseline_only_correct": t_b_only,
                "candidate_only_correct": t_c_only,
                "both_incorrect": t_both_incorrect,
            },
        }

    # Per-class error overlap breakdown
    by_class: dict[str, Any] = {}
    for label in labels:
        indices = [i for i, y in enumerate(true_labels) if y == label]
        cls_b_correct = [b_correct[i] for i in indices]
        cls_c_correct = [c_correct[i] for i in indices]

        by_class[label] = {
            "support": len(indices),
            "both_correct": sum(b and c for b, c in zip(cls_b_correct, cls_c_correct)),
            "baseline_only_correct": sum(b and not c for b, c in zip(cls_b_correct, cls_c_correct)),
            "candidate_only_correct": sum(not b and c for b, c in zip(cls_b_correct, cls_c_correct)),
            "both_incorrect": sum(not b and not c for b, c in zip(cls_b_correct, cls_c_correct)),
        }

    # Directional errors extracted from confusion matrices
    b_directional = extract_directional_errors(b_metrics["confusion_matrix"], labels)
    c_directional = extract_directional_errors(c_metrics["confusion_matrix"], labels)

    directional_comparison: dict[str, Any] = {}
    for pair in sorted(set(b_directional.keys()) | set(c_directional.keys())):
        b_count = b_directional.get(pair, 0)
        c_count = c_directional.get(pair, 0)
        directional_comparison[pair] = {
            "baseline_errors": b_count,
            "candidate_errors": c_count,
            "delta_candidate_minus_baseline": c_count - b_count,
        }

    comparison: dict[str, Any] = {
        "evaluation_scope": "SemEval-2016 Task 6, Task A derived validation partition",
        "validation_rows": n_rows,
        "label_order": labels,
        "models": {
            "baseline": {
                "name": baseline_name,
                "accuracy": b_metrics["accuracy"],
                "macro_f1": b_metrics["macro_f1"],
                "total_errors": n_rows - sum(b_correct),
                "per_class": b_metrics["per_class"],
                "confusion_matrix": b_metrics["confusion_matrix"],
            },
            "candidate": {
                "name": candidate_name,
                "accuracy": c_metrics["accuracy"],
                "macro_f1": c_metrics["macro_f1"],
                "total_errors": n_rows - sum(c_correct),
                "per_class": c_metrics["per_class"],
                "confusion_matrix": c_metrics["confusion_matrix"],
            },
        },
        "deltas": {
            "accuracy": round(c_metrics["accuracy"] - b_metrics["accuracy"], 6),
            "accuracy_percent_change": round(
                (c_metrics["accuracy"] - b_metrics["accuracy"])
                / b_metrics["accuracy"]
                * 100,
                2,
            ),
            "macro_f1": round(c_metrics["macro_f1"] - b_metrics["macro_f1"], 6),
            "macro_f1_percent_change": round(
                (c_metrics["macro_f1"] - b_metrics["macro_f1"])
                / b_metrics["macro_f1"]
                * 100,
                2,
            ),
            "per_class_f1": {
                label: round(
                    c_metrics["per_class"][label]["f1"]
                    - b_metrics["per_class"][label]["f1"],
                    6,
                )
                for label in labels
            },
        },
        "error_overlap_contingency": {
            "both_correct": mcnemar.both_correct,
            "baseline_only_correct": mcnemar.baseline_only_correct,
            "candidate_only_correct": mcnemar.candidate_only_correct,
            "both_incorrect": mcnemar.both_incorrect,
            "total_instances": mcnemar.total_instances,
            "percent_both_correct": round(
                mcnemar.both_correct / mcnemar.total_instances * 100, 2
            ),
            "percent_both_incorrect": round(
                mcnemar.both_incorrect / mcnemar.total_instances * 100, 2
            ),
            "percent_baseline_only": round(
                mcnemar.baseline_only_correct / mcnemar.total_instances * 100, 2
            ),
            "percent_candidate_only": round(
                mcnemar.candidate_only_correct / mcnemar.total_instances * 100, 2
            ),
        },
        "mcnemar_test": mcnemar.to_dict(),
        "directional_errors": directional_comparison,
        "by_class_error_overlap": by_class,
        "by_target": by_target,
    }

    return comparison


def save_validation_predictions(
    predictions: Sequence[dict[str, Any]],
    output_path: Path | str,
) -> None:
    """Save validation predictions to JSON file with strict privacy verification."""
    _verify_no_tweet_text(predictions)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(list(predictions), indent=2) + "\n"
    _verify_no_forbidden_keys(payload)
    path.write_text(payload, encoding="utf-8")


def save_comparison_report(
    report: dict[str, Any],
    output_path: Path | str,
) -> None:
    """Save paired comparison report to JSON file with privacy verification."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    _verify_no_forbidden_keys(payload)
    path.write_text(payload, encoding="utf-8")


def run_validation_comparison(
    *,
    raw_dir: Path = Path("data/raw/stance/semeval2016_task6"),
    baseline_output_dir: Path = Path("experiments/stance_tfidf_logreg"),
    distilbert_output_dir: Path = Path("experiments/stance_distilbert_target"),
    comparison_report_path: Path = Path("experiments/model_comparison_report.json"),
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Execute end-to-end paired comparison without retraining or modifying splits.

    1. Ingests Task A training file and generates exact 583-row validation partition.
    2. Generates baseline predictions and saves to baseline_output_dir/validation_predictions.json.
    3. Loads existing Track B checkpoint, generates predictions, and saves to
       distilbert_output_dir/validation_predictions.json.
    4. Computes paired comparison, contingency table, and McNemar test.
    5. Saves comparison report to comparison_report_path.

    Returns:
        Structured comparison report dictionary.
    """
    train_spec, _ = official_task_a_file_specs(raw_dir)
    data = load_task_a([train_spec], expected_targets=TASK_A_TARGETS)
    split = prepare_task_a_validation_split(data)

    # 1. Baseline predictions
    baseline_preds = generate_baseline_validation_predictions(split)
    save_validation_predictions(
        baseline_preds,
        baseline_output_dir / "validation_predictions.json",
    )

    # 2. DistilBERT Track B predictions
    distilbert_preds = generate_distilbert_validation_predictions(
        split,
        model_path=distilbert_output_dir,
        device=device,
    )
    save_validation_predictions(
        distilbert_preds,
        distilbert_output_dir / "validation_predictions.json",
    )

    # 3. Paired comparison and McNemar test
    report = compare_validation_predictions(
        baseline_predictions=baseline_preds,
        candidate_predictions=distilbert_preds,
    )

    # 4. Save comparison report
    save_comparison_report(report, comparison_report_path)

    return report


def _verify_no_tweet_text(records: Sequence[dict[str, Any]]) -> None:
    """Ensure records contain only identifiers, targets, labels, and probabilities."""
    forbidden = {"text", "text_excerpt", "Tweet", "raw_tweet"}
    for record in records:
        found = forbidden.intersection(record.keys())
        if found:
            raise ValueError(
                f"Privacy violation: Forbidden tweet text keys {found} found in prediction record."
            )


def _verify_no_forbidden_keys(text: str) -> None:
    """Scan serialized text to guarantee no tweet text fields are exported."""
    for forbidden_key in ('"text":', '"text_excerpt":', '"Tweet":', '"raw_tweet":'):
        if forbidden_key in text:
            raise ValueError(
                f"Privacy violation: Forbidden key {forbidden_key} found in serialized output."
            )


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line runner for paired model validation comparison."""
    parser = argparse.ArgumentParser(
        description="Run paired validation comparison between TF-IDF baseline and DistilBERT."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/stance/semeval2016_task6"),
        help="Path to SemEval-2016 Task 6 raw data directory.",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("experiments/stance_tfidf_logreg"),
        help="Directory to save baseline validation predictions.",
    )
    parser.add_argument(
        "--distilbert-dir",
        type=Path,
        default=Path("experiments/stance_distilbert_target"),
        help="Directory containing fine-tuned DistilBERT checkpoint and to save predictions.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("experiments/model_comparison_report.json"),
        help="Path to save paired model comparison report JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target device ('cuda' or 'cpu'). Defaults to CUDA if available.",
    )
    args = parser.parse_args(argv)

    print("Running paired validation comparison...")
    report = run_validation_comparison(
        raw_dir=args.raw_dir,
        baseline_output_dir=args.baseline_dir,
        distilbert_output_dir=args.distilbert_dir,
        comparison_report_path=args.report_path,
        device=args.device,
    )

    m = report["mcnemar_test"]
    print("\n--- Paired Model Comparison Summary ---")
    print(f"Validation Instances: {report['validation_rows']}")
    print(
        f"Baseline Accuracy:   {report['models']['baseline']['accuracy']:.4f} "
        f"({report['models']['baseline']['macro_f1']:.4f} Macro-F1)"
    )
    print(
        f"DistilBERT Accuracy: {report['models']['candidate']['accuracy']:.4f} "
        f"({report['models']['candidate']['macro_f1']:.4f} Macro-F1)"
    )
    print(
        f"Accuracy Delta:      {report['deltas']['accuracy']:+.4f} "
        f"({report['deltas']['accuracy_percent_change']:+.2f}%)"
    )
    print(
        f"Macro-F1 Delta:      {report['deltas']['macro_f1']:+.4f} "
        f"({report['deltas']['macro_f1_percent_change']:+.2f}%)"
    )
    print("\nContingency Table:")
    print(f"  Both Correct:              {m['both_correct']} ({report['error_overlap_contingency']['percent_both_correct']}%)")
    print(f"  Baseline Only Correct:     {m['baseline_only_correct']} ({report['error_overlap_contingency']['percent_baseline_only']}%)")
    print(f"  DistilBERT Only Correct:   {m['candidate_only_correct']} ({report['error_overlap_contingency']['percent_candidate_only']}%)")
    print(f"  Both Incorrect:            {m['both_incorrect']} ({report['error_overlap_contingency']['percent_both_incorrect']}%)")
    print("\nMcNemar's Hypothesis Test:")
    print(f"  Discordant Pairs (b + c):  {m['discordant_pairs']}")
    print(f"  Chi-squared (corrected):   {m['statistic_continuity_corrected']:.4f} (p = {m['p_value_continuity_corrected']:.6e})")
    print(f"  Exact Binomial p-value:    {m['exact_binomial_p_value']:.6e}")
    print(f"  Statistically Significant: {m['is_statistically_significant_01']} (at alpha = 0.01)")
    print(f"\nOutputs saved to:")
    print(f"  - {args.baseline_dir / 'validation_predictions.json'}")
    print(f"  - {args.distilbert_dir / 'validation_predictions.json'}")
    print(f"  - {args.report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
