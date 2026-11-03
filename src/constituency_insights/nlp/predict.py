"""Prediction and inference interfaces for stance classification."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from constituency_insights.nlp.distilbert import LABEL_ORDER, build_label_mappings


DEFAULT_CHECKPOINT_DIR = Path("experiments/stance_distilbert_target")
DEFAULT_BASE_MODEL_ID = "distilbert/distilbert-base-uncased"

DEMO_EXAMPLES: tuple[dict[str, str], ...] = (
    {
        "label": "Illustrative Pro-Hillary Statement",
        "target": "Hillary Clinton",
        "tweet": "Hillary is our best choice if we truly want to continue being a progressive nation. #Ohio",
    },
    {
        "label": "Illustrative Pro-Choice Statement",
        "target": "Legalization of Abortion",
        "tweet": "Whether someone wants to have children or not should be completely up to the person carrying that pregnancy.",
    },
    {
        "label": "Illustrative Pro-Climate Action Statement",
        "target": "Climate Change is a Real Concern",
        "tweet": "Renewable energy investments are essential for protecting future generations from climate disruption.",
    },
)


@dataclass(frozen=True)
class StancePrediction:
    """Stance prediction output, class probabilities, and model provenance."""

    tweet_text: str
    target: str
    predicted_stance: str
    probabilities: dict[str, float]
    model_source: str
    model_identifier: str
    is_fine_tuned: bool
    warning_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert prediction to dictionary representation."""
        return asdict(self)


def validate_checkpoint_path(checkpoint_path: str | Path) -> Path | str:
    """Validate that an explicitly specified checkpoint path or model identifier exists.

    If given a local directory, checks for required weight files ('model.safetensors'
    or 'pytorch_model.bin'). If given a config JSON file, extracts 'model_name_or_path'.
    If given a model identifier string, returns it.

    Raises:
        FileNotFoundError: If a specified local directory or file does not exist,
            or if a checkpoint directory lacks model weights.
        ValueError: If checkpoint path is empty or invalid.
    """
    if not checkpoint_path or not str(checkpoint_path).strip():
        raise ValueError("Checkpoint path must be a non-empty string or Path.")

    path = Path(checkpoint_path)

    # If it's a JSON configuration file (e.g. configs/distilbert_target_baseline.json)
    if path.is_file() and path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if "model_name_or_path" in data:
            return validate_checkpoint_path(data["model_name_or_path"])

    # If it is an existing local directory
    if path.exists():
        if not path.is_dir():
            raise FileNotFoundError(
                f"Checkpoint path '{path}' is a file, but a model directory is expected."
            )

        has_weights = (
            (path / "model.safetensors").is_file()
            or (path / "pytorch_model.bin").is_file()
        )
        if not has_weights:
            raise FileNotFoundError(
                f"Checkpoint directory '{path}' does not contain model weights "
                "('model.safetensors' or 'pytorch_model.bin'). "
                "The experiment directory contains metrics and run metadata, "
                "but no fine-tuned model checkpoint was exported to disk."
            )
        return path

    # If path doesn't exist on disk, check if it was intended as a filesystem path
    path_str = str(checkpoint_path).replace("\\", "/")
    is_explicit_path = (
        path_str.startswith(("./", "../", "/"))
        or (len(path_str) > 1 and path_str[1] == ":")
        or any(
            path_str.startswith(prefix)
            for prefix in ("experiments/", "models/", "checkpoints/", "configs/", "data/")
        )
    )
    if is_explicit_path:
        raise FileNotFoundError(
            f"Specified checkpoint directory '{checkpoint_path}' does not exist on disk."
        )

    # Otherwise treat as Hugging Face model identifier (e.g. 'distilbert/distilbert-base-uncased')
    return str(checkpoint_path)


def resolve_model_source(
    checkpoint_path: str | Path | None,
) -> tuple[Path | str, bool, str | None]:
    """Resolve model source, fine-tuned status, and optional user warning.

    Returns:
        tuple of (resolved_source, is_fine_tuned, warning_message)
    """
    if checkpoint_path is None:
        # Check if default fine-tuned experiment directory has weights
        has_default_weights = (
            DEFAULT_CHECKPOINT_DIR.is_dir()
            and (
                (DEFAULT_CHECKPOINT_DIR / "model.safetensors").is_file()
                or (DEFAULT_CHECKPOINT_DIR / "pytorch_model.bin").is_file()
            )
        )
        if has_default_weights:
            return DEFAULT_CHECKPOINT_DIR, True, None

        warning = (
            "Fine-tuned model checkpoint at 'experiments/stance_distilbert_target' is unavailable "
            "(metrics and metadata were saved, but model weights were not exported to disk). "
            "Falling back to cached base model 'distilbert/distilbert-base-uncased' with an untrained "
            "classification head. Predictions are illustrative only."
        )
        return DEFAULT_BASE_MODEL_ID, False, warning

    # Explicit checkpoint path provided: validate strictly
    valid_source = validate_checkpoint_path(checkpoint_path)
    if isinstance(valid_source, Path):
        return valid_source, True, None

    warning = (
        f"Using base pretrained model '{valid_source}' with an untrained classification head. "
        "Predictions are illustrative demonstration outputs, NOT fine-tuned stance predictions."
    )
    return str(valid_source), False, warning


def load_model_and_tokenizer(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = None,
    label_order: Sequence[str] = LABEL_ORDER,
    local_files_only: bool = True,
) -> tuple[Any, Any, torch.device]:
    """Load model and tokenizer from verified checkpoint path or local cache."""
    valid_path = validate_checkpoint_path(checkpoint_path)
    label_to_id, id_to_label = build_label_mappings(label_order)

    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    model_source = str(valid_path) if isinstance(valid_path, Path) else valid_path
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
        num_labels=len(label_order),
        id2label=id_to_label,
        label2id=label_to_id,
        local_files_only=local_files_only,
    )
    model.to(target_device)
    model.eval()
    return model, tokenizer, target_device


def predict_stance(
    tweet_text: str,
    target: str,
    *,
    model: Any | None = None,
    tokenizer: Any | None = None,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
    max_length: int = 128,
    label_order: Sequence[str] = LABEL_ORDER,
    local_files_only: bool = True,
) -> StancePrediction:
    """Predict stance and class probabilities for a tweet conditioned on a target entity.

    Args:
        tweet_text: Input tweet text (must be non-empty).
        target: Target entity name (must be non-empty).
        model: Optional pre-loaded PyTorch sequence classification model.
        tokenizer: Optional pre-loaded Hugging Face tokenizer.
        checkpoint_path: Path to checkpoint directory, config JSON, or model ID.
            If omitted, attempts default directory, then falls back with warning.
        device: Device to run inference on ('cpu', 'cuda', or torch.device).
        max_length: Max sequence length for tokenization (default 128).
        label_order: Sequence of stance labels in index order.
        local_files_only: Whether to restrict HF loading to local files only.

    Returns:
        StancePrediction containing predicted stance, per-class probabilities,
        and provenance indicators (is_fine_tuned and model_source).

    Raises:
        ValueError: If tweet text or target is empty or whitespace-only.
        FileNotFoundError: If explicitly requested checkpoint path or weights are missing.
    """
    if not isinstance(tweet_text, str) or not tweet_text.strip():
        raise ValueError("Tweet text must be a non-empty string.")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("Target entity must be a non-empty string.")

    resolved_device: torch.device
    is_fine_tuned: bool
    warning_message: str | None = None
    model_identifier: str

    if model is None or tokenizer is None:
        source, is_fine_tuned, warning_message = resolve_model_source(checkpoint_path)
        model_identifier = str(source)
        loaded_model, loaded_tokenizer, resolved_device = load_model_and_tokenizer(
            source,
            device=device,
            label_order=label_order,
            local_files_only=local_files_only,
        )
        if model is None:
            model = loaded_model
        if tokenizer is None:
            tokenizer = loaded_tokenizer
    else:
        # Pre-supplied model (e.g. mocked in unit tests)
        is_fine_tuned = getattr(model, "is_fine_tuned", True)
        model_identifier = str(checkpoint_path or "custom_model")
        if device is None:
            try:
                resolved_device = next(model.parameters()).device
            except (AttributeError, StopIteration):
                resolved_device = torch.device("cpu")
        else:
            resolved_device = torch.device(device)

    # Paired tokenization matching Track B protocol: tokenizer(text, text_pair=target)
    encoding = tokenizer(
        [tweet_text.strip()],
        text_pair=[target.strip()],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    batch = {k: v.to(resolved_device) for k, v in encoding.items()}
    if hasattr(model, "eval"):
        model.eval()

    with torch.no_grad():
        outputs = model(**batch)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
        pred_idx = int(torch.argmax(logits, dim=-1).item())

    _, id_to_label = build_label_mappings(label_order)
    predicted_stance = id_to_label[pred_idx]
    probabilities = {
        label: round(float(prob), 6)
        for label, prob in zip(label_order, probs)
    }

    model_source_desc = (
        "fine_tuned_checkpoint" if is_fine_tuned else "cached_base_model"
    )

    return StancePrediction(
        tweet_text=tweet_text.strip(),
        target=target.strip(),
        predicted_stance=predicted_stance,
        probabilities=probabilities,
        model_source=model_source_desc,
        model_identifier=model_identifier,
        is_fine_tuned=is_fine_tuned,
        warning_message=warning_message,
    )


def run_demonstration(
    *,
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
    json_output: bool = False,
) -> int:
    """Run an illustrative demonstration using predefined examples."""
    results: list[dict[str, Any]] = []

    if not json_output:
        print("=" * 76)
        print("CONSTITUENCY INSIGHTS: STANCE CLASSIFICATION DEMONSTRATION")
        print("=" * 76)
        print("NOTE: The examples below are ILLUSTRATIVE INPUTS to demonstrate the CLI")
        print("and paired-text inference interface. They are NOT benchmark evaluations.")
        print("=" * 76)

    for item in DEMO_EXAMPLES:
        try:
            pred = predict_stance(
                tweet_text=item["tweet"],
                target=item["target"],
                checkpoint_path=checkpoint_path,
                device=device,
            )
        except Exception as err:
            sys.stderr.write(f"Demonstration failed on '{item['label']}': {err}\n")
            return 1

        entry = {
            "label": item["label"],
            "prediction": pred.to_dict(),
        }
        results.append(entry)

        if not json_output:
            print(f"\n--- {item['label']} ---")
            print(f"  Target:           {pred.target}")
            print(f"  Tweet:            {pred.tweet_text}")
            print(f"  Model Source:     {pred.model_source} ({pred.model_identifier})")
            print(f"  Fine-Tuned:       {'Yes' if pred.is_fine_tuned else 'No (untrained classification head)'}")
            if pred.warning_message:
                print(f"  Notice:           {pred.warning_message}")
            print(f"  Predicted Stance: {pred.predicted_stance}")
            print("  Class Probabilities:")
            for label, prob in pred.probabilities.items():
                print(f"    - {label:<8}: {prob:.4f} ({prob * 100:.1f}%)")

    if json_output:
        print(json.dumps({"demonstration_results": results}, indent=2))
    else:
        print("\n" + "=" * 76)
        print("Demonstration completed successfully.")
        print("=" * 76)

    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser for stance prediction."""
    parser = argparse.ArgumentParser(
        description="Predict stance (AGAINST, FAVOR, NONE) for a tweet conditioned on a target entity."
    )
    parser.add_argument(
        "--tweet",
        "-t",
        default=None,
        help="Input tweet text to evaluate (required unless --demo is specified).",
    )
    parser.add_argument(
        "--target",
        "-g",
        default=None,
        help="Target entity name (e.g. 'Hillary Clinton') (required unless --demo is specified).",
    )
    parser.add_argument(
        "--checkpoint-path",
        "-c",
        default=None,
        help=(
            "Optional path to fine-tuned checkpoint directory or model ID. "
            "If omitted, checks 'experiments/stance_distilbert_target' and falls back "
            "with a warning to the cached base model."
        ),
    )
    parser.add_argument(
        "--device",
        "-d",
        default=None,
        help="Compute device ('cpu' or 'cuda'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run an illustrative demonstration with predefined examples.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON instead of formatted text.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run CLI entry point for stance prediction."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.demo:
        return run_demonstration(
            checkpoint_path=args.checkpoint_path,
            device=args.device,
            json_output=args.json,
        )

    if not args.tweet or not args.target:
        parser.print_usage(sys.stderr)
        sys.stderr.write(
            "Error: Both --tweet/-t and --target/-g are required (or run with --demo).\n"
        )
        return 2

    try:
        prediction = predict_stance(
            tweet_text=args.tweet,
            target=args.target,
            checkpoint_path=args.checkpoint_path,
            device=args.device,
        )
    except (FileNotFoundError, ValueError) as err:
        sys.stderr.write(f"Error: {err}\n")
        if "does not contain model weights" in str(err):
            sys.stderr.write(
                "\nExplanation: The requested directory contains metrics and run metadata, "
                "but model weights were not serialized to disk during that experiment.\n"
                "Tip: To run with the cached base model, supply:\n"
                "  --checkpoint-path distilbert/distilbert-base-uncased\n"
                "Or omit --checkpoint-path to allow automatic fallback with a warning.\n"
            )
        return 1
    except Exception as err:
        sys.stderr.write(f"Unexpected error during inference: {err}\n")
        return 1

    if prediction.warning_message and not args.json:
        sys.stderr.write(f"WARNING: {prediction.warning_message}\n\n")

    if args.json:
        print(json.dumps(prediction.to_dict(), indent=2))
    else:
        print("Prediction Result:")
        print(f"  Model Source:     {prediction.model_source} ({prediction.model_identifier})")
        print(f"  Fine-Tuned:       {'Yes' if prediction.is_fine_tuned else 'No (untrained classification head)'}")
        print(f"  Tweet:            {prediction.tweet_text}")
        print(f"  Target:           {prediction.target}")
        print(f"  Predicted Stance: {prediction.predicted_stance}")
        print("  Class Probabilities:")
        for label, prob in prediction.probabilities.items():
            print(f"    - {label:<8}: {prob:.4f} ({prob * 100:.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
