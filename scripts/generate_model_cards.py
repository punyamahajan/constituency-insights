"""Generate model cards from committed, code-generated evaluation reports."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


def _stance_metrics(report: str) -> tuple[str, str, str]:
    row = next((line for line in report.splitlines() if line.startswith("| Frozen DistilBERT |")), "")
    values = re.findall(r"\|\s*([0-9]+\.[0-9]+)\s*\|", row)
    if len(values) < 3:
        raise ValueError("Generated final test report lacks Frozen DistilBERT metrics")
    return values[0], values[1], values[2]


def generate_cards(final_report: Path, outcome_report: Path, stance_card: Path,
                   outcome_card: Path) -> None:
    final_text = final_report.read_text(encoding="utf-8")
    accuracy, macro_f1, official_f1 = _stance_metrics(final_text)
    stance_card.write_text(f"""# Stance model card

## Intended use

The API serves a frozen target-conditioned DistilBERT classifier for stance
screening on English political text paired with a target. It is an inference
service component, not an autonomous decision-maker.

## Training data and metrics

The model was trained on SemEval-2016 Task 6A US English Twitter data. Metrics
below are pulled from the generated `docs/final_test_results.md` report:

- Accuracy: `{accuracy}`
- Macro-F1: `{macro_f1}`
- Official FAVOR/AGAINST F1: `{official_f1}`

## Limitations

SemEval is English-only and represents US political discourse. The model is not
validated for Hindi, Tamil, code-mixed content, Indian politics, constituency
discourse, or distribution shifts. Probabilities are not guaranteed calibrated.

## Ethical considerations

Political stance predictions can misclassify, stereotype, or amplify annotation
bias. Do not use outputs to target individuals, determine eligibility, or make
high-impact decisions without human review and domain-specific validation.
Request text is not logged by this service.
""", encoding="utf-8")
    outcome_text = outcome_report.read_text(encoding="utf-8")
    if "Not evaluated" in outcome_text or "not been generated" in outcome_text:
        metrics = "No real outcome metrics are reported because the trained artifact is unavailable."
    else:
        metrics = "Outcome metrics below are pulled from the generated tabular report.\n\n" + "\n".join(
            line for line in outcome_text.splitlines() if line.startswith("Final held-out") or "coverage" in line.lower()
        )
    outcome_card.write_text(f"""# Outcome model card

## Intended use

The API accepts constituency-level election features and returns a predicted
vote share with a split-conformal interval. It is intended for research
prototyping and decision support with human oversight.

## Training data and metrics

The outcome model is designed for locally supplied constituency election and
demographic data. {metrics}

## Limitations

Conformal coverage depends on the calibration exchangeability assumptions and
can fail under structural political or demographic shifts. The current
repository does not contain a real trained outcome artifact, so the endpoint
returns HTTP 503 rather than fabricate predictions.

## Ethical considerations

Election forecasts can affect political strategy, public trust, and resource
allocation. Do not present intervals as certainties or use predictions to
target protected groups. Validate provenance, fairness, and uncertainty before
deployment.
""", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-report", type=Path, default=Path("docs/final_test_results.md"))
    parser.add_argument("--outcome-report", type=Path, default=Path("docs/tabular_report.md"))
    parser.add_argument("--stance-card", type=Path, default=Path("docs/model_card_stance.md"))
    parser.add_argument("--outcome-card", type=Path, default=Path("docs/model_card_outcome.md"))
    args = parser.parse_args()
    generate_cards(args.final_report, args.outcome_report, args.stance_card, args.outcome_card)


if __name__ == "__main__":
    main()
