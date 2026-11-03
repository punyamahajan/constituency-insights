#!/usr/bin/env python
"""Train and compare YAML-configured stance models."""

from __future__ import annotations

import argparse
from pathlib import Path

from constituency_insights.nlp.training import (
    generate_comparison_from_mlflow,
    load_training_configs,
    train_stance_model,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    configs = load_training_configs(args.config)
    for config in configs:
        result = train_stance_model(config)
        print(f"{result.run_name}: macro-F1={result.best_macro_f1:.6f} run={result.run_id}")
    report = generate_comparison_from_mlflow(configs)
    print(f"Comparison report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
