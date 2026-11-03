#!/usr/bin/env python
"""Run the YAML-defined full fine-tuning and LoRA comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from constituency_insights.nlp.finetuning import (
    generate_finetuning_report,
    load_finetuning_configs,
    train_finetuning_run,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    configs = load_finetuning_configs(args.config)
    for config in configs:
        result = train_finetuning_run(config)
        print(f"{config.name}: macro-F1={result.best_macro_f1:.6f} run={result.run_id}")
    print(f"Report: {generate_finetuning_report(configs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
