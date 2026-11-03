#!/usr/bin/env python
"""Evaluate a saved stance model on a user-provided multilingual CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from constituency_insights.nlp.multilingual_eval import evaluate_csv_checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args(argv)
    metrics = evaluate_csv_checkpoint(
        args.csv, args.checkpoint, batch_size=args.batch_size, device=args.device
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
