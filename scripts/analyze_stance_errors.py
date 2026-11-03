#!/usr/bin/env python
"""Generate validation-only stance error and calibration analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from constituency_insights.nlp.rigorous_error_analysis import (
    load_error_analysis_config,
    run_error_analysis,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    report = run_error_analysis(load_error_analysis_config(args.config))
    print(f"Generated: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
