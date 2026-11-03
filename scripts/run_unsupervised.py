#!/usr/bin/env python
"""Run the YAML-configured unsupervised and semi-supervised stance track."""

from __future__ import annotations

import argparse
from pathlib import Path

from constituency_insights.nlp.unsupervised import (
    generate_unsupervised_report,
    load_unsupervised_config,
    rerun_semi_supervised_track,
    run_unsupervised_track,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--phase", choices=("all", "semi", "report"), default="all"
    )
    args = parser.parse_args()
    config = load_unsupervised_config(args.config)
    if args.phase == "semi":
        report = rerun_semi_supervised_track(config)
    elif args.phase == "report":
        report = generate_unsupervised_report(config)
    else:
        report = run_unsupervised_track(config)
    print(f"Generated: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
