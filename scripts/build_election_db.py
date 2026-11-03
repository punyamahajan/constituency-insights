#!/usr/bin/env python
"""Build the local constituency-election SQLite database."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from constituency_insights.data.elections import (
    BuildPaths,
    ElectionDataError,
    build_database,
)


def main(argv: list[str] | None = None) -> int:
    defaults = BuildPaths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=defaults.results)
    parser.add_argument("--demographics", type=Path, default=defaults.demographics)
    parser.add_argument("--mappings", type=Path, default=defaults.mappings)
    parser.add_argument("--database", type=Path, default=defaults.database)
    parser.add_argument("--review", type=Path, default=defaults.review)
    args = parser.parse_args(argv)
    try:
        counts = build_database(BuildPaths(
            results=args.results, demographics=args.demographics,
            mappings=args.mappings, database=args.database, review=args.review,
        ))
    except ElectionDataError as error:
        sys.stderr.write(f"Error: {error}\n")
        return 1
    print(
        "Built database with "
        f"{counts['constituencies']} constituencies, "
        f"{counts['election_results']} result rows, and "
        f"{counts['demographics']} demographic rows."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
