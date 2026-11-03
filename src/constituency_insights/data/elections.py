"""Strict local CSV-to-SQLite loader for constituency election modeling."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable

from rapidfuzz import fuzz, process


DEFAULT_RESULTS = Path("data/raw/elections/lok_dhaba_results.csv")
DEFAULT_DEMOGRAPHICS = Path("data/raw/demographics/constituency_demographics.csv")
DEFAULT_MAPPINGS = Path("data/raw/elections/constituency_name_mapping.csv")
DEFAULT_DATABASE = Path("outputs/local/constituency_elections.sqlite")
DEFAULT_REVIEW = Path("outputs/local/unmatched_constituencies.csv")

RESULT_COLUMNS = {
    "state_name", "constituency_name", "election_year", "candidate_name",
    "party", "votes", "vote_share", "turnout", "is_winner",
}
DEMOGRAPHIC_COLUMNS = {
    "state_name", "canonical_name", "census_year", "population",
    "literacy_rate", "urban_share", "scheduled_caste_share",
}
MAPPING_COLUMNS = {
    "state_name", "source_name", "canonical_name", "valid_from_year",
    "valid_to_year", "mapping_type",
}


class ElectionDataError(ValueError):
    """Raised when local election data is missing or invalid."""


class UnmatchedConstituencyError(ElectionDataError):
    """Raised after unmatched names have been written for human review."""


@dataclass(frozen=True)
class BuildPaths:
    results: Path = DEFAULT_RESULTS
    demographics: Path = DEFAULT_DEMOGRAPHICS
    mappings: Path = DEFAULT_MAPPINGS
    database: Path = DEFAULT_DATABASE
    review: Path = DEFAULT_REVIEW


def normalize_name(value: str) -> str:
    """Normalize spacing and punctuation for exact reviewed-map lookup only."""

    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _require_local_output(path: Path, description: str) -> None:
    root = Path("outputs/local").resolve()
    resolved = path.resolve()
    if root != resolved and root not in resolved.parents:
        raise ElectionDataError(
            f"{description} must be stored under outputs/local/ because it may "
            "contain source text: {path}"
        )


def _read_csv(path: Path, required: set[str], description: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise ElectionDataError(
            f"Missing {description}: {path}. Follow data/README.md to obtain "
            "and prepare the source locally; nothing is downloaded automatically."
        )
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = sorted(required.difference(columns))
        if missing:
            raise ElectionDataError(
                f"{description} is missing required column(s) {missing}: {path}"
            )
        return [dict(row) for row in reader]


def _optional_int(value: str, field: str) -> int | None:
    if value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ElectionDataError(f"Invalid integer for {field}: {value!r}") from exc


def _required_int(value: str, field: str) -> int:
    parsed = _optional_int(value, field)
    if parsed is None:
        raise ElectionDataError(f"Missing required integer for {field}.")
    return parsed


def _optional_float(value: str, field: str) -> float | None:
    if value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ElectionDataError(f"Invalid number for {field}: {value!r}") from exc


def _required_float(value: str, field: str) -> float:
    parsed = _optional_float(value, field)
    if parsed is None:
        raise ElectionDataError(f"Missing required number for {field}.")
    return parsed


def apply_sql(connection: sqlite3.Connection, path: Path) -> None:
    """Execute a repository SQL file as a script."""

    if not path.is_file():
        raise ElectionDataError(f"Required SQL file is missing: {path}")
    connection.executescript(path.read_text(encoding="utf-8"))


def _load_mappings(
    connection: sqlite3.Connection, rows: list[dict[str, str]]
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    canonical_ids: dict[tuple[str, str, int | None], int] = {}
    lookup: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        state = row["state_name"].strip()
        source = row["source_name"].strip()
        canonical = row["canonical_name"].strip()
        mapping_type = row["mapping_type"].strip()
        if not all((state, source, canonical, mapping_type)):
            raise ElectionDataError("Mapping rows require non-empty names and mapping_type.")
        valid_from = _optional_int(row["valid_from_year"], "valid_from_year")
        valid_to = _optional_int(row["valid_to_year"], "valid_to_year")
        key = (state, canonical, valid_from)
        if key not in canonical_ids:
            cursor = connection.execute(
                "INSERT INTO constituency "
                "(canonical_name, state_name, valid_from_year, valid_to_year) "
                "VALUES (?, ?, ?, ?)",
                (canonical, state, valid_from, valid_to),
            )
            canonical_ids[key] = int(cursor.lastrowid)
        constituency_id = canonical_ids[key]
        connection.execute(
            "INSERT INTO constituency_name_mapping "
            "(constituency_id, state_name, source_name, valid_from_year, "
            "valid_to_year, mapping_type) VALUES (?, ?, ?, ?, ?, ?)",
            (constituency_id, state, source, valid_from, valid_to, mapping_type),
        )
        lookup.setdefault((normalize_name(state), normalize_name(source)), []).append({
            "constituency_id": constituency_id,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "canonical_name": canonical,
        })
    return lookup


def _resolve_mapping(
    lookup: dict[tuple[str, str], list[dict[str, Any]]],
    state: str,
    source: str,
    year: int,
) -> int | None:
    candidates = lookup.get((normalize_name(state), normalize_name(source)), [])
    valid = [
        item for item in candidates
        if (item["valid_from"] is None or year >= item["valid_from"])
        and (item["valid_to"] is None or year <= item["valid_to"])
    ]
    if len(valid) > 1:
        raise ElectionDataError(
            f"Ambiguous reviewed mapping for {state!r}/{source!r} in {year}."
        )
    return int(valid[0]["constituency_id"]) if valid else None


def _write_fuzzy_review(
    review_path: Path,
    unmatched: set[tuple[str, str, int]],
    lookup: dict[tuple[str, str], list[dict[str, Any]]],
) -> None:
    review_path.parent.mkdir(parents=True, exist_ok=True)
    state_choices: dict[str, dict[str, str]] = {}
    for (normalized_state, _source), mappings in lookup.items():
        choices = state_choices.setdefault(normalized_state, {})
        for item in mappings:
            canonical = str(item["canonical_name"])
            choices[normalize_name(canonical)] = canonical
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "state_name", "source_name", "election_year",
            "suggested_canonical_name", "fuzzy_score", "status",
        ])
        writer.writeheader()
        for state, source, year in sorted(unmatched):
            choices = state_choices.get(normalize_name(state), {})
            match = process.extractOne(
                normalize_name(source), choices.keys(), scorer=fuzz.WRatio
            ) if choices else None
            suggested = choices[match[0]] if match else ""
            score = round(float(match[1]), 2) if match else ""
            writer.writerow({
                "state_name": state, "source_name": source,
                "election_year": year, "suggested_canonical_name": suggested,
                "fuzzy_score": score, "status": "REVIEW_REQUIRED",
            })


def _load_results(
    connection: sqlite3.Connection,
    rows: list[dict[str, str]],
    lookup: dict[tuple[str, str], list[dict[str, Any]]],
    review_path: Path,
) -> int:
    resolved: list[tuple[int, dict[str, str], int]] = []
    unmatched: set[tuple[str, str, int]] = set()
    for row in rows:
        state = row["state_name"].strip()
        source = row["constituency_name"].strip()
        year = _required_int(row["election_year"], "election_year")
        constituency_id = _resolve_mapping(lookup, state, source, year)
        if constituency_id is None:
            unmatched.add((state, source, year))
        else:
            resolved.append((constituency_id, row, year))
    if unmatched:
        _write_fuzzy_review(review_path, unmatched, lookup)
        raise UnmatchedConstituencyError(
            f"{len(unmatched)} constituency name/year combination(s) require "
            f"manual review. Suggestions were written to {review_path}; no "
            "fuzzy match was accepted and no election results were loaded."
        )
    review_path.unlink(missing_ok=True)
    for constituency_id, row, year in resolved:
        connection.execute(
            "INSERT INTO election_result "
            "(constituency_id, election_year, candidate_name, party, votes, "
            "vote_share, turnout, is_winner) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                constituency_id, year, row["candidate_name"].strip(),
                row["party"].strip(), _required_int(row["votes"], "votes"),
                _required_float(row["vote_share"], "vote_share"),
                _optional_float(row["turnout"], "turnout"),
                _required_int(row["is_winner"], "is_winner"),
            ),
        )
    return len(resolved)


def _load_demographics(
    connection: sqlite3.Connection,
    rows: list[dict[str, str]],
) -> int:
    for row in rows:
        match = connection.execute(
            "SELECT constituency_id FROM constituency "
            "WHERE lower(state_name) = lower(?) AND lower(canonical_name) = lower(?) "
            "ORDER BY valid_from_year DESC LIMIT 1",
            (row["state_name"].strip(), row["canonical_name"].strip()),
        ).fetchone()
        if match is None:
            raise ElectionDataError(
                "Demographic row has no reviewed canonical constituency: "
                f"{row['state_name']!r}/{row['canonical_name']!r}."
            )
        connection.execute(
            "INSERT INTO demographics "
            "(constituency_id, census_year, population, literacy_rate, "
            "urban_share, scheduled_caste_share) VALUES (?, ?, ?, ?, ?, ?)",
            (
                int(match[0]), _required_int(row["census_year"], "census_year"),
                _optional_int(row["population"], "population"),
                _optional_float(row["literacy_rate"], "literacy_rate"),
                _optional_float(row["urban_share"], "urban_share"),
                _optional_float(row["scheduled_caste_share"], "scheduled_caste_share"),
            ),
        )
    return len(rows)


def validate_database(connection: sqlite3.Connection) -> None:
    """Validate relational keys, ranges, winners, and feature provenance."""

    duplicate_results = connection.execute(
        "SELECT COUNT(*) FROM ("
        "SELECT constituency_id, election_year, candidate_name, party, COUNT(*) n "
        "FROM election_result GROUP BY 1,2,3,4 HAVING n > 1)"
    ).fetchone()[0]
    if duplicate_results:
        raise ElectionDataError("Duplicate election-result keys detected.")
    invalid_shares = connection.execute(
        "SELECT COUNT(*) FROM election_result "
        "WHERE vote_share < 0 OR vote_share > 100"
    ).fetchone()[0]
    if invalid_shares:
        raise ElectionDataError("Vote shares must be between 0 and 100.")
    invalid_winners = connection.execute(
        "SELECT COUNT(*) FROM ("
        "SELECT constituency_id, election_year, SUM(is_winner) winners "
        "FROM election_result GROUP BY 1,2 HAVING winners != 1)"
    ).fetchone()[0]
    if invalid_winners:
        raise ElectionDataError("Each constituency-year must have exactly one winner.")
    future_demographics = connection.execute(
        "SELECT COUNT(*) FROM election_features "
        "WHERE demographics_source_year > election_year"
    ).fetchone()[0]
    if future_demographics:
        raise ElectionDataError("Future demographic data leaked into features.")


def build_database(
    paths: BuildPaths,
    *,
    schema_path: Path = Path("sql/schema.sql"),
    features_path: Path = Path("sql/features.sql"),
) -> dict[str, int]:
    """Build a fresh local SQLite database from explicitly supplied files."""

    _require_local_output(paths.database, "Database")
    _require_local_output(paths.review, "Unmatched-name review file")
    results = _read_csv(paths.results, RESULT_COLUMNS, "Lok Dhaba results CSV")
    demographics = _read_csv(
        paths.demographics, DEMOGRAPHIC_COLUMNS, "demographics CSV"
    )
    mappings = _read_csv(paths.mappings, MAPPING_COLUMNS, "name-mapping CSV")
    paths.database.parent.mkdir(parents=True, exist_ok=True)
    if paths.database.exists():
        paths.database.unlink()
    connection = sqlite3.connect(paths.database)
    try:
        apply_sql(connection, schema_path)
        with connection:
            lookup = _load_mappings(connection, mappings)
            result_count = _load_results(
                connection, results, lookup, paths.review
            )
            demographic_count = _load_demographics(connection, demographics)
            apply_sql(connection, features_path)
            validate_database(connection)
        return {
            "constituencies": connection.execute(
                "SELECT COUNT(*) FROM constituency"
            ).fetchone()[0],
            "election_results": result_count,
            "demographics": demographic_count,
        }
    finally:
        connection.close()
