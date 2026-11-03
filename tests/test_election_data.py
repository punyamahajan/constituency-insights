from __future__ import annotations

import csv
import json
from pathlib import Path
import sqlite3
import uuid

import pytest

from constituency_insights.data.elections import (
    BuildPaths,
    ElectionDataError,
    UnmatchedConstituencyError,
    build_database,
    validate_database,
)


def _write_csv(path: Path, columns: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def _fixture_paths(tmp_path: Path, *, unmatched: bool = False) -> BuildPaths:
    results = tmp_path / "results.csv"
    demographics = tmp_path / "demographics.csv"
    mappings = tmp_path / "mappings.csv"
    source_name = "Unreviewed Alpha" if unmatched else "Alpha Source"
    _write_csv(
        mappings,
        ["state_name", "source_name", "canonical_name", "valid_from_year", "valid_to_year", "mapping_type"],
        [["State One", "Alpha Source", "Alpha", "", "", "same"]],
    )
    result_rows = []
    elections = [
        (2004, "Party A", 60.0, "Party B", 40.0, 55.0),
        (2009, "Party B", 55.0, "Party A", 45.0, 60.0),
        (2014, "Party B", 52.0, "Party A", 48.0, 63.0),
    ]
    for year, winner, winner_share, runner, runner_share, turnout in elections:
        result_rows.extend([
            ["State One", source_name, year, f"Winner {year}", winner, 600, winner_share, turnout, 1],
            ["State One", source_name, year, f"Runner {year}", runner, 400, runner_share, turnout, 0],
        ])
    _write_csv(
        results,
        ["state_name", "constituency_name", "election_year", "candidate_name", "party", "votes", "vote_share", "turnout", "is_winner"],
        result_rows,
    )
    _write_csv(
        demographics,
        ["state_name", "canonical_name", "census_year", "population", "literacy_rate", "urban_share", "scheduled_caste_share"],
        [
            ["State One", "Alpha", 2001, 1000, 70, 30, 15],
            ["State One", "Alpha", 2011, 1200, 75, 35, 16],
        ],
    )
    identifier = uuid.uuid4().hex
    return BuildPaths(
        results=results,
        demographics=demographics,
        mappings=mappings,
        database=Path(f"outputs/local/test_elections_{identifier}.sqlite"),
        review=Path(f"outputs/local/test_unmatched_{identifier}.csv"),
    )


def _cleanup(paths: BuildPaths) -> None:
    paths.database.unlink(missing_ok=True)
    paths.review.unlink(missing_ok=True)


def test_schema_builds_from_scratch_and_lag_is_null_for_first_election(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    try:
        counts = build_database(paths)
        assert counts == {"constituencies": 1, "election_results": 6, "demographics": 2}
        connection = sqlite3.connect(paths.database)
        rows = connection.execute(
            "SELECT election_year, previous_vote_share, previous_margin, swing, "
            "incumbent_party_flag, turnout_change, demographics_source_year "
            "FROM election_features ORDER BY election_year"
        ).fetchall()
        connection.close()

        assert rows[0] == (2004, None, None, None, None, None, 2001)
        assert rows[1] == (2009, 60.0, 20.0, None, None, None, 2001)
        assert rows[2] == (2014, 55.0, 10.0, -5.0, 0, 5.0, 2011)
    finally:
        _cleanup(paths)


def test_future_rows_do_not_change_features_for_earlier_years(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    try:
        build_database(paths)
        connection = sqlite3.connect(paths.database)
        before = connection.execute(
            "SELECT previous_vote_share, previous_margin, swing, turnout_change, "
            "demographics_source_year FROM election_features WHERE election_year = 2014"
        ).fetchone()
        constituency_id = connection.execute(
            "SELECT constituency_id FROM constituency"
        ).fetchone()[0]
        connection.executemany(
            "INSERT INTO election_result (constituency_id, election_year, candidate_name, party, votes, vote_share, turnout, is_winner) VALUES (?, 2019, ?, ?, ?, ?, 65, ?)",
            [
                (constituency_id, "Future winner", "Party C", 700, 58, 1),
                (constituency_id, "Future runner", "Party B", 500, 42, 0),
            ],
        )
        connection.execute(
            "INSERT INTO demographics (constituency_id, census_year, population, literacy_rate, urban_share, scheduled_caste_share) VALUES (?, 2021, 1400, 80, 40, 17)",
            (constituency_id,),
        )
        connection.commit()
        after = connection.execute(
            "SELECT previous_vote_share, previous_margin, swing, turnout_change, "
            "demographics_source_year FROM election_features WHERE election_year = 2014"
        ).fetchone()
        validate_database(connection)
        connection.close()
        assert after == before
    finally:
        _cleanup(paths)


def test_constraints_reject_duplicate_keys_and_invalid_vote_share(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    try:
        build_database(paths)
        connection = sqlite3.connect(paths.database)
        constituency_id = connection.execute("SELECT constituency_id FROM constituency").fetchone()[0]
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO election_result (constituency_id, election_year, candidate_name, party, votes, vote_share, turnout, is_winner) VALUES (?, 2004, 'Winner 2004', 'Party A', 1, 50, 50, 0)",
                (constituency_id,),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO election_result (constituency_id, election_year, candidate_name, party, votes, vote_share, turnout, is_winner) VALUES (?, 2024, 'Invalid', 'Party Z', 1, 101, 50, 1)",
                (constituency_id,),
            )
        connection.close()
    finally:
        _cleanup(paths)


def test_missing_inputs_fail_with_actionable_message(tmp_path: Path) -> None:
    paths = BuildPaths(
        results=tmp_path / "missing.csv",
        demographics=tmp_path / "demographics.csv",
        mappings=tmp_path / "mappings.csv",
        database=Path("outputs/local/missing_test.sqlite"),
        review=Path("outputs/local/missing_review.csv"),
    )
    with pytest.raises(ElectionDataError, match="data/README.md"):
        build_database(paths)


def test_fuzzy_match_is_review_only_and_never_loaded(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path, unmatched=True)
    try:
        with pytest.raises(UnmatchedConstituencyError, match="no election results were loaded"):
            build_database(paths)
        review = paths.review.read_text(encoding="utf-8")
        assert "Unreviewed Alpha" in review
        assert "Alpha" in review
        connection = sqlite3.connect(paths.database)
        assert connection.execute("SELECT COUNT(*) FROM election_result").fetchone()[0] == 0
        connection.close()
    finally:
        _cleanup(paths)


def test_election_eda_notebook_stays_under_ten_cells() -> None:
    notebook = json.loads(
        Path("notebooks/01_election_eda.ipynb").read_text(encoding="utf-8")
    )
    assert len(notebook["cells"]) < 10
    assert [cell["cell_type"] for cell in notebook["cells"]].count("code") == 5
