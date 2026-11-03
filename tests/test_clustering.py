from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from constituency_insights.clustering import (
    CLUSTER_FEATURES, constituency_snapshot, evaluate_cluster_candidates,
    fit_segments, prepare_cluster_matrix, profile_segments,
    segment_error_analysis,
)
from constituency_insights.tabular import (
    ExpandingWindowSplitter, build_classifier, temporal_oof_predictions,
)


def _features() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(19)
    for year_index, year in enumerate([2004, 2009, 2014, 2019]):
        for constituency in range(24):
            group = constituency % 3
            base = group * 15
            rows.append({
                "constituency_id": constituency, "election_year": year,
                "state_name": f"State {constituency % 4}",
                "previous_winner_party": None if year_index == 0 else f"P{group}",
                "target_winner_party": f"P{(group + year_index) % 3}",
                "target_winner_vote_share": 45 + group * 4 + year_index,
                "previous_vote_share": 44 + base + rng.normal(),
                "previous_margin": 3 + base / 3 + rng.normal(),
                "swing": group + rng.normal(scale=.2),
                "incumbent_party_flag": group % 2,
                "turnout_change": group * 2 + rng.normal(scale=.2),
                "population": 100000 + base * 5000 + rng.normal(scale=100),
                "literacy_rate": 55 + base + rng.normal(),
                "urban_share": 15 + base + rng.normal(),
                "scheduled_caste_share": 8 + group + rng.normal(scale=.2),
            })
    return pd.DataFrame(rows)


def test_snapshot_excludes_future_rows_and_uses_latest_eligible_year() -> None:
    frame = _features()
    snapshot = constituency_snapshot(frame, 2014)
    assert set(snapshot.election_year) == {2014}
    future = frame.copy()
    future.loc[future.election_year == 2019, CLUSTER_FEATURES] = 999999
    pdt.assert_frame_equal(snapshot, constituency_snapshot(future, 2014))


def test_cluster_fit_is_deterministic_and_target_blind() -> None:
    snapshot = constituency_snapshot(_features(), 2014)
    first, metadata = fit_segments(snapshot, "kmeans", 3, .9, seed=1729)
    second, _ = fit_segments(snapshot, "kmeans", 3, .9, seed=1729)
    pdt.assert_frame_equal(first, second)
    changed_target = snapshot.copy()
    changed_target["target_winner_party"] = "COMPLETELY DIFFERENT"
    changed, _ = fit_segments(changed_target, "kmeans", 3, .9, seed=1729)
    pdt.assert_frame_equal(first, changed)
    assert "target_winner_party" not in metadata["features"]
    assert "election_year" not in metadata["features"]


def test_all_algorithms_report_selection_criteria() -> None:
    matrix, kept, metadata = prepare_cluster_matrix(
        constituency_snapshot(_features(), 2014), .9
    )
    criteria = evaluate_cluster_candidates(matrix, [2, 3], seed=11, stability_resamples=3)
    assert set(criteria.algorithm) == {"kmeans", "ward", "gmm"}
    assert len(criteria) == 6
    assert criteria.silhouette.between(-1, 1).all()
    assert criteria.stability_ari.between(-1, 1).all()
    assert criteria[criteria.algorithm == "gmm"].bic.notna().all()
    assert kept == metadata["kept_features"]


def test_profiles_and_seeded_error_intervals_are_reproducible() -> None:
    snapshot = constituency_snapshot(_features(), 2014)
    assignments, _ = fit_segments(snapshot, "kmeans", 3, .9, seed=7)
    profile, parties = profile_segments(snapshot, assignments)
    assert profile["size"].sum() == len(snapshot)
    assert np.allclose(parties.groupby("cluster").share.sum(), 1)
    oof = pd.DataFrame({
        "constituency_id": np.repeat(np.arange(24), 2),
        "election_year": np.tile([2009, 2014], 24),
        "is_error": np.tile([False, True, False], 16),
    })
    errors_a, test_a = segment_error_analysis(assignments, oof, 13, 100)
    errors_b, test_b = segment_error_analysis(assignments, oof, 13, 100)
    pdt.assert_frame_equal(errors_a, errors_b)
    assert test_a == test_b
    assert ((errors_a.ci_low <= errors_a.error_rate)
            & (errors_a.error_rate <= errors_a.ci_high)).all()


def test_prompt6_oof_predictions_are_strictly_temporal() -> None:
    frame = _features()
    splitter = ExpandingWindowSplitter(2)
    oof = temporal_oof_predictions(
        frame, build_classifier("logistic", {}, 5), splitter
    )
    assert set(oof.election_year) == {2014, 2019}
    assert len(oof) == 48
    assert oof.is_error.dtype == bool
