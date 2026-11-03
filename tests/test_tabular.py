from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from constituency_insights.tabular import (
    CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, ExpandingWindowSplitter,
    build_classifier, build_preprocessor, calibration_metrics,
    persistence_cv_scores, split_conformal_vote_share,
)


def _frame() -> pd.DataFrame:
    rows = []
    parties = ["A", "B"]
    for year_index, year in enumerate([2000, 2004, 2009, 2014, 2019]):
        for constituency in range(4):
            previous = parties[(year_index + constituency - 1) % 2] if year_index else None
            winner = parties[(year_index + constituency) % 2]
            rows.append({
                "constituency_id": constituency, "election_year": year,
                "state_name": f"S{constituency % 2}",
                "previous_winner_party": previous,
                "target_winner_party": winner,
                "target_winner_vote_share": 50.0 + year_index + constituency,
                "previous_vote_share": np.nan if year_index == 0 else 49 + year_index,
                "previous_margin": np.nan if year_index == 0 else 2 + constituency,
                "swing": np.nan if year_index < 2 else float(year_index),
                "incumbent_party_flag": np.nan if year_index < 2 else constituency % 2,
                "turnout_change": np.nan if year_index < 2 else constituency - 1,
                "population": 1000 + 100 * year_index + constituency,
                "literacy_rate": 60 + constituency,
                "urban_share": 20 + constituency,
                "scheduled_caste_share": 10 + constituency,
            })
    return pd.DataFrame(rows)


def test_expanding_window_never_puts_future_or_validation_year_in_train() -> None:
    frame = _frame().sample(frac=1, random_state=7).reset_index(drop=True)
    splits = list(ExpandingWindowSplitter(min_train_years=2).split(frame))
    assert len(splits) == 3
    for train, valid in splits:
        validation_years = frame.iloc[valid].election_year.unique()
        assert len(validation_years) == 1
        assert frame.iloc[train].election_year.max() < validation_years[0]
        assert not set(train).intersection(valid)


def test_final_year_can_be_excluded_from_development_cv() -> None:
    frame = _frame()
    validation_years = [frame.iloc[v].election_year.iloc[0]
                        for _, v in ExpandingWindowSplitter(2, exclude_final_year=True).split(frame)]
    assert max(validation_years) < frame.election_year.max()


def test_imputer_is_fitted_on_train_fold_not_full_data() -> None:
    """A full-data fit would produce 100; the fold-only median must remain 1."""
    frame = _frame()
    frame.loc[frame.election_year < 2019, "population"] = np.nan
    frame.loc[frame.election_year == 2014, "population"] = 1.0
    frame.loc[frame.election_year == 2019, "population"] = 199.0
    train = frame[frame.election_year < 2019]
    pipeline = build_classifier("logistic", {}, seed=17)
    pipeline.fit(train[FEATURES], train.target_winner_party)
    imputer = pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
    population_index = NUMERIC_FEATURES.index("population")
    assert imputer.statistics_[population_index] == 1.0
    assert frame.population.median() != 1.0


def test_persistence_baseline_uses_train_only_fallback() -> None:
    scores = persistence_cv_scores(_frame(), ExpandingWindowSplitter(2))
    assert list(scores.year) == [2009, 2014, 2019]
    assert scores.accuracy.between(0, 1).all()


def test_multiclass_calibration_brier_matches_hand_computation() -> None:
    probability = np.array([[.8, .2], [.25, .75]])
    metrics = calibration_metrics(np.array(["A", "B"]), probability, np.array(["A", "B"]), bins=2)
    expected = np.mean([(.8 - 1) ** 2 + .2 ** 2, .25 ** 2 + (.75 - 1) ** 2])
    assert metrics["brier"] == pytest.approx(expected)


def test_split_conformal_requires_time_order_and_reports_observed_coverage() -> None:
    frame = _frame()
    result = split_conformal_vote_share(
        frame[frame.election_year < 2014],
        frame[frame.election_year == 2014],
        frame[frame.election_year == 2019], .9,
    )
    assert result.nominal_coverage == .9
    assert 0 <= result.empirical_coverage <= 1
    assert result.mean_width >= 0
    with pytest.raises(ValueError, match="strictly ordered"):
        split_conformal_vote_share(
            frame[frame.election_year < 2014],
            frame[frame.election_year == 2019],
            frame[frame.election_year == 2014], .9,
        )
