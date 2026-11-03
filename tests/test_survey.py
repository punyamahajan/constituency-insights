from __future__ import annotations

from math import log
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from constituency_insights.survey import (
    SurveyDataError, SurveySchema, fit_simplified_mrp,
    generate_synthetic_survey, kish_design_effect, load_survey, rake_weights,
    weighted_crosstab, weighted_logistic_regression, weighted_proportion,
)


def test_weighted_proportion_matches_hand_calculation() -> None:
    result = weighted_proportion([1, 0, 1], [1, 2, 1])
    assert result.estimate == .5
    assert result.effective_n == pytest.approx(16 / 6)
    assert result.design_effect == pytest.approx(3 / (16 / 6))
    assert result.ci_low < result.estimate < result.ci_high


def test_weighted_crosstab_matches_hand_pearson_with_equal_weights() -> None:
    frame = pd.DataFrame({
        "group": ["A"] * 10 + ["B"] * 10,
        "outcome": [1] * 8 + [0] * 2 + [1] * 2 + [0] * 8,
        "weight": np.ones(20),
    })
    result = weighted_crosstab(frame, "group", "outcome", "weight")
    # Expected cells are all 5: 4 * (3^2 / 5) = 7.2.
    assert result.chi_square == pytest.approx(7.2)
    assert result.adjusted_chi_square == pytest.approx(7.2)
    assert result.design_effect == 1
    assert result.degrees_of_freedom == 1


def test_weighted_intercept_logit_equals_log_odds_of_weighted_share() -> None:
    outcome = np.array([1, 0, 1, 0])
    weights = np.array([3, 1, 1, 1])
    design = pd.DataFrame(index=range(4))
    result = weighted_logistic_regression(design, outcome, weights)
    expected_share = 4 / 6
    assert result.coefficients["const"] == pytest.approx(
        log(expected_share / (1 - expected_share)), abs=1e-6
    )
    assert result.converged
    rescaled = weighted_logistic_regression(design, outcome, weights * 1000)
    assert rescaled.coefficients["const"] == pytest.approx(result.coefficients["const"])
    assert rescaled.standard_errors["const"] == pytest.approx(result.standard_errors["const"])


def test_crosstab_inference_is_invariant_to_weight_scale() -> None:
    frame = pd.DataFrame({"group": ["A", "A", "B", "B"],
                          "outcome": [0, 1, 0, 1], "weight": [1, 3, 2, 1]})
    original = weighted_crosstab(frame, "group", "outcome", "weight")
    frame["weight"] *= 500
    scaled = weighted_crosstab(frame, "group", "outcome", "weight")
    assert original.adjusted_chi_square == pytest.approx(scaled.adjusted_chi_square)
    assert original.p_value == pytest.approx(scaled.p_value)


def test_raking_recovers_known_margins() -> None:
    frame = pd.DataFrame({
        "age": ["young", "young", "old", "old"],
        "gender": ["F", "M", "F", "M"],
    })
    result = rake_weights(
        frame, np.ones(4),
        {"age": {"young": .25, "old": .75}, "gender": {"F": .6, "M": .4}},
        tolerance=1e-10,
    )
    assert result.converged
    for variable, expected in {
        "age": {"young": 1.0, "old": 3.0},
        "gender": {"F": 2.4, "M": 1.6},
    }.items():
        actual = pd.Series(result.weights).groupby(frame[variable]).sum().to_dict()
        assert actual == pytest.approx(expected, abs=1e-8)


def test_loader_validates_schema_and_positive_weights(tmp_path: Path) -> None:
    schema = SurveySchema("id", ("age",), "weight", "stance")
    path = tmp_path / "survey.csv"
    pd.DataFrame({"id": [1, 2], "age": ["a", "b"],
                  "weight": [1, 2], "stance": [0, 1]}).to_csv(path, index=False)
    assert len(load_survey(path, schema)) == 2
    pd.DataFrame({"id": [1], "age": ["a"], "weight": [0],
                  "stance": [1]}).to_csv(path, index=False)
    with pytest.raises(SurveyDataError, match="strictly positive"):
        load_survey(path, schema)


def test_synthetic_generator_is_deterministic_and_clearly_labeled() -> None:
    first, population_a = generate_synthetic_survey(41, n=100)
    second, population_b = generate_synthetic_survey(41, n=100)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(population_a, population_b)
    assert first.synthetic_demo.all()
    assert first.respondent_id.str.startswith("SYNTH-").all()


def test_simplified_mrp_recovers_known_synthetic_population_reasonably() -> None:
    survey, cells = generate_synthetic_survey(1729, n=2500)
    model = fit_simplified_mrp(
        survey, "vote_intention", "weight", ["age_group", "gender", "region"],
        "constituency", regularization=.02,
    )
    estimates = model.poststratify(cells, "population")
    truth = (cells.assign(total=lambda x: x.true_probability * x.population)
             .groupby("constituency").agg(total=("total", "sum"), pop=("population", "sum")))
    truth["truth"] = truth.total / truth["pop"]
    checked = estimates.merge(truth[["truth"]], on="constituency")
    assert np.mean(abs(checked.mrp_estimate - checked.truth)) < .10
