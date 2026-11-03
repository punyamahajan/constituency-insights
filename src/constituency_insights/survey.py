"""Weighted survey estimation, raking, and simplified MRP utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2
import statsmodels.api as sm


class SurveyDataError(ValueError):
    """Raised when a survey input violates its declared schema."""


@dataclass(frozen=True)
class SurveySchema:
    respondent_id: str
    demographics: tuple[str, ...]
    weight: str
    outcome: str


@dataclass(frozen=True)
class WeightedProportion:
    estimate: float
    ci_low: float
    ci_high: float
    effective_n: float
    design_effect: float
    unweighted_n: int


@dataclass(frozen=True)
class WeightedCrosstab:
    table: pd.DataFrame
    chi_square: float
    adjusted_chi_square: float
    degrees_of_freedom: int
    p_value: float
    design_effect: float
    method: str


@dataclass(frozen=True)
class RakingResult:
    weights: np.ndarray
    converged: bool
    iterations: int
    max_relative_error: float


@dataclass(frozen=True)
class WeightedLogitResult:
    coefficients: pd.Series
    standard_errors: pd.Series
    converged: bool


def load_survey(path: Path, schema: SurveySchema) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Survey CSV not found: {path}. See data/survey/README.md; no data is downloaded automatically."
        )
    frame = pd.read_csv(path)
    required = {schema.respondent_id, schema.weight, schema.outcome, *schema.demographics}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SurveyDataError(f"Survey CSV is missing required columns: {missing}")
    if frame[schema.respondent_id].isna().any() or frame[schema.respondent_id].duplicated().any():
        raise SurveyDataError("Respondent IDs must be non-missing and unique")
    weights = pd.to_numeric(frame[schema.weight], errors="coerce")
    if weights.isna().any() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise SurveyDataError("Survey weights must be finite and strictly positive")
    if frame[[schema.outcome, *schema.demographics]].isna().any().any():
        raise SurveyDataError("Outcome and configured demographic columns cannot be missing")
    frame = frame.copy()
    frame[schema.weight] = weights
    return frame


def generate_synthetic_survey(seed: int, n: int = 600) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create labeled demo microdata and its known poststratification population."""
    rng = np.random.default_rng(seed)
    cells = pd.DataFrame([
        (age, gender, region, constituency, count)
        for constituency, region, count in [("C1", "north", 7000), ("C2", "south", 9000), ("C3", "west", 6000)]
        for age in ("18-34", "35+") for gender in ("F", "M")
    ], columns=["age_group", "gender", "region", "constituency", "population"])
    cells["population"] = cells["population"] // 4
    logit = (-.4 + .8 * (cells.age_group == "35+").astype(float)
             + .35 * (cells.gender == "F").astype(float)
             + cells.constituency.map({"C1": -.35, "C2": .2, "C3": .55}))
    cells["true_probability"] = 1 / (1 + np.exp(-logit))
    # Deliberately disproportionate sampling; the weight corrects back to cells.
    sampling = np.where(cells.age_group.eq("18-34"), 2.0, .7)
    probability = cells.population.to_numpy() * sampling
    probability = probability / probability.sum()
    selected = rng.choice(len(cells), size=n, p=probability)
    survey = cells.iloc[selected].reset_index(drop=True).copy()
    survey["vote_intention"] = rng.binomial(1, survey.true_probability)
    sample_counts = pd.Series(selected).value_counts()
    survey["weight"] = [cells.iloc[index].population / sample_counts[index] for index in selected]
    survey.insert(0, "respondent_id", [f"SYNTH-{index:05d}" for index in range(n)])
    survey["synthetic_demo"] = True
    survey = survey.drop(columns=["population", "true_probability"])
    return survey, cells


def kish_design_effect(weights: Sequence[float]) -> tuple[float, float]:
    values = np.asarray(weights, dtype=float)
    if len(values) == 0 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Weights must be a non-empty vector of finite positive values")
    effective_n = float(values.sum() ** 2 / np.square(values).sum())
    return float(len(values) / effective_n), effective_n


def weighted_proportion(outcome: Sequence[float], weights: Sequence[float],
                        confidence: float = .95) -> WeightedProportion:
    y, w = np.asarray(outcome, dtype=float), np.asarray(weights, dtype=float)
    if len(y) != len(w) or len(y) == 0 or not np.isin(y, [0, 1]).all():
        raise ValueError("Outcome must be a non-empty binary vector matching weights")
    design_effect, effective_n = kish_design_effect(w)
    estimate = float(np.average(y, weights=w))
    z = NormalDist().inv_cdf(.5 + confidence / 2)
    # Wilson interval with Kish effective sample size.
    denominator = 1 + z * z / effective_n
    center = (estimate + z * z / (2 * effective_n)) / denominator
    half = z * np.sqrt(estimate * (1 - estimate) / effective_n
                       + z * z / (4 * effective_n ** 2)) / denominator
    return WeightedProportion(estimate, max(0.0, center - half), min(1.0, center + half),
                              effective_n, design_effect, len(y))


def weighted_crosstab(frame: pd.DataFrame, row: str, column: str,
                      weight: str) -> WeightedCrosstab:
    normalized = frame.copy()
    normalized["__analysis_weight"] = (
        frame[weight].astype(float) * len(frame) / frame[weight].sum()
    )
    table = pd.pivot_table(normalized, values="__analysis_weight", index=row, columns=column,
                           aggfunc="sum", fill_value=0, observed=False)
    observed = table.to_numpy(dtype=float)
    expected = observed.sum(axis=1, keepdims=True) @ observed.sum(axis=0, keepdims=True) / observed.sum()
    statistic = float(np.sum(np.square(observed - expected) / expected))
    design_effect, _ = kish_design_effect(frame[weight])
    adjusted = statistic / design_effect
    degrees = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    return WeightedCrosstab(table, statistic, adjusted, degrees,
                            float(chi2.sf(adjusted, degrees)), design_effect,
                            "Pearson chi-square divided by Kish design effect; not a full Rao-Scott correction")


def weighted_logistic_regression(X: pd.DataFrame, outcome: Sequence[int],
                                 weights: Sequence[float]) -> WeightedLogitResult:
    design = sm.add_constant(X.astype(float), has_constant="add")
    y, w = np.asarray(outcome, dtype=float), np.asarray(weights, dtype=float)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Weighted logistic regression requires a binary outcome")
    kish_design_effect(w)
    analysis_weights = w * len(w) / w.sum()
    fit = sm.GLM(y, design, family=sm.families.Binomial(),
                 freq_weights=analysis_weights).fit()
    return WeightedLogitResult(fit.params, fit.bse, bool(fit.converged))


def rake_weights(frame: pd.DataFrame, initial_weights: Sequence[float],
                 margins: Mapping[str, Mapping[Any, float]], tolerance: float = 1e-6,
                 max_iterations: int = 500) -> RakingResult:
    weights = np.asarray(initial_weights, dtype=float).copy()
    kish_design_effect(weights)  # validates weights
    total = float(weights.sum())
    targets: dict[str, dict[Any, float]] = {}
    for variable, margin in margins.items():
        if variable not in frame:
            raise ValueError(f"Raking variable is missing: {variable}")
        if set(frame[variable].unique()) - set(margin):
            raise ValueError(f"Population margin for {variable} does not cover sampled categories")
        margin_total = float(sum(margin.values()))
        if margin_total <= 0:
            raise ValueError("Population margins must have positive totals")
        targets[variable] = {category: total * value / margin_total
                             for category, value in margin.items()}
    max_error = float("inf")
    for iteration in range(1, max_iterations + 1):
        for variable, margin in targets.items():
            values = frame[variable].to_numpy()
            for category, target in margin.items():
                mask = values == category
                current = float(weights[mask].sum())
                if current <= 0:
                    raise ValueError(f"Cannot rake empty category {variable}={category!r}")
                weights[mask] *= target / current
        errors = []
        for variable, margin in targets.items():
            for category, target in margin.items():
                actual = weights[frame[variable].to_numpy() == category].sum()
                errors.append(abs(actual - target) / target)
        max_error = float(max(errors, default=0.0))
        if max_error <= tolerance:
            return RakingResult(weights, True, iteration, max_error)
    return RakingResult(weights, False, max_iterations, max_error)


@dataclass
class SimplifiedMRP:
    """Penalized-logit partial-pooling approximation implemented with statsmodels."""
    columns: list[str]
    fit: Any
    demographic_columns: tuple[str, ...]
    constituency_column: str

    def poststratify(self, cells: pd.DataFrame, population_column: str) -> pd.DataFrame:
        design = pd.get_dummies(
            cells[[*self.demographic_columns, self.constituency_column]],
            drop_first=False, dtype=float,
        )
        design = sm.add_constant(design, has_constant="add").reindex(
            columns=self.columns, fill_value=0.0
        )
        cells = cells.copy()
        cells["predicted_probability"] = self.fit.predict(design)
        rows = []
        for constituency, group in cells.groupby(self.constituency_column):
            estimate = np.average(group.predicted_probability,
                                  weights=group[population_column])
            rows.append({self.constituency_column: constituency,
                         "mrp_estimate": float(estimate),
                         "population": float(group[population_column].sum())})
        return pd.DataFrame(rows)


def fit_simplified_mrp(survey: pd.DataFrame, outcome: str, weight: str,
                       demographic_columns: Sequence[str], constituency_column: str,
                       regularization: float = .1) -> SimplifiedMRP:
    predictors = [*demographic_columns, constituency_column]
    design = pd.get_dummies(survey[predictors], drop_first=False, dtype=float)
    design = sm.add_constant(design, has_constant="add")
    analysis_weights = survey[weight] * len(survey) / survey[weight].sum()
    model = sm.GLM(survey[outcome].astype(float), design,
                   family=sm.families.Binomial(), freq_weights=analysis_weights)
    # L2 penalty shrinks sparse constituency/cell coefficients toward zero.
    alpha = np.full(design.shape[1], regularization)
    alpha[0] = 0.0
    fit = model.fit_regularized(alpha=alpha, L1_wt=0.0, maxiter=1000)
    return SimplifiedMRP(list(design.columns), fit, tuple(demographic_columns),
                         constituency_column)


def generate_survey_report(results: Mapping[str, Any], report_path: Path) -> None:
    label = ("Synthetic methodology demonstration" if results["is_synthetic_demo"]
             else "User-supplied survey analysis")
    proportion = results["weighted_proportion"]
    lines = ["# Survey analysis report", "", f"**{label}.**", ""]
    if results["is_synthetic_demo"]:
        lines.append("This report uses generated synthetic respondents and is not evidence about any real population or election.\n")
    lines += ["## Weighted estimate", "",
              f"Weighted vote-intention proportion: `{proportion['estimate']:.4f}` "
              f"(design-effect-aware 95% CI `{proportion['ci_low']:.4f}`–`{proportion['ci_high']:.4f}`).",
              f"Kish effective sample size: `{proportion['effective_n']:.2f}`; design effect: `{proportion['design_effect']:.3f}`.", "",
              "## Weighted cross-tab", "",
              f"Design-effect-adjusted chi-square: `{results['crosstab']['adjusted_chi_square']:.4f}`, p-value: `{results['crosstab']['p_value']:.4f}`.",
              "This is a documented Kish-adjusted Pearson approximation, not a full Rao–Scott correction because PSU and stratum identifiers are not in the minimal schema.", "",
              "## Raking", "",
              f"Converged: `{results['raking']['converged']}` after `{results['raking']['iterations']}` iterations; maximum relative margin error `{results['raking']['max_relative_error']:.8f}`.", "",
              "## Weighted logistic regression", "", "| Term | Coefficient | Standard error |", "|---|---:|---:|"]
    for term, coefficient in results["logistic"]["coefficients"].items():
        lines.append(f"| {term} | {coefficient:.4f} | {results['logistic']['standard_errors'][term]:.4f} |")
    lines += ["", "## Simplified MRP", "",
              "The demonstration uses statsmodels penalized binomial GLM with demographic indicators and L2-shrunk constituency effects as a pragmatic partial-pooling approximation, followed by poststratification. It is not a fully Bayesian multilevel model.", "",
              "| Constituency | MRP estimate |" + (" Known synthetic truth | Absolute error |" if results["is_synthetic_demo"] else ""),
              "|---|---:|" + ("---:|---:|" if results["is_synthetic_demo"] else "")]
    for row in results["mrp"]:
        constituency = row.get("constituency", row.get("constituency_id", ""))
        if results["is_synthetic_demo"]:
            lines.append(f"| {constituency} | {row['mrp_estimate']:.4f} | {row['true_value']:.4f} | {row['absolute_error']:.4f} |")
        else:
            lines.append(f"| {constituency} | {row['mrp_estimate']:.4f} |")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
