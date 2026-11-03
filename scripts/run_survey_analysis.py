"""Generate the weighted survey methodology report from configured data."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

from constituency_insights.survey import (
    SurveySchema, fit_simplified_mrp, generate_survey_report,
    generate_synthetic_survey, load_survey, rake_weights, weighted_crosstab,
    weighted_logistic_regression, weighted_proportion,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--population-csv", type=Path,
                        help="Required for real data; demographic cells plus population column")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    schema_config = config["schema"]
    schema = SurveySchema(schema_config["respondent_id"],
                          tuple(schema_config["demographics"]),
                          schema_config["weight"], schema_config["outcome"])
    synthetic = bool(config["is_synthetic_demo"])
    if synthetic:
        survey, population = generate_synthetic_survey(int(config["seed"]))
        input_path = Path(config["input_csv"])
        input_path.parent.mkdir(parents=True, exist_ok=True)
        survey.to_csv(input_path, index=False)
        population.to_csv(input_path.with_name("survey_population_demo.csv"), index=False)
    else:
        if args.population_csv is None:
            raise ValueError("--population-csv is required for a real survey analysis")
        population = pd.read_csv(args.population_csv)
    survey = load_survey(Path(config["input_csv"]), schema)
    outcome, weight = schema.outcome, schema.weight
    analysis = config["analysis"]
    estimate = weighted_proportion(survey[outcome], survey[weight])
    first_predictor = analysis["logistic_predictors"][0]
    cross = weighted_crosstab(survey, first_predictor, outcome, weight)
    model_frame = pd.get_dummies(
        survey[analysis["logistic_predictors"]], drop_first=True, dtype=float
    )
    logistic = weighted_logistic_regression(model_frame, survey[outcome], survey[weight])
    margins = {}
    population_count = analysis["population_count"]
    for variable in analysis["raking_margin_variables"]:
        margins[variable] = population.groupby(variable)[population_count].sum().to_dict()
    raked = rake_weights(survey, np.ones(len(survey)), margins,
                         float(config["raking"]["tolerance"]),
                         int(config["raking"]["max_iterations"]))
    mrp = fit_simplified_mrp(
        survey, outcome, weight, analysis["mrp_demographics"],
        analysis["constituency"], float(config["mrp"]["regularization"]),
    ).poststratify(population, population_count)
    if synthetic:
        constituency = analysis["constituency"]
        truth = (population.assign(
            weighted_truth=lambda x: x.true_probability * x[population_count]
        ).groupby(constituency).agg(
            weighted_truth=("weighted_truth", "sum"),
            population=(population_count, "sum"),
        ))
        truth["true_value"] = truth.weighted_truth / truth.population
        mrp = mrp.merge(truth[["true_value"]], on=constituency)
        mrp["absolute_error"] = abs(mrp.mrp_estimate - mrp.true_value)
    mrp = mrp.rename(columns={analysis["constituency"]: "constituency"})
    result = {
        "is_synthetic_demo": synthetic,
        "weighted_proportion": asdict(estimate),
        "crosstab": {key: value for key, value in asdict(cross).items() if key != "table"},
        "raking": {key: value for key, value in asdict(raked).items() if key != "weights"},
        "logistic": {"coefficients": logistic.coefficients.to_dict(),
                     "standard_errors": logistic.standard_errors.to_dict(),
                     "converged": logistic.converged},
        "mrp": mrp.to_dict(orient="records"),
    }
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.json"
    results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    generate_survey_report(result, Path(config["report_path"]))
    Path("mlruns").mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(config["tracking_uri"])
    mlflow.set_experiment(config["experiment_name"])
    with mlflow.start_run(run_name="synthetic-demo" if synthetic else "real-survey"):
        mlflow.log_params({"seed": config["seed"], "synthetic_demo": synthetic,
                           "mrp_regularization": config["mrp"]["regularization"]})
        mlflow.log_metric("weighted_proportion", estimate.estimate)
        mlflow.log_metric("raking_max_relative_error", raked.max_relative_error)
        if synthetic:
            mlflow.log_metric("mrp_mean_absolute_error", float(mrp.absolute_error.mean()))
        mlflow.log_artifact(str(results_path))
        mlflow.log_artifact(config["report_path"])


if __name__ == "__main__":
    main()
