"""Run leakage-safe tabular election experiments and optionally unlock final evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np
import optuna
import yaml
from sklearn.metrics import accuracy_score

from constituency_insights.tabular import (
    FEATURES, ExpandingWindowSplitter, build_classifier, compare_calibrators,
    generate_report, load_feature_table, paired_fold_comparison,
    persistence_cv_scores, save_json, save_reliability_plot,
    save_shap_explanations, seed_everything,
    split_conformal_vote_share, temporal_cv_scores, temporal_oof_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evaluate-final-once", action="store_true",
                        help="Irreversibly evaluate the latest year after tuning; do not rerun for selection.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    seed_everything(seed)
    frame = load_feature_table(Path(config["database"]))
    final_year = int(frame.election_year.max())
    development = frame[frame.election_year < final_year].copy()
    splitter = ExpandingWindowSplitter(int(config["cv"]["min_train_years"]))
    mlflow.set_tracking_uri(config["tracking_uri"])
    mlflow.set_experiment(config["experiment_name"])
    output = Path(config["output_dir"])
    lock = output / f"FINAL_EVALUATED_{final_year}.lock"
    model_names = ["ridge", "logistic", "random_forest", "xgboost", "mlp"]
    base_params = {
        "ridge": {}, "logistic": {},
        "random_forest": {"n_estimators": int(config["models"]["random_forest"]["n_estimators"])},
        "xgboost": {"n_estimators": int(config["models"]["xgboost"]["n_estimators"])},
        "mlp": config["models"]["mlp"],
    }
    fold_results = {}
    persistence = persistence_cv_scores(development, splitter)
    fold_results["previous_winner"] = {
        "mean": float(persistence.accuracy.mean()),
        "std": float(persistence.accuracy.std(ddof=0)),
        "folds": persistence.to_dict(orient="records"),
    }
    with mlflow.start_run(run_name="cv-previous-winner"):
        mlflow.log_params({"model": "previous_winner", "seed": seed})
        mlflow.log_metric("mean_cv_accuracy", fold_results["previous_winner"]["mean"])
    for model_name in model_names:
        with mlflow.start_run(run_name=f"cv-{model_name}"):
            mlflow.log_params({"model": model_name, "seed": seed, **base_params[model_name]})
            model = build_classifier(model_name, base_params[model_name], seed)
            scores = temporal_cv_scores(development, model, splitter)
            fold_results[model_name] = {
                "mean": float(scores.accuracy.mean()), "std": float(scores.accuracy.std(ddof=0)),
                "folds": scores.to_dict(orient="records"),
            }
            mlflow.log_metric("mean_cv_accuracy", fold_results[model_name]["mean"])
    # Optuna trial logging is nested and uses development years only.
    def objective(trial: optuna.Trial) -> float:
        tuning = config["tuning"]
        name = trial.suggest_categorical("model", tuning["model_choices"])
        if name == "logistic":
            params = {"C": trial.suggest_float("C", *tuning["logistic_c"], log=True)}
        elif name == "random_forest":
            low, high, step = tuning["random_forest_n_estimators"]
            params = {"n_estimators": trial.suggest_int("rf_n_estimators", low, high, step=step),
                      "max_depth": trial.suggest_int("rf_max_depth", *tuning["random_forest_max_depth"])}
        elif name == "xgboost":
            low, high, step = tuning["xgboost_n_estimators"]
            params = {"n_estimators": trial.suggest_int("xgb_n_estimators", low, high, step=step),
                      "max_depth": trial.suggest_int("xgb_max_depth", *tuning["xgboost_max_depth"]),
                      "learning_rate": trial.suggest_float("xgb_learning_rate", *tuning["xgboost_learning_rate"], log=True)}
        else:
            params = {"hidden_size": trial.suggest_categorical("hidden_size", tuning["mlp_hidden_sizes"]),
                      "epochs": int(config["models"]["mlp"]["epochs"]),
                      "learning_rate": trial.suggest_float("mlp_learning_rate", *tuning["mlp_learning_rate"], log=True),
                      "batch_size": int(config["models"]["mlp"]["batch_size"]),
                      "patience": int(config["models"]["mlp"]["patience"])}
        trial.set_user_attr("model_params", params)
        with mlflow.start_run(run_name=f"{name}-trial-{trial.number}"):
            mlflow.log_params({"model": name, **params, "seed": seed})
            scores = temporal_cv_scores(development, build_classifier(name, params, seed), splitter)
            for step, row in scores.iterrows():
                mlflow.log_metric("fold_accuracy", float(row.accuracy), step=int(step))
                trial.report(float(scores.accuracy.iloc[:step + 1].mean()), step=int(step))
                if trial.should_prune():
                    raise optuna.TrialPruned()
            mean = float(scores.accuracy.mean())
            mlflow.log_metric("mean_cv_accuracy", mean)
            return mean
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed), pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=int(config["tuning"]["trials"]), timeout=config["tuning"]["timeout_seconds"])
    best_name = str(study.best_params["model"])
    best_params = dict(study.best_trial.user_attrs["model_params"])
    tuned_scores = temporal_cv_scores(
        development, build_classifier(best_name, best_params, seed), splitter
    )
    selected_key = f"tuned_{best_name}"
    fold_results[selected_key] = {
        "mean": float(tuned_scores.accuracy.mean()),
        "std": float(tuned_scores.accuracy.std(ddof=0)),
        "folds": tuned_scores.to_dict(orient="records"),
    }
    oof_path = output / "oof_predictions.csv"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    temporal_oof_predictions(
        development, build_classifier(best_name, best_params, seed), splitter
    ).to_csv(oof_path, index=False)
    paired = paired_fold_comparison(
        np.array([x["accuracy"] for x in fold_results[selected_key]["folds"]]),
        np.array([x["accuracy"] for x in fold_results["previous_winner"]["folds"]]),
    )
    # Later-year calibration and conformal partitions are chronological.
    years = sorted(development.election_year.unique())
    if len(years) < 3:
        raise ValueError("At least three development election years are required")
    conformal = split_conformal_vote_share(
        development[development.election_year < years[-2]],
        development[development.election_year == years[-2]],
        development[development.election_year == years[-1]],
        float(config["conformal"]["nominal_coverage"]),
    )
    calibration_train = development[development.election_year < years[-1]]
    calibration_valid = development[development.election_year == years[-1]]
    calibration = compare_calibrators(
        build_classifier("logistic", best_params if best_name == "logistic" else {}, seed),
        calibration_train[FEATURES], calibration_train.target_winner_party.to_numpy(),
        calibration_valid[FEATURES], calibration_valid.target_winner_party.to_numpy(),
        int(config["calibration"]["bins"]),
    )
    reliability_path = output / "reliability.png"
    save_reliability_plot(calibration, reliability_path)
    result = {"cross_validation": fold_results, "paired": paired,
              "calibration": calibration, "conformal": conformal.__dict__, "final_heldout": None}
    if args.evaluate_final_once:
        if lock.exists():
            raise RuntimeError(f"Final year {final_year} was already evaluated; lock exists: {lock}")
        model = build_classifier(best_name, best_params, seed)
        model.fit(development[FEATURES], development.target_winner_party)
        final = frame[frame.election_year == final_year]
        result["final_heldout"] = {"year": final_year, "accuracy": float(accuracy_score(final.target_winner_party, model.predict(final[FEATURES]))), "model": best_name}
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text("Final held-out evaluation completed once.\n", encoding="utf-8")
    explanation_model = build_classifier("random_forest", base_params["random_forest"], seed)
    explanation_model.fit(development[FEATURES], development.target_winner_party)
    with mlflow.start_run(run_name="tabular-summary"):
        mlflow.log_artifact(str(reliability_path), artifact_path="calibration")
        for path in save_shap_explanations(
            explanation_model, development[development.election_year == years[-1]],
            output / "shap", int(config["interpretability"]["local_examples"]),
        ):
            mlflow.log_artifact(str(path), artifact_path="shap")
    results_path = output / "results.json"
    save_json(results_path, result)
    generate_report(results_path, Path(config["report_path"]))
    with mlflow.start_run(run_name="tabular-report"):
        mlflow.log_params({"selected_model": best_name, "seed": seed,
                           "final_year_evaluated": bool(result["final_heldout"])})
        mlflow.log_metric("conformal_empirical_coverage", conformal.empirical_coverage)
        mlflow.log_metric("conformal_nominal_coverage", conformal.nominal_coverage)
        if result["final_heldout"]:
            mlflow.log_metric("final_heldout_accuracy", result["final_heldout"]["accuracy"])
        mlflow.log_artifact(str(results_path))
        mlflow.log_artifact(str(oof_path))
        mlflow.log_artifact(str(config["report_path"]))


if __name__ == "__main__":
    main()
