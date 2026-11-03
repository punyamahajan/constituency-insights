"""Run cutoff-aware constituency clustering and OOF error analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

from constituency_insights.clustering import (
    choose_candidate, constituency_snapshot, evaluate_cluster_candidates,
    fit_segments, generate_clustering_report, profile_segments,
    segment_error_analysis,
)
from constituency_insights.tabular import load_feature_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    clustering = config["clustering"]
    oof_path = Path(clustering["oof_predictions"])
    if not oof_path.is_file():
        raise FileNotFoundError(
            f"Prompt 6 OOF predictions not found: {oof_path}. Run scripts/run_tabular_ml.py first."
        )
    oof = pd.read_csv(oof_path)
    required = {"constituency_id", "election_year", "is_error"}
    if not required.issubset(oof.columns):
        raise ValueError(f"OOF file is missing columns: {sorted(required - set(oof.columns))}")
    cutoff = int(oof.election_year.max())
    snapshot = constituency_snapshot(load_feature_table(Path(config["database"])), cutoff)
    from constituency_insights.clustering import prepare_cluster_matrix
    matrix, _, transform = prepare_cluster_matrix(snapshot, float(clustering["correlation_threshold"]))
    criteria = evaluate_cluster_candidates(
        matrix, list(clustering["k_values"]), int(config["seed"]),
        int(clustering["stability_resamples"]),
    )
    algorithm = clustering["chosen_algorithm"]
    k = choose_candidate(criteria, algorithm)
    assignments, metadata = fit_segments(
        snapshot, algorithm, k, float(clustering["correlation_threshold"]), int(config["seed"])
    )
    profiles, parties = profile_segments(snapshot, assignments)
    errors, test = segment_error_analysis(
        assignments, oof, int(config["seed"]), int(clustering["bootstrap_resamples"])
    )
    output = Path(clustering["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    criteria.to_csv(output / "criteria.csv", index=False)
    assignments.to_csv(output / "assignments.csv", index=False)
    profiles.to_csv(output / "profiles.csv", index=False)
    errors.to_csv(output / "errors.csv", index=False)
    records = criteria.replace({np.nan: None}).to_dict(orient="records")
    result = {"cutoff_year": cutoff, "criteria": records,
              "selected": {"algorithm": algorithm, "k": k, **metadata},
              "transform": transform, "profiles": profiles.to_dict(orient="records"),
              "party_distribution": parties.to_dict(orient="records"),
              "errors": errors.to_dict(orient="records"), "test": test}
    results_path = output / "results.json"
    results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    generate_clustering_report(results_path, Path(clustering["report_path"]))
    mlflow.set_tracking_uri(config["tracking_uri"])
    mlflow.set_experiment(config["experiment_name"] + "-clustering")
    with mlflow.start_run(run_name=f"{algorithm}-k{k}"):
        mlflow.log_params({"seed": config["seed"], "algorithm": algorithm,
                           "k": k, "cutoff_year": cutoff})
        mlflow.log_metric("kruskal_p_value", test["kruskal_p_value"])
        mlflow.log_artifacts(str(output))
        mlflow.log_artifact(clustering["report_path"])


if __name__ == "__main__":
    main()
