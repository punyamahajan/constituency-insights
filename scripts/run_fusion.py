"""Run real linked-feature fusion when available, otherwise synthetic validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd
import yaml

from constituency_insights.fusion import (
    AGGREGATE_PREFIXES, fusion_lift_statistics, generate_fusion_report,
    generate_synthetic_fusion_benchmark, load_linked_text_features,
    temporal_fusion_ablation,
)
from constituency_insights.tabular import (
    NUMERIC_FEATURES, ExpandingWindowSplitter, load_feature_table,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    linked_path = config.get("linked_text_csv")
    if linked_path:
        aggregates = load_linked_text_features(Path(linked_path))
        election = load_feature_table(Path(config["database"]))
        frame = election.merge(aggregates, on=["constituency_id", "election_year"], how="inner")
        if frame.empty:
            raise ValueError("The linked text file has no exact constituency-year overlap")
        frame = frame.rename(columns={"target_winner_party": "target"})
        tabular_columns = [column for column in NUMERIC_FEATURES if column in frame]
        text_columns = [column for column in aggregates if column.startswith(AGGREGATE_PREFIXES)]
        synthetic = False
    else:
        settings = config["synthetic"]
        frame, tabular_columns, text_columns = generate_synthetic_fusion_benchmark(
            seed, int(settings["constituencies"]), settings["years"],
            float(settings["text_signal_strength"]),
            float(settings["tabular_signal_strength"]),
        )
        synthetic = True
    splitter = ExpandingWindowSplitter(int(config["cv"]["min_train_years"]))
    predictions, scores = temporal_fusion_ablation(
        frame, tabular_columns, text_columns, splitter, seed, config["early_fusion"]
    )
    lift = {
        model: fusion_lift_statistics(
            predictions, scores, model, seed, int(config["bootstrap"]["resamples"])
        ) for model in ("late_fusion", "early_fusion")
    }
    result = {"synthetic": synthetic, "scores": scores.to_dict(orient="records"),
              "lift": lift, "text_signal_strength": (
                  config["synthetic"]["text_signal_strength"] if synthetic else None
              )}
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "oof_predictions.csv", index=False)
    results_path = output / "results.json"
    results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    generate_fusion_report(result, Path(config["report_path"]))
    Path("mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(config["tracking_uri"])
    mlflow.set_experiment(config["experiment_name"])
    with mlflow.start_run(run_name="synthetic-pipeline-validation" if synthetic else "real-linked-fusion"):
        mlflow.log_params({"seed": seed, "synthetic": synthetic,
                           "text_signal_strength": result["text_signal_strength"]})
        for model, values in lift.items():
            mlflow.log_metric(f"{model}_lift", values["mean_lift"])
        mlflow.log_artifacts(str(output))
        mlflow.log_artifact(config["report_path"])


if __name__ == "__main__":
    main()
