"""Target-free constituency segmentation and segment-level error analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


CLUSTER_FEATURES = [
    "previous_vote_share", "previous_margin", "swing", "incumbent_party_flag",
    "turnout_change", "population", "literacy_rate", "urban_share",
    "scheduled_caste_share",
]
ALGORITHMS = ("kmeans", "ward", "gmm")


def constituency_snapshot(frame: pd.DataFrame, cutoff_year: int) -> pd.DataFrame:
    """Select each constituency's latest feature row available by the cutoff."""
    eligible = frame.loc[frame["election_year"] <= cutoff_year].copy()
    if eligible.empty:
        raise ValueError(f"No feature rows exist on or before cutoff {cutoff_year}")
    eligible = eligible.sort_values(["constituency_id", "election_year"])
    return eligible.groupby("constituency_id", as_index=False).tail(1).reset_index(drop=True)


def prepare_cluster_matrix(snapshot: pd.DataFrame, correlation_threshold: float
                           ) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Impute, remove redundant correlated columns, and standardize."""
    if not 0 < correlation_threshold <= 1:
        raise ValueError("correlation_threshold must be in (0, 1]")
    raw = snapshot[CLUSTER_FEATURES]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputed = pd.DataFrame(imputer.fit_transform(raw), columns=CLUSTER_FEATURES)
    correlation = imputed.corr().abs()
    keep: list[str] = []
    for feature in CLUSTER_FEATURES:
        if not any(correlation.loc[feature, prior] > correlation_threshold for prior in keep):
            keep.append(feature)
    scaler = StandardScaler()
    matrix = scaler.fit_transform(imputed[keep])
    metadata = {
        "medians": dict(zip(CLUSTER_FEATURES, imputer.statistics_.tolist())),
        "kept_features": keep,
        "dropped_correlated_features": [x for x in CLUSTER_FEATURES if x not in keep],
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    return matrix, keep, metadata


def _fit_labels(algorithm: str, matrix: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, Any]:
    if algorithm == "kmeans":
        model = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(matrix)
        return model.labels_, model
    if algorithm == "ward":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(matrix)
        return model.labels_, model
    if algorithm == "gmm":
        model = GaussianMixture(n_components=k, n_init=5, random_state=seed).fit(matrix)
        return model.predict(matrix), model
    raise ValueError(f"Unknown clustering algorithm: {algorithm}")


def _nearest_centroid_labels(train_matrix: np.ndarray, train_labels: np.ndarray,
                             matrix: np.ndarray, k: int) -> np.ndarray:
    centroids = np.vstack([train_matrix[train_labels == label].mean(axis=0) for label in range(k)])
    distance = ((matrix[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return distance.argmin(axis=1)


def clustering_stability(algorithm: str, matrix: np.ndarray, k: int,
                         seed: int, resamples: int = 20) -> float:
    """Mean ARI against the full fit over deterministic bootstrap refits."""
    reference, _ = _fit_labels(algorithm, matrix, k, seed)
    rng = np.random.default_rng(seed)
    scores = []
    for iteration in range(resamples):
        indices = rng.choice(len(matrix), size=len(matrix), replace=True)
        sample = matrix[indices]
        labels, model = _fit_labels(algorithm, sample, k, seed + iteration + 1)
        if algorithm in {"kmeans", "gmm"}:
            predicted = model.predict(matrix)
        else:
            predicted = _nearest_centroid_labels(sample, labels, matrix, k)
        scores.append(adjusted_rand_score(reference, predicted))
    return float(np.mean(scores))


def _gap_statistic(algorithm: str, matrix: np.ndarray, labels: np.ndarray,
                   k: int, seed: int, references: int = 10) -> float:
    def dispersion(values: np.ndarray, groups: np.ndarray) -> float:
        total = 0.0
        for label in range(k):
            member = values[groups == label]
            if len(member):
                total += float(((member - member.mean(axis=0)) ** 2).sum())
        return max(total, np.finfo(float).tiny)
    observed = dispersion(matrix, labels)
    rng = np.random.default_rng(seed)
    low, high = matrix.min(axis=0), matrix.max(axis=0)
    reference_logs = []
    for index in range(references):
        reference = rng.uniform(low, high, size=matrix.shape)
        reference_labels, _ = _fit_labels(algorithm, reference, k, seed + index + 1)
        reference_logs.append(np.log(dispersion(reference, reference_labels)))
    return float(np.mean(reference_logs) - np.log(observed))


def evaluate_cluster_candidates(matrix: np.ndarray, k_values: list[int], seed: int,
                                stability_resamples: int = 20) -> pd.DataFrame:
    rows = []
    for algorithm in ALGORITHMS:
        for k in k_values:
            if k < 2 or k >= len(matrix):
                continue
            labels, model = _fit_labels(algorithm, matrix, k, seed)
            rows.append({
                "algorithm": algorithm,
                "k": k,
                "silhouette": float(silhouette_score(matrix, labels)),
                "gap_statistic": _gap_statistic(algorithm, matrix, labels, k, seed),
                "bic": float(model.bic(matrix)) if algorithm == "gmm" else np.nan,
                "stability_ari": clustering_stability(
                    algorithm, matrix, k, seed, stability_resamples
                ),
            })
    return pd.DataFrame(rows)


def choose_candidate(criteria: pd.DataFrame, algorithm: str) -> int:
    """Choose stable high-silhouette k; GMM BIC is retained as a diagnostic."""
    candidates = criteria[criteria.algorithm == algorithm].copy()
    if candidates.empty:
        raise ValueError(f"No valid candidates for {algorithm}")
    candidates["selection_score"] = candidates.silhouette + candidates.stability_ari
    return int(candidates.sort_values(
        ["selection_score", "k"], ascending=[False, True]
    ).iloc[0].k)


def fit_segments(snapshot: pd.DataFrame, algorithm: str, k: int,
                 correlation_threshold: float, seed: int
                 ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit using only declared non-target columns and return stable integer labels."""
    matrix, kept, metadata = prepare_cluster_matrix(snapshot, correlation_threshold)
    labels, _ = _fit_labels(algorithm, matrix, k, seed)
    assignments = snapshot[["constituency_id", "election_year"]].copy()
    assignments["cluster"] = labels.astype(int)
    metadata.update({"algorithm": algorithm, "k": k, "features": kept, "seed": seed})
    return assignments, metadata


def profile_segments(snapshot: pd.DataFrame, assignments: pd.DataFrame
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Profile after fitting; target winner party is never an input to clustering."""
    profiled = snapshot.merge(assignments[["constituency_id", "cluster"]], on="constituency_id")
    means = profiled.groupby("cluster")[CLUSTER_FEATURES].mean().reset_index()
    means.insert(1, "size", profiled.groupby("cluster").size().reindex(means.cluster).to_numpy())
    party = (profiled.groupby(["cluster", "target_winner_party"]).size()
             .rename("count").reset_index())
    party["share"] = party["count"] / party.groupby("cluster")["count"].transform("sum")
    return means, party


def segment_error_analysis(assignments: pd.DataFrame, oof: pd.DataFrame,
                           seed: int, bootstrap_resamples: int = 1000
                           ) -> tuple[pd.DataFrame, dict[str, float]]:
    merged = oof.merge(assignments[["constituency_id", "cluster"]], on="constituency_id", how="inner")
    if merged.empty:
        raise ValueError("OOF predictions do not match any cluster assignments")
    merged["is_error"] = merged["is_error"].astype(bool)
    rng = np.random.default_rng(seed)
    rows = []
    groups = []
    for cluster, group in merged.groupby("cluster"):
        values = group.is_error.astype(float).to_numpy()
        bootstrap = np.array([
            rng.choice(values, size=len(values), replace=True).mean()
            for _ in range(bootstrap_resamples)
        ])
        rows.append({"cluster": int(cluster), "n_predictions": len(values),
                     "error_rate": float(values.mean()),
                     "ci_low": float(np.quantile(bootstrap, .025)),
                     "ci_high": float(np.quantile(bootstrap, .975))})
        groups.append(values)
    if len(groups) > 1:
        statistic, p_value = kruskal(*groups)
    else:
        statistic, p_value = np.nan, np.nan
    return pd.DataFrame(rows), {"kruskal_statistic": float(statistic),
                                "kruskal_p_value": float(p_value)}


def _markdown_table(frame: pd.DataFrame, decimals: int = 3) -> str:
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "|".join("---" for _ in columns) + "|"]
    for row in frame.itertuples(index=False, name=None):
        rendered = [f"{value:.{decimals}f}" if isinstance(value, float) else str(value)
                    for value in row]
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def generate_clustering_report(results_path: Path, report_path: Path) -> None:
    result = json.loads(results_path.read_text(encoding="utf-8"))
    lines = ["# Constituency clustering report", "",
             "> Generated from code-produced clustering and Prompt 6 OOF artifacts.", "",
             "Clustering uses only standardized demographic and lagged historical-voting features available by the analysis cutoff. Winner party and model error are joined only after fitting.", "",
             "## Selection criteria", "", "| Algorithm | k | Silhouette | Gap | BIC | Stability ARI |", "|---|---:|---:|---:|---:|---:|"]
    for row in result["criteria"]:
        bic = "—" if row["bic"] is None else f"{row['bic']:.4f}"
        lines.append(f"| {row['algorithm']} | {row['k']} | {row['silhouette']:.4f} | {row['gap_statistic']:.4f} | {bic} | {row['stability_ari']:.4f} |")
    lines += ["", f"Selected `{result['selected']['algorithm']}` with k=`{result['selected']['k']}`.", "", "## Cluster profiles", ""]
    profile = pd.DataFrame(result["profiles"])
    lines.append(_markdown_table(profile))
    lines += ["", "### Party distribution", "", _markdown_table(pd.DataFrame(result["party_distribution"])), "", "## OOF model error by cluster", "", _markdown_table(pd.DataFrame(result["errors"]), 4), "", f"Kruskal–Wallis statistic: `{result['test']['kruskal_statistic']:.4f}`; p-value: `{result['test']['kruskal_p_value']:.4f}`.", "", "The test has limited power with few clusters/elections, observations from the same constituency may be dependent, and a non-significant result does not establish equal error rates."]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_pending_clustering_report(report_path: Path) -> None:
    """Generate an honest report state when real inputs have not been supplied."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# Constituency clustering report\n\n"
        "> Generated by `constituency_insights.clustering.generate_pending_clustering_report`.\n\n"
        "The clustering analysis has not been run because the local election "
        "database and Prompt 6 out-of-fold prediction artifact are absent. No "
        "cluster-selection criteria, profiles, error rates, confidence intervals, "
        "or significance tests have been fabricated.\n\n"
        "After generating Prompt 6 OOF predictions, run `python "
        "scripts/run_clustering.py --config configs/tabular_ml.yaml`.\n",
        encoding="utf-8",
    )
