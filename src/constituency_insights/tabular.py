"""Leakage-safe temporal modeling utilities for constituency election features."""

from __future__ import annotations

import json
import random
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "previous_vote_share", "previous_margin", "swing", "incumbent_party_flag",
    "turnout_change", "population", "literacy_rate", "urban_share",
    "scheduled_caste_share",
]
CATEGORICAL_FEATURES = ["state_name", "previous_winner_party"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ExpandingWindowSplitter:
    """Yield train-before-T/validate-on-T indices in chronological order."""

    def __init__(self, min_train_years: int = 2, *, exclude_final_year: bool = False):
        if min_train_years < 1:
            raise ValueError("min_train_years must be positive")
        self.min_train_years = min_train_years
        self.exclude_final_year = exclude_final_year

    def split(self, X: pd.DataFrame, y: Any = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        years = np.sort(X["election_year"].unique())
        validation_years = years[self.min_train_years:]
        if self.exclude_final_year and len(validation_years):
            validation_years = validation_years[:-1]
        for year in validation_years:
            train = np.flatnonzero(X["election_year"].to_numpy() < year)
            valid = np.flatnonzero(X["election_year"].to_numpy() == year)
            if len(train) and len(valid):
                yield train, valid

    def get_n_splits(self, X: pd.DataFrame, y: Any = None) -> int:
        return sum(1 for _ in self.split(X, y))


def load_feature_table(database: Path) -> pd.DataFrame:
    if not database.is_file():
        raise FileNotFoundError(
            f"Feature database not found: {database}. Run scripts/build_election_db.py first."
        )
    with sqlite3.connect(database) as connection:
        frame = pd.read_sql_query(
            "SELECT * FROM election_features ORDER BY election_year, constituency_id",
            connection,
        )
    if frame.empty:
        raise ValueError("election_features is empty")
    return frame


def build_preprocessor() -> ColumnTransformer:
    """Return a fold-fitted transformer; never fit this before CV splitting."""

    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """Small deterministic sklearn-compatible PyTorch classifier."""

    def __init__(self, hidden_size: int = 32, epochs: int = 50,
                 learning_rate: float = 1e-3, batch_size: int = 32,
                 patience: int = 7, seed: int = 1729):
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed

    def fit(self, X: Any, y: Any) -> "TorchMLPClassifier":
        seed_everything(self.seed)
        array = np.asarray(X, dtype=np.float32)
        self.classes_, encoded = np.unique(np.asarray(y), return_inverse=True)
        self.model_ = torch.nn.Sequential(
            torch.nn.Linear(array.shape[1], self.hidden_size), torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, len(self.classes_)),
        )
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(array), torch.from_numpy(encoded.astype(np.int64))
        )
        generator = torch.Generator().manual_seed(self.seed)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, generator=generator
        )
        best_loss, stale = float("inf"), 0
        for _ in range(self.epochs):
            total = 0.0
            self.model_.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.model_(xb), yb)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(xb)
            epoch_loss = total / len(dataset)
            if epoch_loss < best_loss - 1e-7:
                best_loss, stale = epoch_loss, 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        self.model_.eval()
        with torch.no_grad():
            return torch.softmax(
                self.model_(torch.as_tensor(np.asarray(X), dtype=torch.float32)), dim=1
            ).numpy()

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class EncodedXGBClassifier(ClassifierMixin, BaseEstimator):
    """XGBoost wrapper that maps arbitrary party labels to contiguous integers."""

    def __init__(self, n_estimators: int = 300, max_depth: int = 4,
                 learning_rate: float = .05, subsample: float = .9,
                 colsample_bytree: float = .9, seed: int = 1729):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.seed = seed

    def fit(self, X: Any, y: Any) -> "EncodedXGBClassifier":
        from xgboost import XGBClassifier
        self.classes_, encoded = np.unique(np.asarray(y), return_inverse=True)
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, subsample=self.subsample,
            colsample_bytree=self.colsample_bytree, random_state=self.seed,
            n_jobs=1,
        )
        self.model_.fit(X, encoded)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.model_.predict_proba(X)

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[self.model_.predict(X).astype(int)]


def build_classifier(name: str, params: dict[str, Any], seed: int) -> Pipeline:
    if name == "logistic":
        estimator: Any = LogisticRegression(
            C=float(params.get("C", 1.0)), max_iter=2000, random_state=seed
        )
    elif name == "ridge":
        estimator = RidgeClassifier(alpha=float(params.get("alpha", 1.0)))
    elif name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=params.get("max_depth"), random_state=seed, n_jobs=1,
        )
    elif name == "xgboost":
        estimator = EncodedXGBClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=int(params.get("max_depth", 4)),
            learning_rate=float(params.get("learning_rate", .05)),
            subsample=float(params.get("subsample", .9)),
            colsample_bytree=float(params.get("colsample_bytree", .9)),
            seed=seed,
        )
    elif name == "mlp":
        estimator = TorchMLPClassifier(seed=seed, **params)
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])


def previous_winner_predictions(frame: pd.DataFrame) -> np.ndarray:
    """Persistence baseline, falling back to train-only majority is handled per fold."""
    return frame["previous_winner_party"].to_numpy()


def temporal_cv_scores(
    frame: pd.DataFrame, model: Pipeline, splitter: ExpandingWindowSplitter
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for train_idx, valid_idx in splitter.split(frame):
        train, valid = frame.iloc[train_idx], frame.iloc[valid_idx]
        started = time.perf_counter()
        model.fit(train[FEATURES], train["target_winner_party"])
        prediction = model.predict(valid[FEATURES])
        rows.append({
            "year": int(valid["election_year"].iloc[0]),
            "accuracy": float(accuracy_score(valid["target_winner_party"], prediction)),
            "seconds": time.perf_counter() - started,
        })
    return pd.DataFrame(rows)


def temporal_oof_predictions(
    frame: pd.DataFrame, model: Pipeline, splitter: ExpandingWindowSplitter
) -> pd.DataFrame:
    """Return predictions made strictly out of fold by election year."""
    rows: list[pd.DataFrame] = []
    for train_idx, valid_idx in splitter.split(frame):
        train, valid = frame.iloc[train_idx], frame.iloc[valid_idx]
        model.fit(train[FEATURES], train["target_winner_party"])
        prediction = model.predict(valid[FEATURES])
        rows.append(pd.DataFrame({
            "constituency_id": valid["constituency_id"].to_numpy(),
            "election_year": valid["election_year"].to_numpy(),
            "true_party": valid["target_winner_party"].to_numpy(),
            "predicted_party": prediction,
            "is_error": prediction != valid["target_winner_party"].to_numpy(),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=[
        "constituency_id", "election_year", "true_party", "predicted_party", "is_error"
    ])


def persistence_cv_scores(frame: pd.DataFrame,
                          splitter: ExpandingWindowSplitter) -> pd.DataFrame:
    """Score 'previous winner wins again', with train-only majority fallback."""
    rows = []
    for train_idx, valid_idx in splitter.split(frame):
        train, valid = frame.iloc[train_idx], frame.iloc[valid_idx]
        majority = train["target_winner_party"].mode().iloc[0]
        prediction = valid["previous_winner_party"].fillna(majority)
        rows.append({"year": int(valid.election_year.iloc[0]),
                     "accuracy": float(accuracy_score(valid.target_winner_party, prediction)),
                     "seconds": 0.0})
    return pd.DataFrame(rows)


def paired_fold_comparison(a: np.ndarray, b: np.ndarray) -> dict[str, float | str]:
    difference = np.asarray(a) - np.asarray(b)
    result: dict[str, float | str] = {
        "mean_difference": float(difference.mean()),
        "std_difference": float(difference.std(ddof=1)) if len(difference) > 1 else 0.0,
        "test": "Wilcoxon signed-rank",
    }
    try:
        result["p_value"] = float(wilcoxon(difference).pvalue)
    except ValueError:
        result["p_value"] = 1.0
    return result


def calibration_metrics(y_true: np.ndarray, probability: np.ndarray,
                        classes: np.ndarray, bins: int = 10) -> dict[str, Any]:
    """Top-label reliability and multiclass Brier score."""
    probability = np.asarray(probability)
    predicted = probability.argmax(axis=1)
    truth = np.array([np.flatnonzero(classes == value)[0] for value in y_true])
    confidence = probability.max(axis=1)
    correct = (predicted == truth).astype(int)
    fraction, mean = calibration_curve(correct, confidence, n_bins=bins, strategy="quantile")
    one_hot = np.eye(len(classes))[truth]
    return {
        "brier": float(np.mean(np.sum((probability - one_hot) ** 2, axis=1))),
        "fraction_positive": fraction.tolist(),
        "mean_predicted": mean.tolist(),
    }


def compare_calibrators(estimator: Pipeline, X_train: pd.DataFrame,
                        y_train: np.ndarray, X_valid: pd.DataFrame,
                        y_valid: np.ndarray, bins: int = 10) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for label, method in (("platt", "sigmoid"), ("isotonic", "isotonic")):
        calibrated = CalibratedClassifierCV(estimator, method=method, cv=3)
        calibrated.fit(X_train, y_train)
        output[label] = calibration_metrics(
            y_valid, calibrated.predict_proba(X_valid), calibrated.classes_, bins
        )
    return output


def save_reliability_plot(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot([0, 1], [0, 1], "--", color="grey", label="ideal")
    for label, values in metrics.items():
        axis.plot(values["mean_predicted"], values["fraction_positive"], marker="o", label=label)
    axis.set(xlabel="Mean confidence", ylabel="Observed top-label accuracy",
             title="Winner classifier reliability")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def save_shap_explanations(model: Pipeline, frame: pd.DataFrame,
                           output_dir: Path, count: int = 5) -> list[Path]:
    """Save one global and up to five local SHAP plots without source-row text."""
    import shap
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = frame.iloc[: min(100, len(frame))]
    transformed = model.named_steps["preprocess"].transform(sample[FEATURES])
    estimator = model.named_steps["model"]
    if isinstance(estimator, EncodedXGBClassifier):
        estimator = estimator.model_
    explainer = shap.Explainer(estimator, transformed)
    values = explainer(transformed)
    paths: list[Path] = []
    global_path = output_dir / "shap_global.png"
    shap.plots.bar(values, show=False, max_display=15)
    plt.tight_layout(); plt.savefig(global_path, dpi=140); plt.close()
    paths.append(global_path)
    for index in range(min(count, len(sample))):
        path = output_dir / f"shap_constituency_{int(sample.iloc[index].constituency_id)}.png"
        local = values[index]
        if len(local.shape) > 1:
            local = local[:, int(np.argmax(model.predict_proba(sample.iloc[[index]][FEATURES])[0]))]
        shap.plots.waterfall(local, show=False, max_display=15)
        plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()
        paths.append(path)
    return paths


@dataclass(frozen=True)
class ConformalResult:
    nominal_coverage: float
    empirical_coverage: float
    mean_width: float
    quantile: float


def split_conformal_vote_share(train: pd.DataFrame, calibration: pd.DataFrame,
                               test: pd.DataFrame, nominal_coverage: float) -> ConformalResult:
    """Fit before calibration, calibrate later, then evaluate on a later year."""
    if not (train.election_year.max() < calibration.election_year.min()
            <= calibration.election_year.max() < test.election_year.min()):
        raise ValueError("Conformal train/calibration/test periods must be strictly ordered")
    model = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", Ridge(alpha=1.0)),
    ])
    model.fit(train[FEATURES], train["target_winner_vote_share"])
    residual = np.abs(
        calibration["target_winner_vote_share"].to_numpy()
        - model.predict(calibration[FEATURES])
    )
    n = len(residual)
    level = min(1.0, np.ceil((n + 1) * nominal_coverage) / n)
    quantile = float(np.quantile(residual, level, method="higher"))
    prediction = model.predict(test[FEATURES])
    truth = test["target_winner_vote_share"].to_numpy()
    covered = (truth >= prediction - quantile) & (truth <= prediction + quantile)
    return ConformalResult(nominal_coverage, float(covered.mean()), 2 * quantile, quantile)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def generate_report(results_path: Path, report_path: Path) -> None:
    """Generate the committed report exclusively from code-produced JSON."""
    result = json.loads(results_path.read_text(encoding="utf-8"))
    lines = ["# Tabular election modeling report", "",
             "> Generated by `scripts/run_tabular_ml.py`; numbers are read from its logged output.", "",
             "## Validation protocol", "",
             "Expanding-window validation trains only on elections before each validation year. "
             "Preprocessing is fitted inside each fold. The final election year is excluded from tuning and evaluated once at the end.", "",
             "## Cross-validation", "", "| Model | Mean accuracy | Std |", "|---|---:|---:|"]
    for name, values in result["cross_validation"].items():
        lines.append(f"| {name} | {values['mean']:.4f} | {values['std']:.4f} |")
    lines += ["", "The paired comparison uses a Wilcoxon signed-rank test. With few election-year folds, power is low, folds are not independent in the usual IID sense, and the p-value is descriptive rather than definitive.", "",
              f"Selected-model minus persistence-baseline mean difference: `{result['paired']['mean_difference']:.4f}`; paired p-value: `{result['paired']['p_value']:.4f}`.", "",
              "## Calibration", "", f"Binary Brier scores: Platt `{result['calibration']['platt']['brier']:.4f}`, isotonic `{result['calibration']['isotonic']['brier']:.4f}`.", "",
              "## Vote-share uncertainty", "", f"Split-conformal nominal coverage was `{result['conformal']['nominal_coverage']:.3f}` and empirical coverage was `{result['conformal']['empirical_coverage']:.3f}`; mean interval width was `{result['conformal']['mean_width']:.3f}`. Coverage is reported as observed and is not adjusted upward when below nominal.", "",
              "## Interpretability", "", "SHAP global and five local constituency explanations are stored as MLflow image artifacts. These are associations, not causal effects.", "",
              "## Final held-out election year (evaluated once)", ""]
    heldout = result.get("final_heldout")
    if heldout:
        lines.append(f"Year `{heldout['year']}` was evaluated once after model selection. Accuracy: `{heldout['accuracy']:.4f}`.")
    else:
        lines.append("Not evaluated: no real local database was available, or the explicit `--evaluate-final-once` gate was not supplied.")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_pending_report(report_path: Path) -> None:
    """Generate an honest no-data report without inventing experiment results."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# Tabular election modeling report\n\n"
        "> Generated by `constituency_insights.tabular.generate_pending_report`.\n\n"
        "The real-data report has not been generated because the documented TCPD "
        "and demographic source files are not present locally. No election "
        "observations, validation metrics, conformal coverage, or final held-out "
        "result have been fabricated.\n\n"
        "Run `python scripts/run_tabular_ml.py --config configs/tabular_ml.yaml` "
        "for development evaluation. Only after model selection is complete, add "
        "`--evaluate-final-once`; that gated action evaluates the latest election "
        "year and creates a local lock preventing an accidental repeat.\n",
        encoding="utf-8",
    )
