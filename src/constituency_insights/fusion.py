"""Leakage-safe tabular/text-feature fusion with synthetic pipeline validation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from constituency_insights.tabular import ExpandingWindowSplitter


LINK_KEYS = ["constituency_id", "election_year"]
AGGREGATE_PREFIXES = ("sentiment_", "stance_", "embedding_")


class FusionDataError(ValueError):
    """Raised when linked text features lack an auditable temporal key."""


def load_linked_text_features(path: Path) -> pd.DataFrame:
    """Load aggregate-only features with exact constituency-year keys."""
    if not path.is_file():
        raise FileNotFoundError(f"Linked aggregate text-feature CSV not found: {path}")
    frame = pd.read_csv(path)
    missing = sorted({*LINK_KEYS, "collection_end_year"} - set(frame.columns))
    if missing:
        raise FusionDataError(f"Linked text features are missing key columns: {missing}")
    forbidden = [column for column in frame if column.casefold() in {"text", "tweet", "raw_text"}]
    if forbidden:
        raise FusionDataError("Raw text columns are forbidden; supply aggregate features only")
    features = [column for column in frame if column.startswith(AGGREGATE_PREFIXES)]
    if not features:
        raise FusionDataError(
            "No aggregate sentiment, stance, or embedding columns were found"
        )
    if frame.duplicated(LINK_KEYS).any():
        raise FusionDataError("Linked text features require unique constituency-year keys")
    if (frame.collection_end_year > frame.election_year).any():
        raise FusionDataError("Text collected after an election cannot predict that election")
    if frame[LINK_KEYS + features].isna().any().any():
        raise FusionDataError("Keys and aggregate text features cannot be missing")
    return frame


def generate_synthetic_fusion_benchmark(
    seed: int, constituencies: int, years: Sequence[int],
    text_signal_strength: float, tabular_signal_strength: float,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Inject a known aggregate text signal; creates no raw or pseudo-real text."""
    rng = np.random.default_rng(seed)
    rows = []
    constituency_effect = rng.normal(0, .35, size=constituencies)
    for year_index, year in enumerate(years):
        for constituency in range(constituencies):
            tabular = rng.normal(size=5)
            text_latent = rng.normal()
            text = np.array([
                text_latent + rng.normal(scale=.35),
                .7 * text_latent + rng.normal(scale=.5),
                text_latent + rng.normal(scale=.45),
                -.4 * text_latent + rng.normal(scale=.6),
            ])
            logit = (tabular_signal_strength * (.7 * tabular[0] - .4 * tabular[1])
                     + text_signal_strength * text_latent
                     + constituency_effect[constituency] + .04 * year_index)
            target = int(rng.random() < 1 / (1 + np.exp(-logit)))
            row = {"constituency_id": constituency, "election_year": year,
                   "target": target, "synthetic_benchmark": True}
            row.update({f"tabular_{index}": value for index, value in enumerate(tabular)})
            row.update({"sentiment_mean": text[0], "stance_support_share": text[1],
                        "embedding_0": text[2], "embedding_1": text[3]})
            rows.append(row)
    return pd.DataFrame(rows), [f"tabular_{i}" for i in range(5)], [
        "sentiment_mean", "stance_support_share", "embedding_0", "embedding_1"
    ]


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class JointFusionNetwork(torch.nn.Module):
    def __init__(self, tabular_dim: int, text_dim: int, class_count: int,
                 tabular_hidden: int, text_hidden: int, joint_hidden: int):
        super().__init__()
        self.tabular = torch.nn.Sequential(
            torch.nn.Linear(tabular_dim, tabular_hidden), torch.nn.ReLU()
        )
        self.text = torch.nn.Sequential(
            torch.nn.Linear(text_dim, text_hidden), torch.nn.ReLU()
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(tabular_hidden + text_hidden, joint_hidden),
            torch.nn.ReLU(), torch.nn.Linear(joint_hidden, class_count),
        )

    def forward(self, tabular: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.tabular(tabular), self.text(text)], dim=1))


class EarlyFusionClassifier:
    """Fold-fitted preprocessing plus separate tabular/text PyTorch towers."""

    def __init__(self, tabular_columns: Sequence[str], text_columns: Sequence[str],
                 seed: int, tabular_hidden: int, text_hidden: int,
                 joint_hidden: int, epochs: int, learning_rate: float,
                 batch_size: int, patience: int):
        self.tabular_columns = list(tabular_columns)
        self.text_columns = list(text_columns)
        self.seed = seed
        self.tabular_hidden = tabular_hidden
        self.text_hidden = text_hidden
        self.joint_hidden = joint_hidden
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.patience = patience

    @staticmethod
    def _transformer() -> Pipeline:
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler())])

    def fit(self, frame: pd.DataFrame, target: Sequence[Any]) -> "EarlyFusionClassifier":
        _seed(self.seed)
        self.tabular_transformer_ = self._transformer()
        self.text_transformer_ = self._transformer()
        tabular = self.tabular_transformer_.fit_transform(frame[self.tabular_columns]).astype(np.float32)
        text = self.text_transformer_.fit_transform(frame[self.text_columns]).astype(np.float32)
        self.classes_, encoded = np.unique(np.asarray(target), return_inverse=True)
        self.model_ = JointFusionNetwork(
            tabular.shape[1], text.shape[1], len(self.classes_),
            self.tabular_hidden, self.text_hidden, self.joint_hidden,
        )
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(tabular), torch.from_numpy(text),
            torch.from_numpy(encoded.astype(np.int64)),
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        best, stale, best_state = float("inf"), 0, None
        for _ in range(self.epochs):
            total = 0.0
            self.model_.train()
            for tabular_batch, text_batch, target_batch in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.model_(tabular_batch, text_batch), target_batch)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(target_batch)
            mean_loss = total / len(dataset)
            if mean_loss < best - 1e-6:
                best, stale = mean_loss, 0
                best_state = {key: value.detach().clone()
                              for key, value in self.model_.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        tabular = self.tabular_transformer_.transform(frame[self.tabular_columns]).astype(np.float32)
        text = self.text_transformer_.transform(frame[self.text_columns]).astype(np.float32)
        self.model_.eval()
        with torch.no_grad():
            indices = self.model_(torch.from_numpy(tabular), torch.from_numpy(text)).argmax(1).numpy()
        return self.classes_[indices]


def _linear_model(columns: Sequence[str], seed: int) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=seed)),
    ])


def temporal_fusion_ablation(
    frame: pd.DataFrame, tabular_columns: Sequence[str], text_columns: Sequence[str],
    splitter: ExpandingWindowSplitter, seed: int, early_parameters: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate three models with identical chronological folds."""
    predictions = []
    scores = []
    for fold, (train_index, valid_index) in enumerate(splitter.split(frame)):
        train, valid = frame.iloc[train_index], frame.iloc[valid_index]
        models: dict[str, Any] = {
            "tabular_only": (_linear_model(tabular_columns, seed), list(tabular_columns)),
            "late_fusion": (_linear_model([*tabular_columns, *text_columns], seed),
                            [*tabular_columns, *text_columns]),
            "early_fusion": (EarlyFusionClassifier(
                tabular_columns, text_columns, seed=seed, **early_parameters
            ), None),
        }
        for name, (model, columns) in models.items():
            if columns is None:
                model.fit(train, train.target)
                predicted = model.predict(valid)
            else:
                model.fit(train[columns], train.target)
                predicted = model.predict(valid[columns])
            correct = predicted == valid.target.to_numpy()
            scores.append({"fold": fold, "year": int(valid.election_year.iloc[0]),
                           "model": name, "accuracy": float(correct.mean())})
            predictions.append(pd.DataFrame({
                "constituency_id": valid.constituency_id.to_numpy(),
                "election_year": valid.election_year.to_numpy(), "model": name,
                "target": valid.target.to_numpy(), "prediction": predicted,
                "correct": correct,
            }))
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(scores)


def fusion_lift_statistics(predictions: pd.DataFrame, scores: pd.DataFrame,
                           comparison_model: str, seed: int,
                           bootstrap_resamples: int = 1000) -> dict[str, float]:
    baseline = predictions[predictions.model == "tabular_only"].set_index(LINK_KEYS)
    comparison = predictions[predictions.model == comparison_model].set_index(LINK_KEYS)
    joined = baseline[["correct"]].rename(columns={"correct": "correct_base"}).join(
        comparison[["correct"]].rename(columns={"correct": "correct_fusion"})
    )
    by_constituency = joined.assign(
        difference=joined.correct_fusion.astype(float) - joined.correct_base.astype(float)
    ).groupby(level="constituency_id").difference.mean()
    rng = np.random.default_rng(seed)
    values = by_constituency.to_numpy()
    bootstrap = np.array([rng.choice(values, len(values), replace=True).mean()
                          for _ in range(bootstrap_resamples)])
    pivot = scores.pivot(index="fold", columns="model", values="accuracy")
    differences = pivot[comparison_model] - pivot["tabular_only"]
    try:
        p_value = float(wilcoxon(differences).pvalue)
    except ValueError:
        p_value = 1.0
    return {"mean_lift": float(joined.correct_fusion.mean() - joined.correct_base.mean()),
            "ci_low": float(np.quantile(bootstrap, .025)),
            "ci_high": float(np.quantile(bootstrap, .975)),
            "paired_wilcoxon_p": p_value,
            "folds": int(len(differences))}


def generate_fusion_report(results: dict[str, Any], report_path: Path) -> None:
    synthetic = bool(results["synthetic"])
    title = "SYNTHETIC: pipeline validation only, not evidence of real-world gains"
    lines = ["# Text–tabular fusion report", "", f"**{title}.**" if synthetic else "**Real linked-data analysis.**", ""]
    if synthetic:
        lines += ["No valid real text-to-constituency-year link exists. The benchmark injects a known aggregate text signal solely to test the fusion and ablation machinery. It does not show that text improves Indian election prediction.", ""]
    lines += ["## Expanding-window results", "", "| Model | Mean accuracy | Standard deviation |", "|---|---:|---:|"]
    score_frame = pd.DataFrame(results["scores"])
    for model, group in score_frame.groupby("model"):
        lines.append(f"| {model} | {group.accuracy.mean():.4f} | {group.accuracy.std(ddof=0):.4f} |")
    lines += ["", "## Lift over tabular only", "", "| Model | Mean lift | Bootstrap 95% CI | Paired Wilcoxon p |", "|---|---:|---:|---:|"]
    for model, values in results["lift"].items():
        lines.append(f"| {model} | {values['mean_lift']:.4f} | [{values['ci_low']:.4f}, {values['ci_high']:.4f}] | {values['paired_wilcoxon_p']:.4f} |")
    lines += ["", "Bootstrap intervals resample constituencies. The paired test uses election-year folds; with few folds it has low power and should be treated as descriptive.", ""]
    if synthetic:
        lines += ["## What this does not show", "", "These results do not establish availability, representativeness, geographic validity, or predictive value of any real text source. No SemEval record was linked to an Indian constituency. A real claim requires an audited user-supplied constituency-year aggregate file and a new time-based evaluation."]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
