"""Unsupervised and semi-supervised experiments on official training rows only."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

from bertopic import BERTopic
import hdbscan
import mlflow
import numpy as np
import pandas as pd
import torch
import umap
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    f1_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from constituency_insights.data import (
    TASK_A_TARGETS,
    ValidationSplit,
    load_task_a,
    official_task_a_file_specs,
    prepare_task_a_validation_split,
)
from constituency_insights.nlp.distilbert import LABEL_ORDER, StanceTextDataset
from constituency_insights.nlp.training import (
    StanceTrainingConfig,
    _build_loaders,
    evaluate,
    seed_everything,
    train_one_epoch,
)


@dataclass(frozen=True)
class UnsupervisedConfig:
    seed: int
    tracking_uri: str
    experiment_name: str
    raw_dir: Path
    report: Path
    embedding_model: str
    embedding_cache: Path
    topic_counts: tuple[int, ...]
    topic_top_words: int
    vectorizer_min_df: int
    vectorizer_max_features: int
    umap_components: int
    umap_neighbors: int
    cluster_counts: tuple[int, ...]
    hdbscan_min_cluster_sizes: tuple[int, ...]
    stability_seeds: int
    label_fractions: tuple[float, ...]
    semi_seeds: tuple[int, ...]
    confidence_threshold: float
    semi_max_epochs: int
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float
    mixed_precision: bool
    max_length: int


def load_unsupervised_config(path: Path | str) -> UnsupervisedConfig:
    """Load the unsupervised track from the repository's single YAML config."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    section = payload["unsupervised"]
    semi = section["semi_supervised"]
    return UnsupervisedConfig(
        seed=int(payload["seed"]), tracking_uri=str(payload["tracking_uri"]),
        experiment_name=str(section["experiment_name"]), raw_dir=Path(payload["raw_dir"]),
        report=Path(section["report"]), embedding_model=str(section["embedding_model"]),
        embedding_cache=Path(section["embedding_cache"]),
        topic_counts=tuple(map(int, section["topic_counts"])),
        topic_top_words=int(section["topic_top_words"]),
        vectorizer_min_df=int(section["vectorizer_min_df"]),
        vectorizer_max_features=int(section["vectorizer_max_features"]),
        umap_components=int(section["umap_components"]),
        umap_neighbors=int(section["umap_neighbors"]),
        cluster_counts=tuple(map(int, section["cluster_counts"])),
        hdbscan_min_cluster_sizes=tuple(map(int, section["hdbscan_min_cluster_sizes"])),
        stability_seeds=int(section["stability_seeds"]),
        label_fractions=tuple(map(float, semi["label_fractions"])),
        semi_seeds=tuple(map(int, semi["seeds"])),
        confidence_threshold=float(semi["confidence_threshold"]),
        semi_max_epochs=int(semi["max_epochs"]),
        train_batch_size=int(semi["train_batch_size"]),
        eval_batch_size=int(semi["eval_batch_size"]),
        gradient_accumulation_steps=int(semi["gradient_accumulation_steps"]),
        learning_rate=float(semi["learning_rate"]), weight_decay=float(semi["weight_decay"]),
        warmup_ratio=float(semi["warmup_ratio"]), max_grad_norm=float(semi["max_grad_norm"]),
        mixed_precision=bool(semi["mixed_precision"]), max_length=int(semi["max_length"]),
    )


def npmi_coherence(binary_matrix: np.ndarray, topics: Sequence[Sequence[int]]) -> float:
    """Mean normalized pointwise mutual information over topic word pairs."""

    matrix = np.asarray(binary_matrix) > 0
    document_count = matrix.shape[0]
    if document_count == 0:
        raise ValueError("Cannot compute coherence without documents.")
    scores: list[float] = []
    for topic in topics:
        for first_position in range(len(topic)):
            for second_position in range(first_position + 1, len(topic)):
                first, second = int(topic[first_position]), int(topic[second_position])
                p_first = matrix[:, first].mean()
                p_second = matrix[:, second].mean()
                p_joint = np.logical_and(matrix[:, first], matrix[:, second]).mean()
                if p_joint == 0 or p_first == 0 or p_second == 0:
                    scores.append(-1.0)
                elif p_joint == 1.0:
                    scores.append(1.0)
                else:
                    scores.append(float(np.log(p_joint / (p_first * p_second)) / -np.log(p_joint)))
    if not scores:
        raise ValueError("Topics must contain at least one word pair.")
    return float(np.mean(scores))


def _training_rows(config: UnsupervisedConfig) -> tuple[pd.DataFrame, ValidationSplit]:
    train_spec, _unused_test_spec = official_task_a_file_specs(config.raw_dir)
    data = load_task_a([train_spec], expected_targets=TASK_A_TARGETS)
    split = prepare_task_a_validation_split(data, random_seed=config.seed)
    return data, split


def compute_or_load_embeddings(
    texts: Sequence[str], config: UnsupervisedConfig
) -> np.ndarray:
    """Compute normalized multilingual sentence embeddings or use local cache."""

    digest = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()
    metadata_path = config.embedding_cache.with_suffix(".json")
    if config.embedding_cache.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("text_sha256") == digest and metadata.get("model") == config.embedding_model:
            return np.load(config.embedding_cache)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    model = SentenceTransformer(config.embedding_model, local_files_only=True)
    embeddings = model.encode(
        list(texts), batch_size=32, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    config.embedding_cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(config.embedding_cache, embeddings)
    metadata_path.write_text(json.dumps({
        "model": config.embedding_model, "rows": len(texts),
        "dimensions": int(embeddings.shape[1]), "text_sha256": digest,
    }, indent=2) + "\n", encoding="utf-8")
    return embeddings


def _log_run(config: UnsupervisedConfig, name: str, params: dict[str, Any], metrics: dict[str, float]) -> str:
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        return run.info.run_id


def run_topic_sweeps(
    texts: Sequence[str], embeddings: np.ndarray, config: UnsupervisedConfig
) -> list[str]:
    """Run quantitative LDA and BERTopic NPMI sweeps."""

    vectorizer = CountVectorizer(
        stop_words="english", min_df=config.vectorizer_min_df,
        max_features=config.vectorizer_max_features,
    )
    counts = vectorizer.fit_transform(texts)
    binary = (counts > 0).astype(np.int8).toarray()
    run_ids: list[str] = []
    for topic_count in config.topic_counts:
        lda = LatentDirichletAllocation(
            n_components=topic_count, random_state=config.seed,
            learning_method="batch", max_iter=20,
        ).fit(counts)
        lda_topics = [
            component.argsort()[-config.topic_top_words:][::-1].tolist()
            for component in lda.components_
        ]
        coherence = npmi_coherence(binary, lda_topics)
        run_ids.append(_log_run(config, f"lda_k{topic_count}", {
            "track": "topic", "method": "LDA", "requested_topics": topic_count,
            "actual_topics": topic_count, "top_words": config.topic_top_words,
            "seed": config.seed,
        }, {"npmi_coherence": coherence}))

        reducer = umap.UMAP(
            n_neighbors=config.umap_neighbors, n_components=config.umap_components,
            metric="cosine", random_state=config.seed,
        )
        clusterer = KMeans(n_clusters=topic_count, random_state=config.seed, n_init=10)
        topic_model = BERTopic(
            embedding_model=None, umap_model=reducer, hdbscan_model=clusterer,
            vectorizer_model=vectorizer, top_n_words=config.topic_top_words,
            calculate_probabilities=False, verbose=False,
        )
        topic_model.fit_transform(list(texts), embeddings)
        feature_names = vectorizer.get_feature_names_out()
        feature_lookup = {word: index for index, word in enumerate(feature_names)}
        bertopic_topics: list[list[int]] = []
        for topic_id in sorted(topic for topic in topic_model.get_topics() if topic >= 0):
            word_indices = [
                feature_lookup[word] for word, _weight in topic_model.get_topic(topic_id)
                if word in feature_lookup
            ][:config.topic_top_words]
            if len(word_indices) >= 2:
                bertopic_topics.append(word_indices)
        coherence = npmi_coherence(binary, bertopic_topics)
        run_ids.append(_log_run(config, f"bertopic_k{topic_count}", {
            "track": "topic", "method": "BERTopic", "requested_topics": topic_count,
            "actual_topics": len(bertopic_topics), "top_words": config.topic_top_words,
            "seed": config.seed,
        }, {"npmi_coherence": coherence}))
    return run_ids


def _cluster_quality(reduced: np.ndarray, labels: np.ndarray) -> tuple[float, float, int, float]:
    mask = labels >= 0
    used_labels = labels[mask]
    clusters = len(set(used_labels.tolist()))
    noise_fraction = float((~mask).mean())
    if clusters < 2 or mask.sum() <= clusters:
        return float("nan"), float("nan"), clusters, noise_fraction
    return (
        float(silhouette_score(reduced[mask], used_labels)),
        float(davies_bouldin_score(reduced[mask], used_labels)),
        clusters, noise_fraction,
    )


def mean_pairwise_ari(labelings: Sequence[np.ndarray]) -> float:
    """Mean ARI over every unique pair of seeded cluster assignments."""

    scores = [
        adjusted_rand_score(labelings[first], labelings[second])
        for first in range(len(labelings))
        for second in range(first + 1, len(labelings))
    ]
    if not scores:
        raise ValueError("At least two cluster labelings are required.")
    return float(np.mean(scores))


def run_clustering_sweeps(embeddings: np.ndarray, config: UnsupervisedConfig) -> list[str]:
    """Compare KMeans and HDBSCAN after UMAP, including ten-seed stability."""

    reduced_by_seed = [
        umap.UMAP(
            n_neighbors=config.umap_neighbors, n_components=config.umap_components,
            metric="cosine", random_state=config.seed + offset,
        ).fit_transform(embeddings)
        for offset in range(config.stability_seeds)
    ]
    base = reduced_by_seed[0]
    run_ids: list[str] = []
    for cluster_count in config.cluster_counts:
        labelings = [
            KMeans(n_clusters=cluster_count, random_state=config.seed + offset, n_init=10).fit_predict(reduced)
            for offset, reduced in enumerate(reduced_by_seed)
        ]
        silhouette, db, actual, noise = _cluster_quality(base, labelings[0])
        run_ids.append(_log_run(config, f"kmeans_k{cluster_count}", {
            "track": "clustering", "method": "KMeans", "requested_clusters": cluster_count,
            "stability_seeds": config.stability_seeds,
        }, {"silhouette": silhouette, "davies_bouldin": db,
            "stability_ari": mean_pairwise_ari(labelings), "actual_clusters": actual,
            "noise_fraction": noise}))
    for min_size in config.hdbscan_min_cluster_sizes:
        labelings = [
            hdbscan.HDBSCAN(min_cluster_size=min_size).fit_predict(reduced)
            for reduced in reduced_by_seed
        ]
        silhouette, db, actual, noise = _cluster_quality(base, labelings[0])
        run_ids.append(_log_run(config, f"hdbscan_min{min_size}", {
            "track": "clustering", "method": "HDBSCAN", "min_cluster_size": min_size,
            "stability_seeds": config.stability_seeds,
        }, {"silhouette": silhouette, "davies_bouldin": db,
            "stability_ari": mean_pairwise_ari(labelings), "actual_clusters": actual,
            "noise_fraction": noise}))
    return run_ids


def _shared_config(config: UnsupervisedConfig, seed: int) -> StanceTrainingConfig:
    return StanceTrainingConfig(
        name="semi_supervised", seed=seed, tracking_uri=config.tracking_uri,
        experiment_name=config.experiment_name, raw_dir=config.raw_dir,
        output_dir=config.embedding_cache.parent, comparison_report=config.report,
        model_name_or_path="distilbert/distilbert-base-uncased",
        max_length=config.max_length, train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate, weight_decay=config.weight_decay,
        max_epochs=config.semi_max_epochs, warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm, early_stopping_patience=config.semi_max_epochs,
        min_delta=0.0, mixed_precision=config.mixed_precision,
        imbalance_strategy="none", focal_gamma=2.0, target_conditioned=True,
    )


def _train_classifier(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    config: UnsupervisedConfig,
    seed: int,
) -> tuple[Any, Any, dict[str, Any]]:
    seed_everything(seed)
    model_id = "distilbert/distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=len(LABEL_ORDER),
        id2label={i: label for i, label in enumerate(LABEL_ORDER)},
        label2id={label: i for i, label in enumerate(LABEL_ORDER)},
        local_files_only=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    split = ValidationSplit(development, validation, report=None)  # type: ignore[arg-type]
    shared = _shared_config(config, seed)
    train_loader, validation_loader, _weights = _build_loaders(split, tokenizer, shared)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    steps = math.ceil(len(train_loader) / config.gradient_accumulation_steps) * config.semi_max_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * config.warmup_ratio), steps)
    for _epoch in range(config.semi_max_epochs):
        train_one_epoch(
            model, train_loader, optimizer, scheduler, device=device,
            strategy="none", class_weights=None, focal_gamma=2.0,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_grad_norm=config.max_grad_norm, mixed_precision=config.mixed_precision,
        )
    metrics = evaluate(
        model, validation_loader, device=device, strategy="none", class_weights=None,
        focal_gamma=2.0, mixed_precision=config.mixed_precision,
    )
    return model, tokenizer, metrics


def _predict_probabilities(
    model: Any, tokenizer: Any, frame: pd.DataFrame, config: UnsupervisedConfig
) -> np.ndarray:
    device = next(model.parameters()).device
    encodings = tokenizer(
        frame["text"].tolist(), text_pair=frame["target"].tolist(),
        padding="max_length", truncation=True, max_length=config.max_length,
        return_tensors="pt",
    )
    loader = DataLoader(StanceTextDataset(encodings), batch_size=config.eval_batch_size)
    probabilities: list[list[float]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(**{key: value.to(device) for key, value in batch.items()}).logits
            probabilities.extend(torch.softmax(logits, dim=-1).cpu().tolist())
    return np.asarray(probabilities)


def run_semi_supervised(split: ValidationSplit, config: UnsupervisedConfig) -> list[str]:
    """Run labeled-only and one-round self-training for each fraction and seed."""

    run_ids: list[str] = []
    development = split.development_train
    strata = development["target"] + " || " + development["label"]
    for fraction in config.label_fractions:
        for seed in config.semi_seeds:
            labeled_indices, unlabeled_indices = train_test_split(
                development.index.to_numpy(), train_size=fraction,
                random_state=seed, stratify=strata.to_numpy(),
            )
            labeled = development.loc[labeled_indices].copy()
            unlabeled = development.loc[unlabeled_indices].copy()
            teacher, tokenizer, baseline_metrics = _train_classifier(
                labeled, split.validation, config, seed
            )
            probabilities = _predict_probabilities(teacher, tokenizer, unlabeled, config)
            confidence = probabilities.max(axis=1)
            selected = confidence >= config.confidence_threshold
            pseudo = unlabeled.iloc[np.flatnonzero(selected)].copy()
            pseudo["label"] = [LABEL_ORDER[index] for index in probabilities[selected].argmax(axis=1)]
            combined = pd.concat([labeled, pseudo], axis=0)
            del teacher
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            student, _student_tokenizer, pseudo_metrics = _train_classifier(
                combined, split.validation, config, seed
            )
            del student
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            run_ids.append(_log_run(config, f"semi_f{fraction:.2f}_s{seed}", {
                "track": "semi_supervised", "label_fraction": fraction, "seed": seed,
                "confidence_threshold": config.confidence_threshold,
                "labeled_rows": len(labeled), "unlabeled_rows": len(unlabeled),
                "pseudo_labeled_rows": len(pseudo), "epochs": config.semi_max_epochs,
            }, {"baseline_macro_f1": baseline_metrics["macro_f1"],
                "pseudo_macro_f1": pseudo_metrics["macro_f1"],
                "pseudo_selection_rate": len(pseudo) / len(unlabeled)}))
    return run_ids


def generate_unsupervised_report(config: UnsupervisedConfig) -> Path:
    """Generate all tables exclusively from completed MLflow run data."""

    experiment = mlflow.get_experiment_by_name(config.experiment_name)
    if experiment is None:
        raise RuntimeError("Unsupervised MLflow experiment not found.")
    runs = mlflow.search_runs([experiment.experiment_id], filter_string="attributes.status = 'FINISHED'")
    topic = runs[runs["params.track"] == "topic"].copy()
    clustering = runs[runs["params.track"] == "clustering"].copy()
    semi = runs[
        (runs["params.track"] == "semi_supervised")
        & np.isclose(
            runs["params.confidence_threshold"].astype(float),
            config.confidence_threshold,
        )
    ].copy()
    expected_topic = len(config.topic_counts) * 2
    expected_cluster = len(config.cluster_counts) + len(config.hdbscan_min_cluster_sizes)
    expected_semi = len(config.label_fractions) * len(config.semi_seeds)
    if len(topic) < expected_topic or len(clustering) < expected_cluster or len(semi) < expected_semi:
        raise RuntimeError("MLflow does not contain every required completed run.")

    lines = [
        "# Unsupervised and Semi-Supervised Text Track", "",
        "Generated exclusively from MLflow runs over the official SemEval training split and its fixed derived validation partition. The official test split was not accessed.", "",
        "## Topic-model sweep", "",
        "Topic quality is compared quantitatively using NPMI coherence; topic words were not selected or judged by eye.", "",
        "| Method | Requested topics | Actual topics | NPMI coherence | MLflow run |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for _, row in topic.sort_values(["params.method", "params.requested_topics"]).iterrows():
        lines.append(f"| {row['params.method']} | {int(row['params.requested_topics'])} | {int(row['params.actual_topics'])} | {float(row['metrics.npmi_coherence']):.6f} | `{row['run_id']}` |")
    lines += ["", "## Embedding clustering", "", "Stability is mean pairwise adjusted Rand index across the configured ten seeded UMAP-and-clustering runs. Silhouette and Davies–Bouldin exclude HDBSCAN noise points.", "", "| Method | Configuration | Clusters | Noise fraction | Silhouette | Davies-Bouldin | Stability ARI | MLflow run |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for _, row in clustering.sort_values(["params.method", "tags.mlflow.runName"]).iterrows():
        setting = f"k={row['params.requested_clusters']}" if row["params.method"] == "KMeans" else f"min_cluster_size={row['params.min_cluster_size']}"
        lines.append(f"| {row['params.method']} | {setting} | {int(float(row['metrics.actual_clusters']))} | {float(row['metrics.noise_fraction']):.6f} | {float(row['metrics.silhouette']):.6f} | {float(row['metrics.davies_bouldin']):.6f} | {float(row['metrics.stability_ari']):.6f} | `{row['run_id']}` |")
    lines += ["", "## Label-efficiency and one-round self-training", "", f"Pseudo-labels use a confidence threshold of {config.confidence_threshold:.2f}. Each summary is calculated from three seeded MLflow runs.", "", "| Label fraction | Labeled-only macro-F1 | With pseudo-labels macro-F1 | Mean pseudo-label selection rate | Source runs |", "| ---: | ---: | ---: | ---: | --- |"]
    for fraction in config.label_fractions:
        subset = semi[np.isclose(semi["params.label_fraction"].astype(float), fraction)]
        baseline = subset["metrics.baseline_macro_f1"].astype(float).to_numpy()
        pseudo = subset["metrics.pseudo_macro_f1"].astype(float).to_numpy()
        selection = subset["metrics.pseudo_selection_rate"].astype(float).to_numpy()
        run_ids = ", ".join(f"`{run_id}`" for run_id in subset["run_id"])
        lines.append(f"| {fraction:.0%} | {baseline.mean():.6f} ± {baseline.std(ddof=1):.6f} | {pseudo.mean():.6f} ± {pseudo.std(ddof=1):.6f} | {selection.mean():.6f} | {run_ids} |")
    lines += ["", "All reported values are direct MLflow metrics or deterministic mean/standard-deviation summaries of the linked MLflow runs.", ""]
    config.report.parent.mkdir(parents=True, exist_ok=True)
    config.report.write_text("\n".join(lines), encoding="utf-8")
    return config.report


def run_unsupervised_track(config: UnsupervisedConfig) -> Path:
    """Execute all requested tracks and generate the MLflow-backed report."""

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    data, split = _training_rows(config)
    embeddings = compute_or_load_embeddings(data["text"].tolist(), config)
    run_topic_sweeps(data["text"].tolist(), embeddings, config)
    run_clustering_sweeps(embeddings, config)
    run_semi_supervised(split, config)
    return generate_unsupervised_report(config)


def rerun_semi_supervised_track(config: UnsupervisedConfig) -> Path:
    """Run only semi-supervised settings and regenerate the complete report."""

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    _data, split = _training_rows(config)
    run_semi_supervised(split, config)
    return generate_unsupervised_report(config)
