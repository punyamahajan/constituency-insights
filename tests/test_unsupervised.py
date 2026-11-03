from dataclasses import replace
from pathlib import Path

import numpy as np

from constituency_insights.nlp.unsupervised import (
    compute_or_load_embeddings,
    load_unsupervised_config,
    mean_pairwise_ari,
    npmi_coherence,
)


def test_npmi_coherence_matches_perfect_cooccurrence() -> None:
    binary = np.array([
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
    ])

    assert npmi_coherence(binary, [[0, 1]]) == 1.0


def test_mean_pairwise_ari_is_one_for_stable_clusterings() -> None:
    labels = np.array([0, 0, 1, 1, 2, 2])
    relabeled = np.array([2, 2, 0, 0, 1, 1])

    assert mean_pairwise_ari([labels, relabeled, labels.copy()]) == 1.0


def test_single_yaml_has_required_unsupervised_sweeps() -> None:
    config = load_unsupervised_config("configs/stance_training.yaml")

    assert config.topic_counts == (5, 10, 15, 20)
    assert config.stability_seeds == 10
    assert config.label_fractions == (0.10, 0.25, 0.50)
    assert config.semi_seeds == (42, 43, 44)


def test_embedding_cache_avoids_recomputation(tmp_path: Path, monkeypatch) -> None:
    config = replace(
        load_unsupervised_config("configs/stance_training.yaml"),
        embedding_cache=tmp_path / "embeddings.npy",
    )

    class FakeSentenceTransformer:
        calls = 0

        def __init__(self, *_args, **_kwargs):
            pass

        def encode(self, texts, **_kwargs):
            type(self).calls += 1
            return np.arange(len(texts) * 2, dtype=float).reshape(len(texts), 2)

    monkeypatch.setattr(
        "constituency_insights.nlp.unsupervised.SentenceTransformer",
        FakeSentenceTransformer,
    )
    first = compute_or_load_embeddings(["alpha", "beta"], config)
    second = compute_or_load_embeddings(["alpha", "beta"], config)

    np.testing.assert_array_equal(first, second)
    assert FakeSentenceTransformer.calls == 1
