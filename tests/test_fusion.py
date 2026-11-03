from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from constituency_insights.fusion import (
    EarlyFusionClassifier, FusionDataError, fusion_lift_statistics,
    generate_synthetic_fusion_benchmark, load_linked_text_features,
    temporal_fusion_ablation,
)
from constituency_insights.tabular import ExpandingWindowSplitter


EARLY = {
    "tabular_hidden": 6, "text_hidden": 4, "joint_hidden": 6,
    "epochs": 8, "learning_rate": .01, "batch_size": 32, "patience": 3,
}


def test_linked_loader_requires_exact_aggregate_only_keys(tmp_path: Path) -> None:
    valid = pd.DataFrame({"constituency_id": [1, 2], "election_year": [2014, 2019],
                          "collection_end_year": [2014, 2018],
                          "sentiment_mean": [.2, -.1], "embedding_0": [1., 2.]})
    path = tmp_path / "linked.csv"
    valid.to_csv(path, index=False)
    pdt.assert_frame_equal(load_linked_text_features(path), valid)
    duplicated = pd.concat([valid, valid.iloc[[0]]])
    duplicated.to_csv(path, index=False)
    with pytest.raises(FusionDataError, match="unique"):
        load_linked_text_features(path)


def test_linked_loader_rejects_raw_text_and_post_election_collection(tmp_path: Path) -> None:
    path = tmp_path / "linked.csv"
    pd.DataFrame({"constituency_id": [1], "election_year": [2019],
                  "collection_end_year": [2020], "sentiment_mean": [.1]}).to_csv(path, index=False)
    with pytest.raises(FusionDataError, match="after an election"):
        load_linked_text_features(path)
    pd.DataFrame({"constituency_id": [1], "election_year": [2019],
                  "collection_end_year": [2019], "sentiment_mean": [.1],
                  "text": ["forbidden"]}).to_csv(path, index=False)
    with pytest.raises(FusionDataError, match="Raw text"):
        load_linked_text_features(path)


def test_synthetic_benchmark_is_seeded_and_explicitly_labeled() -> None:
    first, tabular_a, text_a = generate_synthetic_fusion_benchmark(9, 10, [2004, 2009], 1.2, .5)
    second, tabular_b, text_b = generate_synthetic_fusion_benchmark(9, 10, [2004, 2009], 1.2, .5)
    pdt.assert_frame_equal(first, second)
    assert tabular_a == tabular_b and text_a == text_b
    assert first.synthetic_benchmark.all()
    assert not any("text" in column for column in first.columns)


def test_early_fusion_is_deterministic_given_seed() -> None:
    frame, tabular, text = generate_synthetic_fusion_benchmark(
        11, 20, [2004, 2009], 1.5, .5
    )
    first = EarlyFusionClassifier(tabular, text, seed=17, **EARLY).fit(frame, frame.target)
    second = EarlyFusionClassifier(tabular, text, seed=17, **EARLY).fit(frame, frame.target)
    assert np.array_equal(first.predict(frame), second.predict(frame))


def test_temporal_ablation_and_lift_use_matching_oof_rows() -> None:
    frame, tabular, text = generate_synthetic_fusion_benchmark(
        21, 60, [2004, 2009, 2014, 2019, 2024], 2.0, .25
    )
    predictions, scores = temporal_fusion_ablation(
        frame, tabular, text, ExpandingWindowSplitter(2), 21, EARLY
    )
    assert set(scores.year) == {2014, 2019, 2024}
    counts = predictions.groupby("model").size()
    assert counts.nunique() == 1
    late = fusion_lift_statistics(predictions, scores, "late_fusion", 21, 100)
    assert late["folds"] == 3
    assert late["ci_low"] <= late["mean_lift"] <= late["ci_high"]
    assert late["mean_lift"] > 0
