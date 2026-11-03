from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml


SPEC = importlib.util.spec_from_file_location("final_test_eval", "scripts/final_test_eval.py")
assert SPEC and SPEC.loader
final_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(final_eval)


def _frozen_config(tmp_path: Path) -> tuple[Path, str, Path]:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"frozen weights")
    digest = final_eval.sha256_file(weights)
    selection = tmp_path / "selection.md"
    selection.write_text("selected using validation only", encoding="utf-8")
    lock = tmp_path / "lock.json"
    config = {
        "frozen": True, "selection_document": str(selection), "seed": 42,
        "raw_dir": str(tmp_path / "must-not-be-read"),
        "checkpoint": str(checkpoint), "weights_file": weights.name,
        "expected_checkpoint_sha256": digest, "model_name": "frozen",
        "target_conditioned": True, "max_length": 8, "batch_size": 2,
        "bootstrap_resamples": 10, "confidence_level": .95,
        "output_dir": str(tmp_path / "output"), "report": str(tmp_path / "report.md"),
        "lock_file": str(lock),
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path, digest, lock


def test_preflight_requires_matching_provided_and_actual_hash(tmp_path: Path) -> None:
    config, digest, _ = _frozen_config(tmp_path)
    loaded, warning = final_eval.validate_preflight(config, digest, False)
    assert loaded["frozen"] is True and warning is None
    with pytest.raises(ValueError, match="does not match the frozen config"):
        final_eval.validate_preflight(config, "0" * 64, False)
    (tmp_path / "checkpoint" / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Actual checkpoint"):
        final_eval.validate_preflight(config, digest, False)


def test_lock_refuses_second_run_unless_force_and_warns(tmp_path: Path) -> None:
    config, digest, lock = _frozen_config(tmp_path)
    lock.write_text("already evaluated", encoding="utf-8")
    with pytest.raises(RuntimeError, match="requires --force"):
        final_eval.validate_preflight(config, digest, False)
    _, warning = final_eval.validate_preflight(config, digest, True)
    assert warning and "WARNING" in warning and "tuning" in warning


def test_official_metric_is_average_of_favor_and_against_f1() -> None:
    truth = ["FAVOR", "FAVOR", "AGAINST", "AGAINST", "NONE"]
    predicted = ["FAVOR", "AGAINST", "AGAINST", "AGAINST", "NONE"]
    metrics = final_eval.classification_metrics(truth, predicted)
    expected = (metrics["per_class"]["FAVOR"]["f1"]
                + metrics["per_class"]["AGAINST"]["f1"]) / 2
    assert metrics["official_f1_favor_against"] == pytest.approx(expected)


def test_bootstrap_intervals_are_seeded() -> None:
    truth = ["FAVOR", "AGAINST", "NONE"] * 5
    first = final_eval.bootstrap_confidence_intervals(truth, truth, 7, 25, .95)
    second = final_eval.bootstrap_confidence_intervals(truth, truth, 7, 25, .95)
    assert first == second
    assert first["accuracy"] == [1.0, 1.0]
