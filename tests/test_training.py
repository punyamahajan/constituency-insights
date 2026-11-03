"""Unit tests for the explicit stance training loop."""

from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import DataLoader

from constituency_insights.nlp.distilbert import StanceTextDataset
from constituency_insights.nlp.training import (
    focal_loss,
    load_training_configs,
    seed_everything,
    train_one_epoch,
)


class TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.classifier = torch.nn.Linear(4, 3)

    def forward(self, input_ids: torch.Tensor):
        pooled = self.embedding(input_ids).mean(dim=1)
        return type("Output", (), {"logits": self.classifier(pooled)})()


def test_focal_loss_matches_hand_computed_value() -> None:
    logits = torch.tensor([[math.log(0.8), math.log(0.2)]], dtype=torch.float64)
    targets = torch.tensor([0])

    actual = focal_loss(logits, targets, gamma=2.0)
    expected = -((1.0 - 0.8) ** 2) * math.log(0.8)

    assert actual.item() == pytest.approx(expected, abs=1e-12)


def _run_two_steps() -> dict[str, torch.Tensor]:
    seed_everything(1234)
    model = TinyClassifier()
    dataset = StanceTextDataset(
        {"input_ids": torch.tensor([[1, 2], [2, 3], [3, 4], [4, 5]])},
        [0, 1, 2, 0],
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.5, total_iters=2
    )
    train_one_epoch(
        model,
        loader,
        optimizer,
        scheduler,
        device=torch.device("cpu"),
        strategy="none",
        class_weights=None,
        focal_gamma=2.0,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        mixed_precision=True,
    )
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def test_two_tiny_training_steps_are_deterministic() -> None:
    first = _run_two_steps()
    second = _run_two_steps()

    assert first.keys() == second.keys()
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)


def test_single_yaml_defines_all_three_imbalance_runs() -> None:
    configs = load_training_configs("configs/stance_training.yaml")

    assert [config.imbalance_strategy for config in configs] == [
        "none", "weighted", "focal"
    ]
    assert {config.seed for config in configs} == {42}
