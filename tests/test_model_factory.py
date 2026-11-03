from constituency_insights.nlp.model_factory import (
    ModelFactoryConfig,
    parameter_counts,
    resolve_backbone,
)

import torch


def test_supported_backbone_names_resolve_to_model_ids() -> None:
    assert resolve_backbone("distilbert-base-uncased") == "distilbert/distilbert-base-uncased"
    assert resolve_backbone("xlm-roberta-base") == "FacebookAI/xlm-roberta-base"
    assert resolve_backbone("muril-base-cased") == "google/muril-base-cased"
    assert resolve_backbone("indicbert") == "ai4bharat/indic-bert"


def test_parameter_counts_distinguish_frozen_parameters() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    for parameter in model[0].parameters():
        parameter.requires_grad = False

    trainable, total = parameter_counts(model)

    assert trainable == 4
    assert total == 13
