from constituency_insights.nlp.finetuning import load_finetuning_configs


def test_finetuning_yaml_defines_required_comparison_runs() -> None:
    configs = load_finetuning_configs("configs/stance_training.yaml")

    assert [(item.backbone, item.tuning_method) for item in configs] == [
        ("distilbert-base-uncased", "full"),
        ("xlm-roberta-base", "full"),
        ("xlm-roberta-base", "lora"),
    ]
    assert {item.seed for item in configs} == {42}
