"""Natural-language processing features and models."""
"""Natural-language processing baselines."""

from .baseline import (
    LABEL_ORDER,
    BaselineRun,
    build_baseline_pipeline,
    fit_and_evaluate_baseline,
    run_official_validation_baseline,
    write_local_outputs,
)
from .distilbert import (
    DistilBertExperimentConfig,
    DistilBertRun,
    run_official_distilbert_experiment,
    train_and_evaluate_distilbert,
    write_distilbert_outputs,
)
from .comparison import (
    McNemarResult,
    compare_validation_predictions,
    compute_mcnemar_test,
    generate_baseline_validation_predictions,
    generate_distilbert_validation_predictions,
    run_validation_comparison,
    save_comparison_report,
    save_validation_predictions,
)
from .error_analysis import build_validation_error_report, privacy_reduced_excerpt
from .predict import StancePrediction, predict_stance, validate_checkpoint_path
from .training import (
    StanceTrainingConfig,
    TrainingResult,
    focal_loss,
    generate_comparison_from_mlflow,
    load_training_configs,
    train_stance_model,
)

__all__ = [
    "LABEL_ORDER",
    "BaselineRun",
    "build_baseline_pipeline",
    "fit_and_evaluate_baseline",
    "run_official_validation_baseline",
    "write_local_outputs",
    "DistilBertExperimentConfig",
    "DistilBertRun",
    "train_and_evaluate_distilbert",
    "run_official_distilbert_experiment",
    "write_distilbert_outputs",
    "build_validation_error_report",
    "privacy_reduced_excerpt",
    "StancePrediction",
    "predict_stance",
    "validate_checkpoint_path",
    "McNemarResult",
    "compute_mcnemar_test",
    "generate_baseline_validation_predictions",
    "generate_distilbert_validation_predictions",
    "compare_validation_predictions",
    "save_validation_predictions",
    "save_comparison_report",
    "run_validation_comparison",
    "StanceTrainingConfig",
    "TrainingResult",
    "focal_loss",
    "generate_comparison_from_mlflow",
    "load_training_configs",
    "train_stance_model",
]
