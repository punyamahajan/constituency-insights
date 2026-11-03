# Final SemEval model selection

Selection was completed using only the fixed validation partition derived from
the official SemEval training split. The official test split was not read,
scored, inspected, or used to choose this model.

## Frozen choice

- Model: target-conditioned `distilbert-base-uncased`
- Training method: full fine-tuning
- Class-imbalance loss: focal loss
- Validation accuracy: `0.675815`
- Validation macro-F1: `0.658690`
- Local checkpoint: `outputs/local/stance_training/finetuning/distilbert_full/best_checkpoint`
- Frozen weights SHA-256: `c59ffaa1776d9af28aa64332877fd8971a1a2b760ffb207e03d828e3118170df`

## Validation-only justification

This checkpoint had the highest recorded validation macro-F1 among the
completed, directly comparable supervised runs:

| Candidate | Validation macro-F1 |
|---|---:|
| DistilBERT full, focal loss | 0.658690 |
| XLM-R full, focal loss | 0.642980 |
| DistilBERT weighted cross-entropy | 0.646164 |
| DistilBERT unweighted cross-entropy | 0.633514 |
| XLM-R LoRA | 0.365471 |

The fixed TF-IDF validation baseline was also lower (`0.5517` macro-F1). The
selection criterion was validation macro-F1 because the three stance classes
are imbalanced and all classes matter. XLM-R full did not exceed DistilBERT and
had substantially higher compute and checkpoint cost. LoRA was not competitive
on validation.

No additional tuning, threshold selection, calibration fitting, checkpoint
comparison, or error-driven changes are permitted after final test evaluation.
The final evaluator must verify the frozen weights hash before loading the test
split and must create a one-shot lock.
