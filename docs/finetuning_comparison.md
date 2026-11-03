# Full Fine-Tuning and LoRA Comparison

Generated from MLflow metrics on the fixed validation partition derived exclusively from the official SemEval training split.

SemEval-2016 Task 6 is English-only. These experiments do not provide evidence of cross-lingual transfer to Hindi, Tamil, code-mixed text, or Indian constituency discourse.

| Model | Method | Accuracy | Macro-F1 | Trainable parameters | Peak GPU MB | Peak CPU MB | Seconds/epoch | Checkpoint MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| distilbert-base-uncased | full | 0.675815 | 0.658690 | 66,955,779 | 1349.52 | 2209.89 | 19.05 | 256.11 |
| xlm-roberta-base | full | 0.670669 | 0.642980 | 278,045,955 | 5339.29 | 3419.39 | 47.17 | 1076.99 |
| xlm-roberta-base | lora | 0.526587 | 0.365471 | 1,035,267 | 1605.84 | 3423.27 | 20.27 | 20.27 |

## Paired McNemar comparison

Best new model: `xlm_roberta_full`. Compared with the existing target-conditioned DistilBERT predictions on the identical validation rows.

- Baseline-only correct: 47
- New-model-only correct: 50
- Continuity-corrected statistic: 0.041237
- Continuity-corrected p-value: 0.839081

## Analysis

**TODO (author): Add interpretation of performance, efficiency, and practical trade-offs here.**
