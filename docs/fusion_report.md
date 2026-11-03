# Text–tabular fusion report

**SYNTHETIC: pipeline validation only, not evidence of real-world gains.**

No valid real text-to-constituency-year link exists. The benchmark injects a known aggregate text signal solely to test the fusion and ablation machinery. It does not show that text improves Indian election prediction.

## Expanding-window results

| Model | Mean accuracy | Standard deviation |
|---|---:|---:|
| early_fusion | 0.6889 | 0.0375 |
| late_fusion | 0.7194 | 0.0239 |
| tabular_only | 0.5500 | 0.0312 |

## Lift over tabular only

| Model | Mean lift | Bootstrap 95% CI | Paired Wilcoxon p |
|---|---:|---:|---:|
| late_fusion | 0.1694 | [0.1056, 0.2306] | 0.2500 |
| early_fusion | 0.1389 | [0.0806, 0.2028] | 0.2500 |

Bootstrap intervals resample constituencies. The paired test uses election-year folds; with few folds it has low power and should be treated as descriptive.

## What this does not show

These results do not establish availability, representativeness, geographic validity, or predictive value of any real text source. No SemEval record was linked to an Indian constituency. A real claim requires an audited user-supplied constituency-year aggregate file and a new time-based evaluation.
