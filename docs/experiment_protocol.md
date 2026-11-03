# Stance Classification Experiment Protocol

This protocol defines a fair validation comparison between the existing
TF-IDF + Logistic Regression baseline and future models, including a
multilingual transformer. It applies to SemEval-2016 Task 6, Task A.

## Fixed task and data

- The prediction input is the normalized `text` field, sourced from the
  official `Tweet` column. The existing baseline does not concatenate or
  otherwise provide `target` to the model. For a direct comparison, future
  models must also use text only. A target-aware model must be reported as a
  separate experimental track, not as a like-for-like replacement.
- `target` is retained as evaluation metadata. The five exact values are
  `Atheism`, `Climate Change is a Real Concern`, `Feminist Movement`,
  `Hillary Clinton`, and `Legalization of Abortion`.
- The three labels, and the fixed reporting order, are `AGAINST`, `FAVOR`, and
  `NONE`. `NONE` means that neither favor nor opposition can be inferred; it
  is not another form of `AGAINST`.
- Only the 2,914 official Task A training rows are eligible for model
  development. The fixed in-memory split contains 2,331 development-training
  rows and 583 validation rows.
- The split uses random seed 42 and an 80/20 allocation stratified on the
  combined `target` and `label`. Every observed combination is represented in
  both partitions. The smallest validation cell is `Climate Change is a Real
  Concern` / `AGAINST`, with support 3.
- All models must use the exact split produced by
  `prepare_task_a_validation_split` with its established settings. Models may
  not create a different random split for their primary comparison.
- The official test set remains untouched until model selection and all
  development decisions are complete. It must not be used for fitting,
  vocabulary or tokenizer decisions, hyperparameter selection, early stopping,
  threshold selection, or validation reporting.

## Leakage controls

All learned preprocessing must be fitted using development-training data only.
This includes TF-IDF vocabularies and statistics, feature selection, tokenizer
training or adaptation, normalization learned from the corpus, calibration,
and any resampling operation. Validation labels or metrics must not be used to
fit preprocessing.

Hyperparameters and model variants must not be selected using the validation
partition without recording that selection. Repeatedly choosing configurations
from validation performance turns that partition into tuning data and makes its
score optimistic. If tuning is introduced, its search space and selection
procedure must be specified in advance and operate within development-training
data, for example through fixed cross-validation. The fixed validation
partition should then be used once for the model comparison.

## Required metrics and reporting

The primary metric is three-class macro-F1 over `AGAINST`, `FAVOR`, and `NONE`.
Every experiment must also report:

- accuracy;
- precision, recall, F1, and support for each class;
- a confusion matrix with the explicit order `AGAINST`, `FAVOR`, `NONE`;
- macro-F1, accuracy, per-class metrics, and confusion matrix separately for
  each target;
- a warning beside any per-target class support below 10;
- the majority-class baseline for context;
- development-training and validation row counts;
- random seeds and the relevant library/runtime versions;
- model identifier or checkpoint revision, preprocessing configuration,
  training settings, and hardware where applicable; and
- whether results are from a single deterministic run or a distribution over
  multiple seeds.

Aggregate metrics and confusion-matrix data may be stored in the ignored local
experiment-output directory. Raw text and error excerpts must remain local and
untracked. Observations directly supported by metrics must be distinguished
from hypotheses about why a model behaves as it does.

## Current baseline configuration

The existing baseline is a scikit-learn `Pipeline`, ensuring TF-IDF is fitted
only on development-training text:

- `TfidfVectorizer`: text only, word analyzer, `lowercase=True`, word unigrams
  and bigrams (`ngram_range=(1, 2)`). Other vectorizer behavior uses the
  defaults of the installed scikit-learn version; it is not independently
  specified in the repository.
- `LogisticRegression`: `class_weight="balanced"`, `max_iter=1000`, and
  `random_state=42`. These are fixed initial settings, not tuned optima.
- No target feature, target-text concatenation, external corpus, or official
  test example is used.
- The two losslessly retained encoding-anomalous training records, source row
  indices 109 and 1827, are in the development-training partition. Their
  intended characters remain unresolved; no corrected text is invented.

## Preliminary validation result

These are measured validation results from the current baseline, not final test
results:

| Metric | Value |
| --- | ---: |
| Three-class macro-F1 | 0.5516610154291314 |
| Accuracy | 0.5883361921097771 |
| Majority-class accuracy (`AGAINST`) | 0.48027444253859347 |
| Majority-class macro-F1 | 0.21629972962533797 |

Per-class validation results:

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| `AGAINST` | 0.6528662420382165 | 0.7321428571428571 | 0.6902356902356902 | 280 |
| `FAVOR` | 0.4931506849315068 | 0.48 | 0.4864864864864865 | 150 |
| `NONE` | 0.5365853658536586 | 0.43137254901960786 | 0.4782608695652174 | 153 |

The validation confusion matrix, with rows as true labels and columns as
predictions in the order `AGAINST`, `FAVOR`, `NONE`, is:

```text
[[205, 45, 30],
 [ 51, 72, 27],
 [ 58, 29, 66]]
```

The per-target validation macro-F1 values are 0.4789613201911589 for Atheism,
0.4250298685782557 for Climate Change is a Real Concern, 0.5200162544990131
for Feminist Movement, 0.5665570774958756 for Hillary Clinton, and
0.5239858906525573 for Legalization of Abortion. The Climate Change / AGAINST
support is only 3, so its per-class and aggregate target metrics are unstable.

## Observed error patterns

The deterministic validation error report contains 240 errors. Counts by true
and predicted label are:

| True label → prediction | Count |
| --- | ---: |
| `NONE` → `AGAINST` | 58 |
| `FAVOR` → `AGAINST` | 51 |
| `AGAINST` → `FAVOR` | 45 |
| `AGAINST` → `NONE` | 30 |
| `NONE` → `FAVOR` | 29 |
| `FAVOR` → `NONE` | 27 |

The largest observed error count is `NONE` predicted as `AGAINST`. Feminist
Movement has the most errors in absolute count (59), followed by Legalization
of Abortion (52), Hillary Clinton (50), Atheism (41), and Climate Change is a
Real Concern (38). These are observations, not explanations: raw counts are
affected by target sample sizes and label distributions. Whether lexical cues,
implicit stance, target omission from model input, temporal language, or other
factors cause these errors remains a hypothesis requiring controlled analysis.
No tweet text is reproduced here.

## Unresolved items for a transformer experiment

- The transformer checkpoint and immutable revision have not been selected.
- Transformer tokenization, maximum sequence length, optimization settings,
  stopping rule, compute environment, and seed policy are not yet specified.
- A development-training-only tuning procedure has not been defined.
- The policy for comparing one transformer run versus multiple seeded runs is
  unresolved.

These decisions must be recorded before training so that validation results are
interpretable and reproducible.
