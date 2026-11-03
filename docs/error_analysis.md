# Validation Error and Calibration Analysis

Generated from the fixed validation partition derived only from the official SemEval training split. The official test split was not accessed.

All confidence intervals below are seeded 95% percentile bootstrap intervals from 1,000 resamples.

## Per-class metrics

| Class | Precision [CI] | Recall [CI] | F1 [CI] |
| --- | ---: | ---: | ---: |
| AGAINST | 0.7992 [0.7480, 0.8459] | 0.7107 [0.6581, 0.7634] | 0.7524 [0.7070, 0.7914] |
| FAVOR | 0.5526 [0.4829, 0.6237] | 0.7000 [0.6258, 0.7703] | 0.6176 [0.5562, 0.6769] |
| NONE | 0.6250 [0.5419, 0.7072] | 0.5882 [0.5066, 0.6644] | 0.6061 [0.5357, 0.6644] |

## Per-target metrics

| Target | Class | Precision [CI] | Recall [CI] | F1 [CI] |
| --- | --- | ---: | ---: | ---: |
| Atheism | AGAINST | 0.8594 [0.7761, 0.9394] | 0.9016 [0.8135, 0.9677] | 0.8800 [0.8166, 0.9333] |
| Atheism | FAVOR | 0.4348 [0.2381, 0.6402] | 0.5556 [0.3333, 0.7895] | 0.4878 [0.2927, 0.6667] |
| Atheism | NONE | 0.8000 [0.5788, 1.0000] | 0.5217 [0.3333, 0.7308] | 0.6316 [0.4375, 0.7895] |
| Climate Change is a Real Concern | AGAINST | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| Climate Change is a Real Concern | FAVOR | 0.6667 [0.5468, 0.7762] | 0.9524 [0.8810, 1.0000] | 0.7843 [0.6896, 0.8598] |
| Climate Change is a Real Concern | NONE | 0.8947 [0.7368, 1.0000] | 0.5000 [0.3333, 0.6562] | 0.6415 [0.4761, 0.7693] |
| Feminist Movement | AGAINST | 0.8125 [0.7045, 0.9184] | 0.5909 [0.4614, 0.7097] | 0.6842 [0.5714, 0.7769] |
| Feminist Movement | FAVOR | 0.5000 [0.3750, 0.6271] | 0.6905 [0.5500, 0.8276] | 0.5800 [0.4615, 0.6909] |
| Feminist Movement | NONE | 0.6296 [0.4374, 0.8148] | 0.6800 [0.4762, 0.8519] | 0.6538 [0.4878, 0.7931] |
| Hillary Clinton | AGAINST | 0.7778 [0.6761, 0.8750] | 0.7089 [0.6024, 0.8001] | 0.7417 [0.6573, 0.8125] |
| Hillary Clinton | FAVOR | 0.6875 [0.4443, 0.9091] | 0.4583 [0.2609, 0.6429] | 0.5500 [0.3529, 0.7112] |
| Hillary Clinton | NONE | 0.5490 [0.4151, 0.6923] | 0.7778 [0.6363, 0.9092] | 0.6437 [0.5168, 0.7561] |
| Legalization of Abortion | AGAINST | 0.7538 [0.6470, 0.8525] | 0.6901 [0.5750, 0.8000] | 0.7206 [0.6281, 0.7974] |
| Legalization of Abortion | FAVOR | 0.4545 [0.2759, 0.6334] | 0.6250 [0.4209, 0.8077] | 0.5263 [0.3529, 0.6667] |
| Legalization of Abortion | NONE | 0.5000 [0.3333, 0.6787] | 0.4571 [0.3125, 0.6216] | 0.4776 [0.3385, 0.6230] |

## Highest-confidence errors

Raw text for these cases is stored only in the Git-ignored local file `outputs/local/error_cases.csv`.

| Index | Target | True label | Predicted label | Confidence |
| ---: | --- | --- | --- | ---: |
| 781 | Climate Change is a Real Concern | FAVOR | NONE | 0.792003 |
| 685 | Climate Change is a Real Concern | NONE | FAVOR | 0.780541 |
| 687 | Climate Change is a Real Concern | AGAINST | FAVOR | 0.752877 |
| 730 | Climate Change is a Real Concern | NONE | FAVOR | 0.748587 |
| 783 | Climate Change is a Real Concern | NONE | FAVOR | 0.747786 |
| 905 | Climate Change is a Real Concern | AGAINST | FAVOR | 0.736479 |
| 2194 | Hillary Clinton | AGAINST | NONE | 0.723250 |
| 2584 | Legalization of Abortion | NONE | AGAINST | 0.721470 |
| 2850 | Legalization of Abortion | AGAINST | NONE | 0.714121 |
| 622 | Climate Change is a Real Concern | AGAINST | FAVOR | 0.708421 |
| 991 | Climate Change is a Real Concern | NONE | FAVOR | 0.698834 |
| 2765 | Legalization of Abortion | AGAINST | NONE | 0.683834 |
| 1497 | Feminist Movement | AGAINST | FAVOR | 0.677445 |
| 787 | Climate Change is a Real Concern | NONE | FAVOR | 0.676793 |
| 2489 | Legalization of Abortion | NONE | AGAINST | 0.668840 |
| 2792 | Legalization of Abortion | AGAINST | NONE | 0.662420 |
| 2308 | Hillary Clinton | AGAINST | NONE | 0.659472 |
| 964 | Climate Change is a Real Concern | NONE | FAVOR | 0.658777 |
| 482 | Atheism | NONE | FAVOR | 0.656674 |
| 1664 | Feminist Movement | AGAINST | FAVOR | 0.654070 |
| 218 | Atheism | FAVOR | AGAINST | 0.653543 |
| 2109 | Hillary Clinton | AGAINST | NONE | 0.653357 |
| 921 | Climate Change is a Real Concern | NONE | FAVOR | 0.649442 |
| 701 | Climate Change is a Real Concern | NONE | FAVOR | 0.644355 |
| 121 | Atheism | FAVOR | AGAINST | 0.643565 |
| 1854 | Hillary Clinton | AGAINST | NONE | 0.642239 |
| 2274 | Hillary Clinton | AGAINST | NONE | 0.639590 |
| 2457 | Legalization of Abortion | NONE | AGAINST | 0.637577 |
| 996 | Climate Change is a Real Concern | NONE | FAVOR | 0.636682 |
| 1141 | Feminist Movement | AGAINST | FAVOR | 0.635174 |
| 2676 | Legalization of Abortion | FAVOR | NONE | 0.629953 |
| 2426 | Legalization of Abortion | FAVOR | AGAINST | 0.629315 |
| 2415 | Legalization of Abortion | AGAINST | FAVOR | 0.626187 |
| 2004 | Hillary Clinton | FAVOR | NONE | 0.625719 |
| 2230 | Hillary Clinton | FAVOR | NONE | 0.618233 |
| 2898 | Legalization of Abortion | AGAINST | NONE | 0.616686 |
| 2849 | Legalization of Abortion | AGAINST | NONE | 0.615946 |
| 1586 | Feminist Movement | AGAINST | NONE | 0.615329 |
| 792 | Climate Change is a Real Concern | NONE | FAVOR | 0.612573 |
| 1600 | Feminist Movement | AGAINST | FAVOR | 0.611371 |
| 245 | Atheism | FAVOR | AGAINST | 0.611202 |
| 1763 | Hillary Clinton | NONE | AGAINST | 0.610171 |
| 1486 | Feminist Movement | AGAINST | NONE | 0.607027 |
| 2168 | Hillary Clinton | AGAINST | NONE | 0.605101 |
| 1213 | Feminist Movement | FAVOR | AGAINST | 0.603550 |
| 829 | Climate Change is a Real Concern | NONE | FAVOR | 0.603154 |
| 2653 | Legalization of Abortion | NONE | FAVOR | 0.598065 |
| 2694 | Legalization of Abortion | AGAINST | NONE | 0.596701 |
| 2502 | Legalization of Abortion | FAVOR | NONE | 0.596191 |
| 2913 | Legalization of Abortion | AGAINST | NONE | 0.595998 |

## Heuristic tag error rates

Tags are overlapping surface-form indicators; rates are conditional on a row having the tag.

| Tag | Tagged rows | Errors | Error rate |
| --- | ---: | ---: | ---: |
| contains_negation | 168 | 49 | 0.2917 |
| contains_hashtag | 571 | 186 | 0.3257 |
| very_short | 15 | 5 | 0.3333 |
| contains_question_mark | 75 | 29 | 0.3867 |
| contains_mention | 242 | 76 | 0.3140 |

## Calibration

- Uncalibrated ECE: 0.095345
- Uncalibrated multiclass Brier score: 0.462155
- Same-split fitted temperature: 0.716748
- Same-split temperature-scaled ECE: 0.041851
- Same-split temperature-scaled Brier score: 0.450981

> **Warning:** The same-split temperature is fitted and evaluated on this validation set, so its calibration estimates are optimistically biased and must not be treated as held-out evidence.

The preferred estimate uses 5-fold cross-fitting: each row is calibrated by a temperature fitted without that row's fold.

- Cross-fitted temperatures: 0.660561, 0.752750, 0.708282, 0.717139, 0.744558
- Cross-fitted ECE: 0.044800
- Cross-fitted multiclass Brier score: 0.452156

### Reliability diagram data

| Variant | Bin | Range | Count | Mean confidence | Accuracy |
| --- | ---: | --- | ---: | ---: | ---: |
| uncalibrated | 1 | [0.0, 0.1] | 0 | N/A | N/A |
| uncalibrated | 2 | [0.1, 0.2] | 0 | N/A | N/A |
| uncalibrated | 3 | [0.2, 0.3] | 0 | N/A | N/A |
| uncalibrated | 4 | [0.3, 0.4] | 22 | 0.373155 | 0.409091 |
| uncalibrated | 5 | [0.4, 0.5] | 122 | 0.466729 | 0.483607 |
| uncalibrated | 6 | [0.5, 0.6] | 203 | 0.548890 | 0.669951 |
| uncalibrated | 7 | [0.6, 0.7] | 144 | 0.641912 | 0.750000 |
| uncalibrated | 8 | [0.7, 0.8] | 88 | 0.752115 | 0.886364 |
| uncalibrated | 9 | [0.8, 0.9] | 4 | 0.804289 | 1.000000 |
| uncalibrated | 10 | [0.9, 1.0] | 0 | N/A | N/A |
| same_split_temperature | 1 | [0.0, 0.1] | 0 | N/A | N/A |
| same_split_temperature | 2 | [0.1, 0.2] | 0 | N/A | N/A |
| same_split_temperature | 3 | [0.2, 0.3] | 0 | N/A | N/A |
| same_split_temperature | 4 | [0.3, 0.4] | 16 | 0.376861 | 0.500000 |
| same_split_temperature | 5 | [0.4, 0.5] | 60 | 0.469450 | 0.466667 |
| same_split_temperature | 6 | [0.5, 0.6] | 167 | 0.550755 | 0.592814 |
| same_split_temperature | 7 | [0.6, 0.7] | 144 | 0.651016 | 0.708333 |
| same_split_temperature | 8 | [0.7, 0.8] | 106 | 0.745124 | 0.716981 |
| same_split_temperature | 9 | [0.8, 0.9] | 88 | 0.854448 | 0.897727 |
| same_split_temperature | 10 | [0.9, 1.0] | 2 | 0.903788 | 1.000000 |
| cross_fitted_temperature | 1 | [0.0, 0.1] | 0 | N/A | N/A |
| cross_fitted_temperature | 2 | [0.1, 0.2] | 0 | N/A | N/A |
| cross_fitted_temperature | 3 | [0.2, 0.3] | 0 | N/A | N/A |
| cross_fitted_temperature | 4 | [0.3, 0.4] | 16 | 0.376936 | 0.500000 |
| cross_fitted_temperature | 5 | [0.4, 0.5] | 58 | 0.468158 | 0.465517 |
| cross_fitted_temperature | 6 | [0.5, 0.6] | 168 | 0.549325 | 0.589286 |
| cross_fitted_temperature | 7 | [0.6, 0.7] | 144 | 0.650277 | 0.715278 |
| cross_fitted_temperature | 8 | [0.7, 0.8] | 101 | 0.742271 | 0.702970 |
| cross_fitted_temperature | 9 | [0.8, 0.9] | 90 | 0.851080 | 0.888889 |
| cross_fitted_temperature | 10 | [0.9, 1.0] | 6 | 0.908265 | 1.000000 |

## Manual qualitative notes

**TODO (author): Add qualitative observations after reviewing the local error-case CSV.**
