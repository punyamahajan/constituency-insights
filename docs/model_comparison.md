# Multi-Model Comparison: TF-IDF + Logistic Regression vs. DistilBERT (Track A & Track B)

This document reports the comparative evaluation of the classical baseline, the text-only transformer model (Track A), and the target-conditioned transformer model (Track B) on SemEval-2016 Task 6, Task A.

---

## 1. Dataset and Evaluation Protocol

- **Dataset:** SemEval-2016 Task 6: Detecting Stance in Tweets (Task A).
- **Target Entities:** `Atheism`, `Climate Change is a Real Concern`, `Feminist Movement`, `Hillary Clinton`, and `Legalization of Abortion`.
- **Target Stance Labels:** `AGAINST`, `FAVOR`, and `NONE`.
- **Input Modalities:**
  - **Baseline (TF-IDF + LogReg):** Text only. Receives normalized tweet text (`text` column).
  - **Track A (DistilBERT Text-Only):** Text only. Encoded via `tokenizer(tweet_text, ...)`.
  - **Track B (DistilBERT Target-Conditioned):** Text paired with target entity. Encoded via `tokenizer(tweet_text, text_pair=target_entity, ...)` using standard Hugging Face paired-text formatting (`[CLS] text [SEP] target [SEP]`). Tweet text remains unpolluted; no evaluation metadata or stance labels are leaked into model inputs.
- **Development/Validation Split:**
  - Derived strictly from the 2,914 official Task A training rows.
  - In-memory stratified 80/20 partition created by `prepare_task_a_validation_split` with fixed random seed 42.
  - **Development-training partition:** 2,331 rows.
  - **Validation partition:** 583 rows.
- **Official Test Set Status:** **The official gold test set has never been loaded or evaluated.** The official test file (`data/raw/stance/semeval2016_task6/test.csv`) was completely excluded from model training, validation, vocabulary generation, tokenizer adaptation, and hyperparameter selection.

---

## 2. Model Architectures and Training Configurations

| Characteristic | Baseline (TF-IDF + LogReg) | Track A (DistilBERT Text-Only) | Track B (DistilBERT Target-Conditioned) |
| :--- | :--- | :--- | :--- |
| **Model ID / Architecture** | Scikit-learn `Pipeline` (TF-IDF + `LogisticRegression`) | `distilbert/distilbert-base-uncased` (6 layers, 66M params) | `distilbert/distilbert-base-uncased` (6 layers, 66M params) |
| **Input Representation** | Word unigrams + bigrams (`ngram_range=(1, 2)`), lowercase | WordPiece tokenization, max sequence length 128 | Paired WordPiece tokenization, max sequence length 128 |
| **Target Conditioning** | None (text-only) | None (text-only) | Paired text input (`text_pair=target`) |
| **Vocabulary** | Fitted strictly on development-training text | Pretrained uncased WordPiece (30,522 tokens) | Pretrained uncased WordPiece (30,522 tokens) |
| **Optimization** | L-BFGS, `max_iter=1000`, `class_weight="balanced"` | AdamW (`lr=2e-5`, `weight_decay=0.01`, linear decay) | AdamW (`lr=2e-5`, `weight_decay=0.01`, linear decay) |
| **Batch Size** | Full-dataset batching | Device batch size 8, grad accum 2 (Effective 16) | Device batch size 8, grad accum 2 (Effective 16) |
| **Epochs / Steps** | Iterative solver until convergence | 3 epochs (438 optimization steps) | 3 epochs (438 optimization steps) |
| **Precision** | 64-bit float | Mixed precision (FP16 AMP) | Mixed precision (FP16 AMP) |
| **Hardware** | CPU | `NVIDIA RTX 4050 Laptop GPU` (6 GB VRAM) | `NVIDIA RTX 4050 Laptop GPU` (6 GB VRAM) |
| **Training Duration** | < 1 second | 81.26 seconds | 36.81 seconds |
| **Peak GPU Memory** | Not applicable | 1,349.52 MB (~1.32 GB) | 1,349.52 MB (~1.32 GB) |
| **Random Seed** | 42 | 42 | 42 |

---

## 3. Overall Performance Comparison

All models were evaluated on the identical 583 validation instances:

| Metric | TF-IDF + LogReg | Track A (DistilBERT Text-Only) | Track B (DistilBERT Target-Conditioned) | Delta (Track B vs. Baseline) | Delta (Track B vs. Track A) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Macro-F1 (Primary)** | **0.5517** (`0.551661`) | **0.6427** (`0.642691`) | **0.6326** (`0.632634`) | **+0.0809** (+14.7%) | **-0.0101** (-1.6%) |
| **Accuracy** | **0.5883** (`0.588336`) | **0.6655** (`0.665523`) | **0.6655** (`0.665523`) | **+0.0772** (+13.1%) | **0.0000** (identical) |
| **Polar Stance F1 (Avg of AGAINST & FAVOR)** | 0.5884 | 0.6556 | **0.6688** | **+0.0804** (+13.7%) | **+0.0132** (+2.0%) |
| **Total Validation Errors** | 240 / 583 | 195 / 583 | 195 / 583 | -45 errors (-18.8%) | 0 errors (equal count) |
| Majority-Class Baseline (`AGAINST`) | Macro-F1: `0.2163` / Acc: `0.4803` | Macro-F1: `0.2163` / Acc: `0.4803` | Macro-F1: `0.2163` / Acc: `0.4803` | — | — |

---

## 4. Per-Class Precision, Recall, F1, and Support

Fixed label reporting order: `AGAINST`, `FAVOR`, `NONE`.

### 4.1 Comparison Across Classes

| Model | Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline (TF-IDF + LogReg)** | **`AGAINST`** | 0.6529 | 0.7321 | 0.6902 | 280 |
| | **`FAVOR`** | 0.4932 | 0.4800 | 0.4865 | 150 |
| | **`NONE`** | 0.5366 | 0.4314 | 0.4783 | 153 |
| **Track A (DistilBERT Text-Only)** | **`AGAINST`** | 0.7226 | 0.7536 | 0.7378 | 280 |
| | **`FAVOR`** | 0.6029 | 0.5467 | 0.5734 | 150 |
| | **`NONE`** | 0.6129 | 0.6209 | 0.6169 | 153 |
| **Track B (DistilBERT Target-Conditioned)** | **`AGAINST`** | 0.7143 | 0.8036 | **0.7563** | 280 |
| | **`FAVOR`** | 0.6043 | 0.5600 | **0.5813** | 150 |
| | **`NONE`** | 0.6124 | 0.5163 | 0.5603 | 153 |

### 4.2 Key Per-Class Observations
1. **Target conditioning strengthens polar stance classification:**
   - `AGAINST` recall jumped from 75.36% (Track A) to **80.36%** (Track B), elevating `AGAINST` F1 from 0.7378 to **0.7563** (+0.0185).
   - `FAVOR` recall improved from 54.67% (Track A) to **56.00%** (Track B), lifting `FAVOR` F1 from 0.5734 to **0.5813** (+0.0079).
2. **Trade-off on neutral stance (`NONE`):**
   - Track B recall for `NONE` decreased from 62.09% to **51.63%**, dropping `NONE` F1 from 0.6169 to **0.5603** (-0.0566).
   - Supplying the explicit target entity conditioned the model to resolve ambiguous or neutral text toward an active stance (primarily `AGAINST`, the majority class) rather than `NONE`.
   - Because Macro-F1 computes an unweighted arithmetic mean across all three classes, the decline in `NONE` (-0.0566) outweighed the gains in `AGAINST` (+0.0185) and `FAVOR` (+0.0079), reducing aggregate Macro-F1 by 0.0101 despite superior polar performance.

---

## 5. Confusion Matrices and Directional Error Analysis

Rows represent the true stance label; columns represent predicted labels in the order `[AGAINST, FAVOR, NONE]`.

### 5.1 Confusion Matrices

```text
Baseline (TF-IDF + Logistic Regression):
               Pred AGAINST   Pred FAVOR   Pred NONE   Total True
True AGAINST       205            45           30         280
True FAVOR          51            72           27         150
True NONE           58            29           66         153
Total Pred         314           146          123         583

Track A (DistilBERT Text-Only):
               Pred AGAINST   Pred FAVOR   Pred NONE   Total True
True AGAINST       211            26           43         280
True FAVOR          51            82           17         150
True NONE           30            28           95         153
Total Pred         292           136          155         583

Track B (DistilBERT Target-Conditioned):
               Pred AGAINST   Pred FAVOR   Pred NONE   Total True
True AGAINST       225            25           30         280
True FAVOR          46            84           20         150
True NONE           44            30           79         153
Total Pred         315           139          129         583
```

### 5.2 Directional Error Pair Comparison (Confusion-Matrix-Derived)

> [!NOTE]
> **Methodology Distinction:** The directional error counts below are **aggregate statistics derived mathematically from each model's saved confusion matrix** via `extract_directional_errors`. They represent partition-wide counts of off-diagonal transitions (where row = true label and column = predicted label in `[AGAINST, FAVOR, NONE]`). They do **not** constitute an example-by-example paired comparison or instance-level model agreement analysis, as individual row-level predictions for Track A and Track B were not saved in their original runs.

| Directional Error Pair | Baseline (TF-IDF) | Track A (Text-Only) | Track B (Target-Conditioned) | Difference (Track B vs. Track A) | Trend Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`AGAINST -> FAVOR`** | 45 | 26 | **25** | -1 | Lowest polar opposition error |
| **`FAVOR -> AGAINST`** | 51 | 51 | **46** | -5 | Target context rescues pro-target statements |
| **`AGAINST -> NONE`** | 30 | 43 | **30** | -13 | Reverted to baseline rate; less over-hedging |
| **`FAVOR -> NONE`** | 27 | 17 | **20** | +3 | Low variation across transformer tracks |
| **`NONE -> AGAINST`** | 58 | 30 | **44** | +14 | Target conditioning induces bias toward against |
| **`NONE -> FAVOR`** | 29 | 28 | **30** | +2 | Stable across all three models |
| **Total Polar Errors (`AGAINST <-> FAVOR`)** | **96** | **77** | **71** | **-6** | **Continuous reduction with richer context** |
| **Total `NONE` Misattributions** | **87** | **58** | **74** | **+16** | Target presence pushes neutral into polar |

### 5.3 Paired Error Overlap & McNemar's Hypothesis Test (Row-Level Verification)

Row-level predictions for all 583 validation instances were generated and verified using the trained models without modifying splits or accessing test data. The 2x2 contingency table below evaluates paired instance-level agreement between the TF-IDF Baseline and DistilBERT Track B:

| Contingency Cell | Definition | Instances | Percentage |
| :--- | :--- | :---: | :---: |
| **$n_{00}$ (Both Correct)** | Both baseline and DistilBERT predicted the true stance | **285** | 48.89% |
| **$n_{01}$ (Baseline Only)** | Baseline was correct, but DistilBERT misclassified | **58** | 9.95% |
| **$n_{10}$ (DistilBERT Only)** | DistilBERT was correct, but baseline misclassified | **103** | 17.67% |
| **$n_{11}$ (Both Incorrect)** | Both models misclassified the validation instance | **137** | 23.50% |
| **Total Validation Instances** | All instances in the derived Task A validation split | **583** | 100.0% |

#### Statistical Significance (McNemar's Test)
- **Discordant Pairs ($n_{01} + n_{10}$):** $58 + 103 = 161$ instances where the models disagreed.
- **Continuity-Corrected McNemar Statistic:**
  $$\chi^2 = \frac{(|103 - 58| - 1)^2}{161} = \frac{44^2}{161} = 12.0248 \quad (df = 1)$$
- **Asymptotic $p$-value:** $p = 5.25 \times 10^{-4}$ ($0.000525$).
- **Uncorrected McNemar Statistic:** $\chi^2 = 12.5776 \quad (p = 0.000390)$.
- **Exact Two-Sided Binomial Test:** $p = 4.87 \times 10^{-4}$ ($0.000487$).
- **Conclusion:** The disagreement between DistilBERT Track B and the TF-IDF baseline is statistically significant at $\alpha = 0.01$, confirming that DistilBERT's net accuracy advantage (+45 correct instances on validation) is highly unlikely under the null hypothesis of symmetric discordant errors.

#### Per-Class Paired Error Overlap Breakdown

| Class | Support | Both Correct | Baseline Only | DistilBERT Only | Both Incorrect |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`AGAINST`** | 280 | 188 (67.1%) | 17 (6.1%) | 37 (13.2%) | 38 (13.6%) |
| **`FAVOR`** | 150 | 54 (36.0%) | 18 (12.0%) | 30 (20.0%) | 48 (32.0%) |
| **`NONE`** | 153 | 43 (28.1%) | 23 (15.0%) | 36 (23.5%) | 51 (33.3%) |

### 5.4 Evidence-Backed Error Patterns (From Baseline Representative Excerpts)

> [!IMPORTANT]
> The qualitative examples below are drawn from the **12 genuine representative validation error excerpts** saved during the TF-IDF baseline run (`experiments/stance_tfidf_logreg/validation_error_analysis.csv`). These excerpts substantiate the domain and linguistic mechanisms that drive persistent errors across models:


1. **Persistent `FAVOR -> AGAINST` Confusion (Opposition Language & Sarcasm):**
   - *Row 1 (`Hillary Clinton`):* `"Hillary is our best choice if we truly want to continue being a progressive nation. #Ohio"` — Misclassified as `AGAINST` due to severe corpus-level class imbalance against Hillary Clinton (3.3:1 AGAINST bias in training) and general political terminology.
   - *Row 25 (`Hillary Clinton`):* `"Yes HRC subject 2 dbl standard Smh Come on [USER] [USER] U Knew [USER] Donations Would b Scrutinized; Spun!"` — Sarcastic phrasing containing defensive surface tokens (`"subject 2 dbl standard"`) that appear adversarial to n-gram and surface encoders.
2. **Lexical Conflation in `NONE -> AGAINST` Errors:**
   - *Row 61 (`Legalization of Abortion`):* `"They say God isn't listening? Just look at that radar today. #prayfornepal"` — True stance is `NONE` (humanitarian prayer regarding Nepal earthquake), but contains religious tokens (`"God"`, `"#prayfornepal"`). In the training set for abortion, religious lexical items correlate overwhelmingly with anti-abortion stance (`AGAINST`), causing false anti-abortion predictions when abortion is the evaluated target.
   - *Row 62 (`Legalization of Abortion`):* `"If you believe in its power; then pray. #Nepal #Baltimore #ISIS #Kenya #GenderChange #Injustice #Slavery #Racism #MiddleEast"` — Multitude of sociopolitical hashtags triggers strong stance activations despite lacking stance on abortion.

---

## 6. Per-Target Breakdown

Comparative evaluation across all 5 SemEval Task A target entities:

| Target | Metric | Baseline (TF-IDF) | Track A (Text-Only) | Track B (Target-Conditioned) | Delta (Track B vs. Track A) | Rows |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Atheism** | Macro-F1<br>Accuracy | 0.4790<br>0.5980 | **0.5245**<br>0.6961 | 0.5212<br>**0.6961** | -0.0033<br>0.0000 | 102 |
| **Climate Change is a Real Concern** | Macro-F1<br>Accuracy | 0.4250<br>0.5190 | 0.4805<br>0.6962 | **0.5028**<br>**0.7468** | **+0.0223**<br>**+0.0506** | 79 |
| **Feminist Movement** | Macro-F1<br>Accuracy | 0.5200<br>0.5564 | **0.6236**<br>**0.6241** | 0.5956<br>0.6165 | -0.0280<br>-0.0076 | 133 |
| **Hillary Clinton** | Macro-F1<br>Accuracy | 0.5666<br>0.6403 | **0.6468**<br>0.6906 | 0.6351<br>**0.7122** | -0.0117<br>**+0.0216** | 139 |
| **Legalization of Abortion** | Macro-F1<br>Accuracy | 0.5240<br>0.6000 | **0.5706**<br>**0.6385** | 0.5314<br>0.5923 | -0.0392<br>-0.0462 | 130 |

*Special Note on Climate Change:* In the validation partition, `Climate Change is a Real Concern` has support of 42 `FAVOR`, 34 `NONE`, and only 3 `AGAINST`. Track B boosted `FAVOR` F1 from 0.7312 to **0.7872** and accuracy from 69.62% to **74.68%**.

---

## 7. Row-Level Prediction Artifacts & Reproducibility

- **Saved Validation Predictions:** Row-level predictions for all 583 validation instances are saved with class probabilities and stable row identifiers:
  - Baseline (TF-IDF + LogReg): `experiments/stance_tfidf_logreg/validation_predictions.json`
  - Track B (DistilBERT Target-Conditioned): `experiments/stance_distilbert_target/validation_predictions.json`
  - Paired Model Comparison Report: `experiments/model_comparison_report.json`
- **Data Safety Guard:** In compliance with Twitter/NRC data-distribution policy and project privacy standards, these files contain solely identifiers (`row_index`, optional `source_id`), `target`, `true_label`, `predicted_label`, and `probabilities`. Raw tweet texts and excerpts are strictly excluded.
- **Reproducibility:** Predictions and paired comparison metrics can be re-generated deterministically without model retraining using:
  ```powershell
  .\.venv\Scripts\python.exe scripts/compare_models.py
  ```

---

## 8. Known Limitations and Scientific Boundary Conditions

1. **Unseen Gold Test Set:** All reported evaluations are strictly measured on the 583-row derived validation split. Generalization to the 1,249 official test rows must not be inferred prior to official protocol release.
2. **McNemar Test Scope and Domain Boundary:** While the McNemar test rejects the null hypothesis of symmetric disagreement ($p < 0.001$), this significance applies strictly to this specific 583-instance validation partition (English US political tweets, 2015–2016). **It does not establish that DistilBERT will outperform classical baselines on out-of-domain data, across different train/val splits, or on Indian constituency discourse.**
3. **Macro-F1 Sensitivity to Neutral Stance:** Track B demonstrates that target conditioning enhances polar stance discrimination (`AGAINST` and `FAVOR`), but penalizes unweighted Macro-F1 due to increased misattribution of neutral (`NONE`) instances to the majority polar class.
4. **Severe Support Asymmetry:** Certain target-label cells (notably Climate Change `AGAINST`, support 3) are too small for stable per-class estimation.
5. **Dataset Domain and Language Boundaries:** SemEval-2016 Task 6 is English-only and reflects United States political discourse from 2015–2016. **These findings do not establish performance on non-English political speech, regional dialects, or Indian constituency discourse.** Any deployment to Indian legislative contexts will require localized annotations, multilingual LLMs, and constituency-specific entity ontologies.
6. **Fixed Seed Evaluation:** All models were trained and evaluated using fixed random seed 42. Variance across stochastic initializations has not been measured.
