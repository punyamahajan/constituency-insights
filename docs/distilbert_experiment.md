# DistilBERT Stance Classification Experiment

This document details the configuration, protocol, execution instructions, and
evaluation methodology for the DistilBERT baseline on SemEval-2016 Task 6, Task A.

---

## 1. Objectives and Scope

The goal of this experiment is to evaluate a lightweight transformer architecture
(`distilbert/distilbert-base-uncased`) under a strict like-for-like protocol against
the existing TF-IDF + Logistic Regression baseline.

Key principles:
- **Direct comparison:** Identical data inputs, identical partitions, identical evaluation metrics.
- **Text-only input:** Predictions rely strictly on normalized tweet text; target names and metadata are not fed to the model.
- **Test set protection:** The official gold test set is completely isolated and never loaded or evaluated.
- **Privacy preservation:** No tweet text, raw records, or validation error excerpts are exported to git-tracked directories.

---

## 2. Model Identifier and Checkpoint Verification

- **Hugging Face Model ID:** `distilbert/distilbert-base-uncased`
- **Architecture:** 6-layer, 768-hidden, 12-heads, 66M parameters.
- **Vocabulary:** 30,522 tokens (uncased).
- **Task Head:** Single linear classification head mapping hidden dimension (768) to 3 output logits corresponding to `AGAINST`, `FAVOR`, and `NONE`.

---

## 3. Fixed Experimental Settings

All settings are fixed in advance (`configs/distilbert_baseline.json`) and must not
be tuned against the validation partition:

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Random seed** | `42` | Consistent with data splitting and classical baseline. |
| **Max sequence length** | `128` | Sufficient for full 140-character tweets plus special tokens. |
| **Device batch size** | `8` | Fits comfortably within 6 GB VRAM budget (RTX 4050 Laptop GPU). |
| **Gradient accumulation steps** | `2` | Provides effective batch size of 16 without exceeding VRAM. |
| **Effective batch size** | `16` | Standard fine-tuning batch size for sequence classification. |
| **Learning rate** | `2e-5` | Recommended AdamW fine-tuning rate for BERT/DistilBERT. |
| **Weight decay** | `0.01` | Regularization on non-bias/non-LayerNorm parameters. |
| **Epochs** | `3` | Standard convergence horizon for fine-tuning on ~2.3k rows. |
| **Warmup ratio** | `0.0` | Fixed initial schedule without warmup tuning. |
| **Precision** | `FP16` | Automated mixed precision (AMP) enabled only on CUDA devices. |
| **Label order** | `AGAINST`, `FAVOR`, `NONE` | Fixed index mapping: `0: AGAINST`, `1: FAVOR`, `2: NONE`. |

### 3.1 CUDA Feasibility Check and OOM Fallback Tiers

Before launching the full 3-epoch training loop, a pre-run feasibility trial is performed on the first accumulation cycle of development-training data on the active device to verify that peak activation memory does not trigger a CUDA Out-of-Memory (OOM) exception.

If an OOM exception is encountered during the feasibility check or training, the runner executes the following fallback tiers strictly in order:
- **Tier 1 (Default):** Device batch size `8`, gradient accumulation `2` (Effective batch size `16`).
- **Tier 2 (Fallback):** Device batch size `4`, gradient accumulation `4` (Effective batch size `16`).
- **Tier 3 (Fallback):** Device batch size `2`, gradient accumulation `8` (Effective batch size `16`).
- **Tier 4 (Fallback):** Device batch size `1`, gradient accumulation `16` (Effective batch size `16`).

**Invariance rules:**
- Effective batch size remains fixed at `16` across all tiers.
- Sequence length (`128`), learning rate (`2e-5`), weight decay (`0.01`), epoch count (`3`), seed (`42`), and precision mode (`FP16`) are strictly preserved.
- When shifting to a fallback tier, CUDA memory is completely cleared (`torch.cuda.empty_cache()`), and the model is re-instantiated fresh from the original pretrained checkpoint with seed `42`.

---

## 4. Split Isolation and Data Protocol

- **Eligible data:** Only the 2,914 official Task A training rows (`data/raw/stance/semeval2016_task6/train.csv`).
- **Derived split:** Reuses the exact in-memory stratified 80/20 partition created by `prepare_task_a_validation_split(data, random_seed=42)`:
  - **Development-training:** 2,331 rows.
  - **Validation:** 583 rows.
- **Test isolation:** `data/raw/stance/semeval2016_task6/test.csv` is not loaded, inspected, or referenced by the training or evaluation pipelines.
- **Encoding anomalies:** The two losslessly decoded ISO-8859-1 anomaly rows (data row indices 109 and 1827) reside in the development-training partition and are retained unchanged.

---

## 5. Fair Comparison with TF-IDF + Logistic Regression

To ensure valid comparative claims:
1. **Identical inputs:** Both models receive only the raw `text` column.
2. **Identical evaluation rows:** Both models are evaluated on the exact same 583 validation instances.
3. **Primary metric:** Three-class macro-F1 across `AGAINST`, `FAVOR`, and `NONE`.
4. **Baseline benchmark:**
   - TF-IDF + Logistic Regression achieved validation macro-F1: `0.5517` (accuracy: `0.5883`).
   - Majority-class baseline (`AGAINST`) macro-F1: `0.2163` (accuracy: `0.4803`).
5. **Statistical comparison:** Both models report identical aggregate tables:
   - Overall accuracy and macro-F1.
   - Per-class precision, recall, F1, and support.
   - 3×3 confusion matrix (`AGAINST`, `FAVOR`, `NONE`).
   - Per-target macro-F1, accuracy, and support breakdown.

---

## 6. How to Run the Experiment

### Prerequisites
Activate the project virtual environment with the `transformer` optional dependency installed:
```powershell
.\.venv\Scripts\Activate.ps1
```

### Execution

#### Track A: Text-Only DistilBERT
Run the text-only DistilBERT experiment entry point:
```powershell
python -c "from constituency_insights.nlp.distilbert import run_official_distilbert_experiment; run_official_distilbert_experiment()"
```
Outputs are automatically written to `experiments/stance_distilbert/`:
- `validation_metrics.json`: Full aggregate evaluation metrics, per-target metrics, and majority baseline context.
- `run_metadata.json`: Runtime configuration, PyTorch/Transformers version info, hardware specs, and execution timestamps.

#### Track B: Target-Conditioned DistilBERT
Run the target-conditioned DistilBERT experiment entry point:
```powershell
python -c "from pathlib import Path; from constituency_insights.nlp.distilbert import run_official_distilbert_experiment; run_official_distilbert_experiment(config_path=Path('configs/distilbert_target_baseline.json'))"
```
Outputs are automatically written to `experiments/stance_distilbert_target/`:
- `validation_metrics.json`: Full aggregate evaluation metrics, per-target metrics, and majority baseline context.
- `run_metadata.json`: Runtime configuration, PyTorch/Transformers version info, hardware specs, and execution timestamps.

---

## 7. Known Limitations and Threats to Validity

1. **Text-Only Input Constraint:** In SemEval-2016 Task 6, tweets are annotated relative to a specific target entity (e.g. `Atheism` or `Hillary Clinton`). When target entities are omitted from the input text, the model must infer stance from implicit lexical or topical cues alone.
2. **Small Strata:** The `Climate Change is a Real Concern` / `AGAINST` stratum has only 3 validation examples. Per-class metrics for this category are subject to high variance.
3. **Fixed Hyperparameters:** No hyperparameter tuning is performed against the validation set to prevent optimistic leakage. Consequently, the reported results reflect initial default behavior rather than the global optimum for this architecture.
4. **Temporal and Domain Specificity:** The data stems from 2015–2016 US political Twitter discourse, which limits direct zero-shot generalization to Indian election constituency contexts.
