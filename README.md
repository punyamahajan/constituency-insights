# Constituency Insights

Constituency Insights is an ongoing machine learning research project exploring political stance classification, historical election-outcome modeling, and whether text-derived features can improve election prediction when the available data supports a valid comparison.

---

## 1. Project Overview & Current Milestones

The project evaluates political stance classification on **SemEval-2016 Task 6 (Task A)** across three benchmark tracks:
1. **Classical Baseline:** TF-IDF (word unigrams + bigrams) + Logistic Regression (`Macro-F1: 0.5517`, `Accuracy: 0.5883`).
2. **Track A (Text-Only Transformer):** DistilBERT (`distilbert/distilbert-base-uncased`) evaluated on tweet text alone (`Macro-F1: 0.6427`, `Accuracy: 0.6655`).
3. **Track B (Target-Conditioned Transformer):** DistilBERT evaluating tweet text paired with target entities via `tokenizer(tweet, text_pair=target)` (`Macro-F1: 0.6326`, `Accuracy: 0.6655`, with polar stance F1 improving to `0.6688`).

All experiments use an in-memory stratified 80/20 train/validation partition (2,331 development-training rows, 583 validation rows, fixed seed 42) derived strictly from official training data. The official gold test set (`data/raw/stance/semeval2016_task6/test.csv`) remains untouched and unevaluated.

---

## 2. Environment Setup

The repository runs on Python 3.11+ (verified on Python 3.13.5 with PyTorch 2.15.0+cu132 on Windows 11).

```powershell
# Activate local virtual environment
.\.venv\Scripts\Activate.ps1

# Install editable package with development and transformer dependencies
pip install -e ".[dev,transformer]"
```

---

## 3. Stance Classification CLI

The repository includes a reusable command-line inference tool at [`scripts/predict_stance.py`](file:///D:/constituency-insights/scripts/predict_stance.py) (also accessible via `python -m constituency_insights.nlp.predict`).

### Model Weights & Provenance Transparency
> [!NOTE]
> The fine-tuned Track B DistilBERT checkpoint is saved locally in [`experiments/stance_distilbert_target/`](file:///D:/constituency-insights/experiments/stance_distilbert_target/) (containing `model.safetensors`, `config.json`, and tokenizer files).
> 
> - By default, the CLI automatically loads this fine-tuned checkpoint (`Fine-Tuned: Yes`, `model_source: fine_tuned_checkpoint`).
> - If the checkpoint directory is deleted or absent, the CLI safely falls back to the cached base model (`distilbert/distilbert-base-uncased` with an untrained sequence-classification head) and issues a prominent warning. **Never interpret base model predictions as fine-tuned benchmark results.**

---

## 4. Usage Commands

### A. Run an Illustrative Demonstration
Run the built-in demonstration with three labeled illustrative examples:
```powershell
.\.venv\Scripts\python.exe scripts/predict_stance.py --demo
```

For JSON-formatted demonstration results:
```powershell
.\.venv\Scripts\python.exe scripts/predict_stance.py --demo --json
```

### B. Single Tweet Prediction (Fine-Tuned Checkpoint)
```powershell
.\.venv\Scripts\python.exe scripts/predict_stance.py --tweet "Hillary is our best choice if we truly want to continue being a progressive nation. #Ohio" --target "Hillary Clinton"
```

**Example Output:**
```text
Prediction Result:
  Model Source:     fine_tuned_checkpoint (experiments\stance_distilbert_target)
  Fine-Tuned:       Yes
  Tweet:            Hillary is our best choice if we truly want to continue being a progressive nation. #Ohio
  Target:           Hillary Clinton
  Predicted Stance: FAVOR
  Class Probabilities:
    - AGAINST : 0.4307 (43.1%)
    - FAVOR   : 0.5452 (54.5%)
    - NONE    : 0.0242 (2.4%)
```

### C. Pure JSON Output
```powershell
.\.venv\Scripts\python.exe scripts/predict_stance.py --tweet "Hillary is our best choice" --target "Hillary Clinton" --json
```

### D. Run Paired Model Validation Comparison
Compare the classical TF-IDF baseline against the fine-tuned Track B DistilBERT model on the exact 583-row validation partition, generating row-level predictions, error overlap, and McNemar's hypothesis test:
```powershell
.\.venv\Scripts\python.exe scripts/compare_models.py
```

Generated outputs (excluded from version control per project data policy):
- `experiments/stance_tfidf_logreg/validation_predictions.json`
- `experiments/stance_distilbert_target/validation_predictions.json`
- `experiments/model_comparison_report.json`

---

## 5. Running the Test Suite

Due to Windows default temporary directory permission restrictions, run `pytest` by setting `TEMP` and `TMP` to an in-project directory:

```powershell
$env:TEMP="D:\constituency-insights\.pytest_tmp"; $env:TMP="D:\constituency-insights\.pytest_tmp"; .\.venv\Scripts\python.exe -m pytest tests/ --basetemp="D:\constituency-insights\.pytest_tmp"
Remove-Item -Recurse -Force "D:\constituency-insights\.pytest_tmp" -ErrorAction SilentlyContinue
```

---

## 6. Scientific Limitations & Boundaries

1. **Dataset Scope:** SemEval-2016 Task 6 is English-only and derived from 2015–2016 US political Twitter discourse. These models do not generalize to multilingual Indian constituency discourse or regional legislative contexts.
2. **Base vs. Fine-Tuned Model:** Pretrained base models without downstream fine-tuning contain randomly initialized sequence classification heads. Predictions from base models reflect architectural priors rather than learned stance signals.
3. **Official Test Partition:** All experimental evaluation metrics are measured on the 583-row derived validation partition. The official gold test set has never been loaded or evaluated.
