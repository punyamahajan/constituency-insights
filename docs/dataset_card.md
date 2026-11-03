# SemEval-2016 Task 6, Task A — Dataset Card

This card records verified documentation and the read-only inspection of the
locally supplied official files. No validation split or model has been created.

## Identity and source

- **Dataset name and version:** SemEval-2016 Task 6: Detecting Stance in Tweets,
  Task A. A separate dataset version identifier is not stated on the official
  task page.
- **Author or maintaining organization:** Saif Mohammad, Svetlana Kiritchenko,
  Parinaz Sobhani, Xiaodan Zhu, and Colin Cherry; rights holder: National
  Research Council Canada (NRC).
- **Authoritative source URL:**
  https://alt.qcri.org/semeval2016/task6/index.php?id=data-and-tools
- **NRC resource and terms:**
  https://www.saifmohammad.com/WebPages/StanceDataset.htm
- **Date accessed:** 2026-11-03
- **Citation:** Mohammad, S., Kiritchenko, S., Sobhani, P., Zhu, X., and Cherry,
  C. (2016). *SemEval-2016 Task 6: Detecting Stance in Tweets*.
  https://aclanthology.org/S16-1003/
- **Local raw-file checksum(s):**
  - `train.csv`: `f2c14b3cd8df4787efa37bafacc79d37017b6ad95e60acab41dd5fb2ccfbdc6e`
  - `test.csv`: `613fb5a49da6a66adf94298865cba78b9e4c96b03dc1dded125950e64a884dae`

## License and permitted use

- **License:** Custom NRC terms of use; not a standard open-data license.
- **License URL:** https://www.saifmohammad.com/WebPages/StanceDataset.htm
- **Research, redistribution, or commercial-use restrictions:** Free for
  research purposes with citation. Attribution is required in papers, online
  posts, and relevant application documentation. Redistribution of the resource
  or source data is prohibited by default. Commercial use and third-party
  hosting require contacting the designated NRC representative. Raw and
  processed tweet text must remain local and untracked in this project.

## Contents

- **Languages:** English.
- **Unit of observation:** A tweet–target pair with a stance annotation.
- **Input fields:** Both files have the exact header `Tweet,Target,Stance,Opinion
  Towards,Sentiment`. They are comma-delimited, double-quoted CSV with doubled
  internal quotes and CR-only record terminators. The loader uses `Tweet`,
  `Target`, and `Stance`; the additional annotations are not normalized.
- **Target field:** `Target`. Verified Task A values: `Atheism`, `Climate Change
  is a Real Concern`, `Feminist Movement`, `Hillary Clinton`, and `Legalization
  of Abortion`.
- **Total sample count:** 4,163 Task A instances: 2,914 training and 1,249 test.
  The supplied `test.csv` also contains 707 Task B `Donald Trump` rows, which
  are explicitly filtered and reported rather than treated as Task A.
- **Sample counts by language:** All reported Task A instances are English.
- **Sample counts by label:** Training: 1,395 AGAINST, 753 FAVOR, 766 NONE.
  Task A test counts are obtained after the explicit target filter.

## Labels

| Label | Verified definition | Local sample count |
| --- | --- | ---: |
| `FAVOR` | Support for the target can be inferred directly or indirectly. | Pending combined validation report |
| `AGAINST` | Opposition to the target can be inferred directly or indirectly. | Pending combined validation report |
| `NONE` | Neither FAVOR nor AGAINST can be inferred. This combines neutral and no-clue annotations; it does not mean opposition. | Pending combined validation report |

## Splits

- **Provided train/validation/test splits:** Official training and gold-labeled
  test splits are present. There is no official validation split. The project
  prepares an in-memory 80/20 development-training/validation partition from
  official training rows only, using seed 42 and combined target-label
  stratification.
- **Split sample counts:** 2,914 Task A train and 1,249 Task A test after
  excluding the 707 explicitly identified Task B rows.
- **How the splits were created:** The paper states that annotated tweets were
  ordered by timestamp, with the first 70% used for training and the last 30%
  for testing.
- **Potential author, target, time, or duplicate leakage:** No ID column is
  present. Inspection found no duplicate tweet text, canonical records, or
  train/test tweet overlap. Any future validation set must be derived from
  training data only; the official test split must be preserved.
- **Derived partition counts:** 2,331 development-training and 583 validation
  rows. All 2,914 eligible rows are assigned exactly once; official test rows
  are excluded from both partitions.
- **Smallest stratum:** Climate Change is a Real Concern / AGAINST has 15 total
  rows, allocated as 12 development-training and 3 validation. Metrics for this
  cell will be unstable.

## Data quality checks

- **Missing values:** No missing or empty values observed in either file.
- **Duplicate records or duplicate text:** None observed within either file;
  no tweet overlap observed between train and test.
- **Encoding or parsing problems:** `test.csv` is ASCII-only and strictly valid
  UTF-8. `train.csv` is mostly ASCII but has 12 non-ASCII bytes across two rows
  and is not valid UTF-8 or Windows-1252. ISO-8859-1 provides a lossless
  one-to-one decode, but the intended characters are unknown. The loader retains
  the resulting C1 controls unchanged and reports zero-based data-row indices
  109 and 1827. It never replaces the bytes silently.
- **Unexpected labels or targets:** Labels are exactly FAVOR, AGAINST, and NONE.
  `Donald Trump` occurs only in 707 Task B test rows and is excluded through an
  explicit test-file filter. Unexpected targets in unfiltered files still fail
  validation.
- **Other validation findings:** The loader rejects missing files, incomplete or
  unexpected schemas, missing/empty fields, unsupported labels, unexpected
  configured targets, duplicate normalized records, and source IDs crossing
  split boundaries.

## Known limitations

- **Collection or sampling bias:** Tweets were collected through target-related
  query hashtags and manually annotated; the resulting corpus is not a random
  sample of political speech.
- **Geographic and temporal coverage:** United States-oriented targets and
  annotators, collected around the 2015–2016 political context.
- **Class or language imbalance:** English only; published per-target label
  distributions are uneven. Local distributions pending verification.
- **Ethical and privacy considerations:** Public social-media text can still
  contain personal information. Do not republish examples or source text.
- **Limits on intended use or generalization:** Results should not be treated as
  representative of current Indian constituency discourse. `NONE` includes
  both neutrality and insufficient evidence.

## Processing record

- **Raw data retained unchanged:** Yes, under
  `data/raw/stance/semeval2016_task6/`; contents are ignored by Git.
- **Processing steps:** The ingestion layer parses the verified schema,
  explicitly removes and reports Task B rows from the combined test file,
  normalizes Task A records in memory, and does not export tweet text. The split
  preparation assigns official training rows in memory and adds a `partition`
  column while retaining `split=train` as provenance.
- **Processed data version:** Not created.
- **Validation command:** `pytest`; unit tests use synthetic fixtures only.
- **Validation split:** Prepared reproducibly in memory only; no split files
  containing tweet text are written.
- **Encoding anomalies in split:** Rows 109 and 1827 are retained unchanged and
  assigned to development-training with seed 42. Their intended characters
  remain unknown.
- **Validation baseline:** A fixed, untuned TF-IDF word unigram/bigram plus
  class-balanced Logistic Regression baseline was fitted on the 2,331-row
  development-training partition and evaluated on the 583-row validation
  partition. The TF-IDF vocabulary was fitted inside the pipeline on
  development-training text only. Official test data was not loaded or used by
  the baseline entry point.
- **Local outputs:** Aggregate metrics, aggregate error counts, and a
  deterministic private set of up to two representative excerpts per observed
  true/predicted error pair are stored under ignored
  `experiments/stance_tfidf_logreg/`. Mentions, URLs, email addresses, and
  phone-like numbers are redacted from the bounded excerpts. The JSON outputs
  contain no tweet text; excerpts remain local and untracked.
- **Model selection status:** `class_weight="balanced"`, word unigrams/bigrams,
  and other settings are initial baseline choices, not tuned optima. No official
  test evaluation has been performed.
