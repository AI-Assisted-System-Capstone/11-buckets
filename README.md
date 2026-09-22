

## What the code does

The code reads hospital safety reports and makes two guesses about each one:

1. **What kind of event was it?** One of 11 types (the "11 buckets"), such as fall, medication error or equipment problem.
2. **Was the patient hurt?** Yes or no. This guess uses the report *and* the answer from guess 1.

Then it checks how good those guesses are. Here is what happens, in order, and which file does each part:

1. **Load and clean the data** (`src/prepare_data.py`, using `src/pipeline.py`)
   - Reads the CSV of 80,000 reports.
   - Works out the right answers for training: the event type from the `Event Type` column, and hurt / not hurt from the harm score letter (A–D = not hurt, E–I = hurt).
   - Splits the reports into train (Jan–Aug), validation (Sep) and test (Oct) using the report ID.
   - Removes columns that would give the answer away (see "No cheating" below).
   - Builds the text the model reads: a few intake details (unit, service, age, medications) followed by the report narrative.
   - Runs safety checks and **stops with an error** if anything looks wrong: wrong row counts, a blocked column slipping through, the wrong number of event types, or harm rates that don't match what we expect.

2. **Train the baseline model** (`src/baseline.py`, using `src/mtl.py`)
   - Turns each report into numbers using a small ready-made language model (MiniLM).
   - Trains two small "heads" on top: one guesses the event type, the other guesses hurt / not hurt.
   - Also trains an **ablation** copy where the hurt guess does *not* get the event-type answer, so we can see whether that answer actually helps.
   - Repeats everything with 5 different random seeds to tell a real difference from luck.

3. **Train the main model** (`notebooks/main_model_colab.ipynb`, using `src/train_encoder.py`)
   - Same idea, but with a bigger medical language model (Bio_ClinicalBERT) that is retrained on our reports.
   - Needs a GPU, so it runs in Google Colab and saves progress to Google Drive.

4. **Pick the cut-off and score the models** (`src/evaluation.py`, `src/make_report.py`)
   - The model gives a probability that the patient was hurt. We pick the cut-off on the validation set so that at least 95% of hurt reports get flagged.
   - We apply that cut-off to the test set and measure recall, precision and PR-AUC for "hurt", plus accuracy for each event type.
   - Everything is written to `reports/evaluation_report.md`, along with tables and charts.

## How to test the code

### 1. Set up (once)

You need Python 3.9 or newer. Put `LOCAL_ONLY_student_facing_candidate (1).csv` in the project folder, then run:

```bash
python3 -m venv .venv
.venv/bin/pip install torch transformers sentence-transformers scikit-learn pandas pyarrow nbformat matplotlib
export PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false
```

### 2. Check the data

```bash
.venv/bin/python src/prepare_data.py
```

This loads the data and runs every data check. **It passes if the last line says `All data-contract checks passed.`** If anything is wrong, it stops with an error that says what failed. Results are saved to `outputs/processed/data_checks.json`.

### 3. Run the smoke test (the main test)

```bash
.venv/bin/python src/smoke_test.py
```

This runs the whole system on a small sample of 500 reports, using tiny models so it finishes in about 20 seconds. Each step prints `PASS` or `FAIL`, and the results are saved to `outputs/smoke_test_results.json`. It checks that:

| # | What it checks |
|---|---|
| 1 | The data loads and a balanced 500-report sample can be taken |
| 2 | The labelling rules are right (event-type codes, the A–D / E–I harm cut) |
| 3 | **The safety checks really stop the code.** It breaks the data on purpose (sneaks in a blocked column, removes an event type, adds a 12th, swaps train and test, and so on) and confirms each one causes an error |
| 4 | Reports are turned into tokens correctly |
| 5 | Both heads give outputs of the right shape, and the hurt head really learns from the event-type head |
| 6 | Training, saving and resuming from a checkpoint work |
| 6b | Early stopping stops at the right time and keeps the best version |
| 6c | Grouping reports by length still uses every report exactly once |
| 6d | The GPU batch-size check runs |
| 7 | The baseline (MiniLM) trains for a step, both versions |
| 8 | The scoring and cut-off functions give correct results on made-up predictions |
| 9 | One real training step of the main Bio_ClinicalBERT model works |

Step 9 downloads Bio_ClinicalBERT (about 400 MB). To skip it, run `.venv/bin/python src/smoke_test.py --skip-main-encoder`.

**Last result: all 12 steps pass.**

### 4. Run the full baseline and make the report

```bash
.venv/bin/python src/baseline.py      # trains the baseline + ablation, 5 seeds each (minutes on a laptop)
.venv/bin/python src/make_report.py   # writes reports/evaluation_report.md
```

Open `reports/evaluation_report.md` to see the results. The numbers should match the "Baseline results so far" table further down.

### 5. Run the main model (Google Colab)

1. Open `notebooks/main_model_colab.ipynb` in Google Colab and pick a **T4 GPU** runtime.
2. Upload the CSV to `MyDrive/psrs_harm/` on Google Drive.
3. Run all cells. If the session disconnects, run them again and training picks up from the last checkpoint.
4. Copy `MyDrive/psrs_harm/outputs/main` and `.../main_ablation` into this repo's `outputs/`, then run `src/make_report.py` again to get all four models in one report.

If you edit anything in `src/`, run `.venv/bin/python src/build_notebook.py` so the Colab notebook uses the same code.

## The data

- **80,000 synthetic safety reports** modelled on 24,238 real 2024 hospital reports. They contain no real patient information. The CSV (`LOCAL_ONLY_student_facing_candidate (1).csv`) is **local only and must not be committed or shared**.
- The data is **already split by date**, and the split is encoded in each report's ID (`SYNPROD-TRAIN-…`, etc.):

| Split | Months (2024) | Reports | Reports with a harm score | Share hurt |
|---|---|---|---|---|
| Train | Jan–Aug | 60,000 | 50,091 | 15.1% (harm cases deliberately over-sampled) |
| Validation | Sep | 10,000 | 8,600 | 4.1% (natural rate) |
| Test | Oct | 10,000 | 8,312 | 4.8% (natural rate) |

- **Harm label:** from the `Significance (PSRS Harm score)` letter. **A–D = not hurt, E–I = hurt.** About 16% of reports have no harm score; they are left out of harm training and scoring but are still used to learn event types.
- **Event-type label (the 11 buckets):** the code at the start of the `Event Type` field (there are 231 detailed sub-types):

| Code | Meaning | Reports |
|---|---|---|
| E | General / lab / documentation | 40,194 |
| ME | Medication error | 14,651 |
| C | Care complication | 7,916 |
| EQ | Equipment / product | 3,903 |
| T | Transfusion / blood product | 3,230 |
| I | Security / violence / admin | 2,950 |
| O | Outcome (unplanned transfer, death, discharge) | 2,568 |
| FALL | Fall | 1,961 |
| SI | Skin integrity | 1,312 |
| ADR | Adverse drug reaction | 1,215 |
| SH | Self-harm | **36** (very rare; only 1 in the test set) |

64 reports ("signed out against medical advice", "left without treatment") have no code. They are excluded from the event-type task only.

### No cheating: fields the model is never allowed to see

Some columns are filled in **after** a safety officer has already investigated, so they basically give the answer away. Examples are manager comments or an "investigation level" that is only set to "root cause analysis" for serious harm. A model that reads them would look amazing offline and be useless in real life. These six columns are **blocked in code**, and the pipeline crashes on purpose if any of them reaches the model:

`manager_comments`, `unit_actions_taken`, `shareable_lessons`, `HPI Designation... Name`, `Analyst-Report Type*`, `Level of Invet`

The model only sees what is available when a report is submitted: the free-text narrative (`event_comments`) plus a short prefix of intake fields (unit, service, patient age, and the prescribed / administered / suspect medication and dose). The event type itself is never included in the input because it is one of the things being predicted.

## The AI models (language models) used

These are **text-understanding (encoder) language models**. They read a report and turn it into numbers the classifier can use. They are not chat-style models that write text. No report data is sent to any external AI service; everything runs locally or in our own Google Colab session.

### 1. Baseline: `sentence-transformers/all-MiniLM-L6-v2` (done ✅)

- A small, fast, general-purpose English model (about 22M parameters). It turns each report into a 384-number "embedding".
- It is **frozen**, meaning we don't retrain it. On top of it sit two small neural-network heads (one hidden layer of 256 units, dropout 0.2):
  - **Head A** → 11-way event type
  - **Head B** → hurt / not hurt, using the embedding **plus Head A's 11 probabilities**
- Both heads are trained **together** with one combined loss: `event-type loss + harm loss`, equal weights.
- Reads up to 256 tokens; about 2% of reports get cut off at that limit.
- Trained with 5 different random seeds so we can tell a real difference from luck.
- Runs on a laptop in minutes (Apple MPS or CPU).

### 2. Main model: `emilyalsentzer/Bio_ClinicalBERT`, fine-tuned (built, not trained yet ⏳)

- A BERT-base model (about 110M parameters) pre-trained on biomedical papers and clinical notes, so it already knows medical language.
- Unlike the baseline, the whole model is **fine-tuned**, so it learns from our reports. The same two heads sit on top (single linear layers), with Head B again receiving Head A's probabilities.
- Settings: max 320 tokens (covers 99.8% of reports in full), learning rate 2e-5, 32 reports per update, up to 4 passes over the data, early stopping on validation, fp16 mixed precision.
- Needs a GPU, so it runs in **Google Colab (T4 GPU)** through `notebooks/main_model_colab.ipynb`. Checkpoints are saved to Google Drive so a disconnect doesn't lose progress.
- **Why not Clinical-Longformer?** The original plan used `yikuan8/Clinical-Longformer`, the model the sponsor used. It is designed for very long documents, but these reports are short (median about 76 tokens). It padded every report to 512 tokens and was much slower for no benefit, so it was swapped for Bio_ClinicalBERT. Speed tests are in `outputs/encoder_benchmark.json`.

### 3. Ablation versions of both

The same models with Head B **not** given Head A's probabilities (text only). Everything else is kept identical: settings, seeds and data rows.

### How the "hurt" cut-off is chosen

The model outputs a probability of harm. We choose the cut-off on the **validation** set (natural harm rate) as the highest threshold that still catches at least 95% of hurt reports, and only then apply it to the test set. The test set is never used to tune anything. Because training data has three times the natural harm rate, raw probabilities run high, so we also check **calibration** (Platt scaling). That makes the probabilities readable as real risks without changing which reports get flagged.

## Where things stand right now

| Step | Status |
|---|---|
| Data audit (columns, labels, split, leakage) | ✅ Done |
| Data pipeline with hard safety checks | ✅ Done, all checks pass |
| 500-row end-to-end smoke test | ✅ Passes (`outputs/smoke_test_results.json`) |
| Baseline (MiniLM) + its ablation, 5 seeds each | ✅ Done |
| Evaluation report | ✅ Done for the baseline: `reports/evaluation_report.md` |
| Main model (Bio_ClinicalBERT) + its ablation | ⏳ Code and Colab notebook ready; **not trained yet** |
| Head-to-head comparison with the partner's harm model | ⏳ Waiting for the partner's predictions on the same test rows |

### Baseline results so far (test set, October 2024)

| Model | Recall (hurt) | Precision (hurt) | PR-AUC | Event-type accuracy |
|---|---|---|---|---|
| Baseline, with event-type clue | 95.7% ± 1.3 | 8.1% ± 0.3 | 0.472 ± 0.011 | 87.3% |
| Baseline ablation, no clue | 95.2% ± 1.1 | 8.1% ± 0.2 | 0.466 ± 0.005 | 87.3% |

*(Mean ± standard deviation over 5 seeds. Random-guess PR-AUC is about 0.048.)*

What this means in plain English:

- The baseline **catches about 96% of harmed patients**, but to do so it flags roughly half of all reports, and only about 1 in 12 flagged reports is a real harm case. It is a useful first filter, not a final answer.
- It is **much better than random** at ranking reports (PR-AUC 0.47 vs 0.05).
- **Event-type prediction works well** (87% accuracy). The weak spots are self-harm (too few examples to learn from), "security/violence/admin" and "care complication", where the model misses about 30–40% of cases.
- **The event-type clue does not help the baseline yet.** The difference between the two versions is smaller than the seed-to-seed noise. That is the main open question for the fine-tuned Bio_ClinicalBERT model: a frozen encoder may simply not be able to make use of the clue.

### Known caveats

- The data is synthetic, generated from about 1,258 event templates. Scores here are likely an **optimistic upper bound**; the real test is the sponsor's run on real hospital data.
- The self-harm bucket has only 1 test example, so its test score means nothing. A supplementary train + validation number is reported instead.
- There is no Nov–Dec 2024 data in this file, even though a later hold-out period was mentioned.

## What's in the repo

| Path | What it is |
|---|---|
| `src/pipeline.py` | Column names, blocked columns, 11-bucket and harm label rules, split, and all hard-fail data checks |
| `src/prepare_data.py` | Step 1: runs the pipeline and saves the processed dataset |
| `src/mtl.py` | The two heads (Head B gets Head A's probabilities), the combined loss, prediction |
| `src/baseline.py` | MiniLM baseline and ablation, multi-seed |
| `src/train_encoder.py` | Fine-tuning loop for Bio_ClinicalBERT with Google Drive checkpoint and resume |
| `src/evaluation.py` | Threshold selection, hurt-class metrics, calibration, guardrail checks |
| `src/make_report.py` | Builds the evaluation report from saved predictions |
| `src/smoke_test.py` | Quick end-to-end test on 500 rows, including tests that the safety checks fail when they should |
| `src/benchmark_encoder.py` | Speed tests used to choose the main encoder and batch settings |
| `src/build_notebook.py` | Generates the Colab notebook from `src/`, so the notebook and local code never drift apart |
| `notebooks/main_model_colab.ipynb` | Colab notebook for the main model and its ablation |
| `reports/` | Evaluation report, per-bucket metrics, confusion matrices, calibration tables, per-row test predictions |
| `outputs/` | Processed data, embeddings, model weights, predictions, logs (large; generated by the scripts) |
