"""Generate notebooks/main_model_colab.ipynb (main model + ablation, trained on Colab).

The notebook writes this repo's src/*.py into the Colab runtime with %%writefile cells,
so it runs exactly the code that was smoke-tested locally -- there is one source of truth.
Re-run this script after editing anything in src/.

Usage: python src/build_notebook.py
"""
import os

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
MODULES = ("pipeline.py", "mtl.py", "evaluation.py", "train_encoder.py", "make_report.py")


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def main():
    cells = [md("""
# Harm prediction via 11 event-type buckets: main model + ablation (Colab)

Trains the main model and its ablation on Colab:

- **Shared encoder:** `emilyalsentzer/Bio_ClinicalBERT`, fine-tuned (standard BERT-base, 512-token limit). Its CLS token is the shared representation.
  It replaced `yikuan8/Clinical-Longformer`, which padded every input to its 512-token attention window even though the narratives are short (median ~76 tokens), so its long-document machinery was pure overhead.
- **Head A:** linear layer → 11-way softmax (event type), cross-entropy.
- **Head B:** linear layer on **[encoder output ; Head A's 11-way soft probabilities]** → sigmoid (hurt), BCE. Soft probabilities, not an argmax, so gradients flow end to end.
- **Loss:** `w_a * CE + w_b * BCE`, masked per row (the 64 uncoded rows train Head B only; rows with no harm score train Head A only).
- **Ablation:** the same model, but Head B sees only the encoder output.
- **Threshold:** swept on **validation** for ≥95% hurt recall, then applied to test.
- **Training:** validation early stopping using the same score as the baseline (val harm PR-AUC + event macro-F1), checked every `eval_every` steps with a `patience` limit and a `max_epochs` cap; the best weights are restored.
- **Speed:** fp16 mixed precision; dynamic per-batch padding with length-bucketed batches; `max_length=320` (covers 99.8% of inputs); the largest batch size that fits the GPU (probed in step 5b) at the same 32 rows per optimizer step. Step 5c times ~50 steps so you can check the ETA before committing.

**Before running:**
1. Runtime → Change runtime type → **T4 GPU**.
2. Upload `LOCAL_ONLY_student_facing_candidate (1).csv` to `MyDrive/psrs_harm/` (or change `DRIVE_DIR` below).

Everything (processed data, checkpoints, predictions, report) is written under `DRIVE_DIR` on Google Drive, never only to `/content`, which Colab wipes on disconnect. If the session disconnects, re-run all cells: training resumes from the latest checkpoint.
"""),
             md("## 1. Mount Google Drive and check the GPU"),
             code("""
from google.colab import drive
drive.mount('/content/drive')

import subprocess
print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout or 'NO GPU: switch runtime to T4')
"""),
             md("## 2. Configuration"),
             code("""
import os
DRIVE_DIR = '/content/drive/MyDrive/psrs_harm'          # all persistent state lives here
CSV_PATH = os.path.join(DRIVE_DIR, 'LOCAL_ONLY_student_facing_candidate (1).csv')
OUT_DIR = os.path.join(DRIVE_DIR, 'outputs')

# Guardrail: refuse to checkpoint anywhere Colab will wipe.
assert OUT_DIR.startswith('/content/drive/'), 'Checkpoints must go to Google Drive, not Colab local disk'
assert os.path.exists(CSV_PATH), f'Upload the CSV to {CSV_PATH}'
os.makedirs(OUT_DIR, exist_ok=True)

EFFECTIVE_BATCH = 32       # rows per optimizer step, same as the original 8 x 4, so optimization dynamics are unchanged

BASE_CFG = dict(
    model_name='emilyalsentzer/Bio_ClinicalBERT',
    max_length=320,        # covers 99.83% of inputs untruncated (p99 = 281 tokens after tokenization; 133 of 80k truncated vs 7 at 512)
    batch_size=8,          # placeholder: step 5b replaces this with the largest batch that fits the GPU...
    grad_accum=4,          # ...and sets grad_accum = EFFECTIVE_BATCH // batch_size
    length_bucketing=True, # similar-length reports batched together; each batch padded only to its longest report
    eval_batch_size=128,
    lr=2e-5,
    max_epochs=4,          # cap only: validation early stopping normally ends sooner
    eval_every=500,        # optimizer steps between validation checks (~3.75 checks per epoch)
    patience=3,            # stop after 3 checks without improvement (same score as the baseline's early stopping)
    w_a=1.0, w_b=1.0,      # start equal; adjust if one head stalls (loss-imbalance warning)
    seed=42,
    log_every=50,          # the step-50 log line prints steps/s and ETA
    ckpt_every=250,        # optimizer steps between Drive checkpoints
    keep_ckpts=2,
    fp16=True,             # mixed precision on CUDA (autocast fp16 + GradScaler)
)
# One checkpoint folder per encoder, so switching models never resumes from another model's checkpoints.
CKPT_ROOT = os.path.join(OUT_DIR, 'checkpoints', BASE_CFG['model_name'].split('/')[-1])
CONFIGS = {
    'main':          dict(BASE_CFG, use_event_probs=True,  ckpt_dir=os.path.join(CKPT_ROOT, 'main')),
    'main_ablation': dict(BASE_CFG, use_event_probs=False, ckpt_dir=os.path.join(CKPT_ROOT, 'main_ablation')),
}
"""),
             md("## 3. Install dependencies"),
             code("!pip -q install transformers scikit-learn pandas pyarrow matplotlib"),
             md("## 4. Project code\n\nThese cells are generated from `src/` by `src/build_notebook.py`, so they are the same code the local smoke test ran.")]
    for m in MODULES:
        body = open(os.path.join(SRC, m)).read()
        cells.append(code(f"%%writefile {m}\n{body}"))

    cells += [
        md("## 5. Data pipeline + hard checks\n\nThis fails loudly if the split, prevalence, 11-class set, or exclusion list is violated."),
        code("""
import sys, json
sys.path.insert(0, '.')
import pandas as pd
from pipeline import load_dataset, validate_full_dataset, check_head_a_classes, assert_no_excluded_columns, SPLITS

df = load_dataset(CSV_PATH)
checks = validate_full_dataset(df)
print(json.dumps({k: v for k, v in checks.items()}, indent=2))
train_df, val_df, test_df = (df[df.split == s].reset_index(drop=True) for s in SPLITS)
for name, d in (('train', train_df), ('validation', val_df), ('test', test_df)):
    check_head_a_classes(d, name)
    assert_no_excluded_columns(d.columns)
os.makedirs(os.path.join(OUT_DIR, 'processed'), exist_ok=True)
df.to_parquet(os.path.join(OUT_DIR, 'processed', 'dataset.parquet'), index=False)
print(train_df.text.iloc[0][:300])
"""),
        md("## 5b. Batch-size probe\n\nThis tries the worst case (every row at `max_length`) with a forward pass, backward pass and optimizer step in fp16, at batch 8, 16, 32, 64 and 128. It keeps the largest batch that stays under 85% of GPU memory, then sets `grad_accum` so each optimizer step still sees 32 rows. If 64 or 128 fits and you want a bigger effective batch, raise `EFFECTIVE_BATCH` (and consider scaling `lr`)."),
        code("""
import torch
from train_encoder import probe_batch_size
device = 'cuda' if torch.cuda.is_available() else 'cpu'
assert device == 'cuda', 'No GPU: switch the runtime to T4'
best, probe_report = probe_batch_size(BASE_CFG['model_name'], BASE_CFG['max_length'], device)
assert best, 'Even batch 8 does not fit at this max_length'
bs = min(best, EFFECTIVE_BATCH)
for cfg in CONFIGS.values():
    cfg.update(batch_size=bs, grad_accum=EFFECTIVE_BATCH // bs)
json.dump(probe_report, open(os.path.join(OUT_DIR, 'batch_probe.json'), 'w'), indent=2)
print(f"Largest batch that fits (worst case): {best}. Training with batch {bs} x grad_accum {EFFECTIVE_BATCH // bs} = {EFFECTIVE_BATCH} rows per optimizer step.")
"""),
        md("## 5c. Speed check: ~50 optimizer steps before committing\n\nThis runs the real training loop and config for 50 steps into a throwaway folder (nothing is kept), then prints steps/s and the projected time. Continue to step 6 only if the ETA is acceptable. Early stopping usually ends training before the `max_epochs` cap, so the cap figure is an upper bound."),
        code("""
import re, shutil, tempfile
from train_encoder import fit
speed_dir = tempfile.mkdtemp(dir='/content')  # throwaway timing run, deliberately not on Drive
speed_cfg = dict(CONFIGS['main'], max_steps=50, log_every=10, eval_every=10**9, ckpt_every=10**9, ckpt_dir=speed_dir)
speed_logs = []
fit(train_df, val_df.sample(512, random_state=0), speed_cfg, device, log=lambda m: (speed_logs.append(m), print(m)))
shutil.rmtree(speed_dir)
rate = [float(x) for l in speed_logs for x in re.findall(r'([\\d.]+) steps/s', l)][-1]
spe = (-(-len(train_df) // CONFIGS['main']['batch_size'])) // CONFIGS['main']['grad_accum']
print(f"\\n{rate:.2f} optimizer steps/s ({EFFECTIVE_BATCH} rows/step) -> {spe / rate / 3600:.2f} h per epoch; "
      f"upper bound {BASE_CFG['max_epochs'] * spe / rate / 3600:.2f} h per model at the {BASE_CFG['max_epochs']}-epoch cap "
      f"(x2 for main + ablation). Early stopping typically ends sooner.")
"""),
        md("## 6. Train the main model (Head B gets Head A's soft probabilities)\n\nThis resumes automatically from the latest Drive checkpoint. It validates every `eval_every` steps and stops early after `patience` checks without improvement. If the session disconnects, re-run steps 1–5b and then this cell: the probe gives the same batch size, and training resumes where it stopped."),
        code("""
from train_encoder import fit, predict_df, predictions_frame, train_prior

def run(variant):
    cfg = CONFIGS[variant]
    model, tok, history = fit(train_df, val_df, cfg, device)
    vdir = os.path.join(OUT_DIR, variant)
    os.makedirs(vdir, exist_ok=True)
    prior = train_prior(train_df) if cfg['use_event_probs'] else None
    for split, d in (('validation', val_df), ('test', test_df), ('train', train_df)):
        pred = predict_df(model, tok, d, cfg, device, event_prior=prior)
        predictions_frame(d, pred).to_parquet(os.path.join(vdir, f'predictions_{split}.parquet'))
    json.dump({'config': cfg, 'history': history}, open(os.path.join(vdir, 'train_log.json'), 'w'), indent=2)
    del model
    torch.cuda.empty_cache()
    return history

history_main = run('main')
history_main
"""),
        md("## 7. Train the ablation (Head B text-only)"),
        code("history_ablation = run('main_ablation')\nhistory_ablation"),
        md("""
## 8. Evaluation report

This runs the same `make_report.py` as the local project, writing to `DRIVE_DIR/reports/`. It includes the validation-chosen threshold, hurt-class metrics for main vs ablation, the per-bucket confusion matrix, calibration, SH flagged as n=1 plus the supplementary train+val number, and the per-bucket event-type-reliance guardrail.

To fold the local baseline numbers into the same report: copy `outputs/baseline/` and `outputs/baseline_ablation/` from the local project into `DRIVE_DIR/outputs/` before running this cell. Or copy `DRIVE_DIR/outputs/main*` back to the local project and run `python src/make_report.py` there.
"""),
        code("""
import make_report
make_report.OUT = OUT_DIR
make_report.REP = os.path.join(DRIVE_DIR, 'reports')
make_report.main()
from IPython.display import Markdown, display
display(Markdown(open(os.path.join(DRIVE_DIR, 'reports', 'evaluation_report.md')).read()))
"""),
    ]
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        "accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"}})
    os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
    path = os.path.join(ROOT, "notebooks", "main_model_colab.ipynb")
    nbf.write(nb, path)
    print(f"wrote {path} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
