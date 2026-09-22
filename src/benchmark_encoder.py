"""Measure training speed through the notebook's exact fit() code.

Each variant runs N optimizer steps on the real TRAIN rows and reports optimizer steps/s
and the ETA to the notebook's step cap. Numbers are for whatever device this runs on:
MPS locally (fp32; CUDA-only fp16 is off there), CUDA in Colab.

Usage: python src/benchmark_encoder.py <variant> [<variant> ...] [--steps N] [--probe]
Variants: see VARIANTS. Appends results to outputs/encoder_benchmark.json.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import pandas as pd

from mtl import pick_device
from train_encoder import fit, probe_batch_size

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BERT = "emilyalsentzer/Bio_ClinicalBERT"
LONGFORMER = "yikuan8/Clinical-Longformer"
COMMON = dict(eval_batch_size=64, lr=2e-5, max_epochs=4, w_a=1.0, w_b=1.0, seed=42, keep_ckpts=1, fp16=True,
              use_event_probs=True, patience=3)
VARIANTS = {
    "longformer_512_bs8x4": dict(model_name=LONGFORMER, max_length=512, batch_size=8, grad_accum=4, length_bucketing=False),
    "bert_512_bs8x4_nobucket": dict(model_name=BERT, max_length=512, batch_size=8, grad_accum=4, length_bucketing=False),
    "bert_320_bs8x4_bucket": dict(model_name=BERT, max_length=320, batch_size=8, grad_accum=4, length_bucketing=True),
    "bert_320_bs16x2_bucket": dict(model_name=BERT, max_length=320, batch_size=16, grad_accum=2, length_bucketing=True),
    "bert_320_bs32x1_bucket": dict(model_name=BERT, max_length=320, batch_size=32, grad_accum=1, length_bucketing=True),
    "bert_512_bs32x1_bucket": dict(model_name=BERT, max_length=512, batch_size=32, grad_accum=1, length_bucketing=True),
}


def swapped_out_gb():
    """Cumulative macOS swap-outs in GB (None elsewhere). A jump during a run means memory thrashing."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        page = int(re.search(r"page size of (\d+) bytes", out).group(1))
        return int(re.search(r"Swapouts:\s+(\d+)", out).group(1)) * page / 2 ** 30
    except (OSError, AttributeError):
        return None


def bench(name, n_steps, train_df, val_df, device):
    d = tempfile.mkdtemp()
    try:
        cfg = dict(COMMON, **VARIANTS[name], max_steps=n_steps, log_every=10, eval_every=10 ** 9,
                   ckpt_every=10 ** 9, ckpt_dir=d)
        cap = (-(-len(train_df) // cfg["batch_size"]) // cfg["grad_accum"]) * cfg["max_epochs"]
        per_epoch = cap // cfg["max_epochs"]
        logs = []
        t = time.time()
        swap0 = swapped_out_gb()
        fit(train_df, val_df, cfg, device, log=lambda m: (logs.append(m), print(f"[{name}] {m}", flush=True)))
        rate = [float(x) for l in logs for x in re.findall(r"([\d.]+) steps/s", l)][-1]
        swap = None if swap0 is None else round(swapped_out_gb() - swap0, 2)
        return {"variant": name, **VARIANTS[name], "device": device, "optimizer_steps_measured": n_steps,
                "rows_per_optimizer_step": cfg["batch_size"] * cfg["grad_accum"],
                "steps_per_sec": rate, "sec_per_step": 1 / rate,
                "hours_per_epoch": per_epoch / rate / 3600,
                "eta_hours_to_cap": cap / rate / 3600, "cap_steps": cap, "cap_epochs": cfg["max_epochs"],
                "wall_sec_incl_load_and_val": round(time.time() - t, 1),
                "swapped_out_gb_during_run": swap,
                "timing_reliable": swap is None or swap < 0.5}
    finally:
        shutil.rmtree(d)


def main():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    args = sys.argv[1:]
    steps = int(args[args.index("--steps") + 1]) if "--steps" in args else 50
    names = [a for a in args if a in VARIANTS]
    df = pd.read_parquet(os.path.join(ROOT, "outputs", "processed", "dataset.parquet"))
    train_df = df[df.split == "TRAIN"].reset_index(drop=True)
    val_df = df[df.split == "VALIDATION"].sample(256, random_state=0).reset_index(drop=True)
    device = pick_device()
    out_path = os.path.join(ROOT, "outputs", "encoder_benchmark.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else []
    if "--probe" in args:
        for L in (320, 512):
            best, rep = probe_batch_size(BERT, L, device)
            results.append({"probe": BERT, "max_length": L, "device": device, "largest_fitting": best, "report": rep})
            print(f"PROBE {BERT} max_length {L} on {device}: largest fitting batch {best}", flush=True)
            json.dump(results, open(out_path, "w"), indent=2)
    for name in names:
        r = bench(name, steps, train_df, val_df, device)
        results.append(r)
        print(f"RESULT {name} on {device}: {r['steps_per_sec']:.3f} optimizer steps/s ({r['sec_per_step']:.2f} s/step, "
              f"{r['rows_per_optimizer_step']} rows/step) -> {r['hours_per_epoch']:.2f} h/epoch, "
              f"{r['eta_hours_to_cap']:.1f} h to the {r['cap_epochs']}-epoch cap; swapped out "
              f"{r['swapped_out_gb_during_run']} GB -> {'OK' if r['timing_reliable'] else 'UNRELIABLE (memory thrashing)'}",
              flush=True)
        json.dump(results, open(out_path, "w"), indent=2)


if __name__ == "__main__":
    sys.exit(main())
