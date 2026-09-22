"""Step 3: all-MiniLM-L6-v2 (frozen) + MLP heads baseline, full and ablation.

Head A and Head B are trained jointly on frozen sentence embeddings with the same
multi-task wiring as the main model. The ablation variant drops Head A's probability
vector from Head B's input. Both are trained over several seeds so the ablation
gap can be compared with seed-to-seed noise.

Usage: python src/baseline.py
Writes outputs/baseline/ and outputs/baseline_ablation/ (predictions, metrics, weights).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from evaluation import check_head_balance, event_metrics
from mtl import EmbeddingMultiTask, event_prior_from_labels, multitask_loss, pick_device, predict, seed_everything
from pipeline import SPLITS, UNLABELED, assert_no_excluded_columns, check_head_a_classes, validate_full_dataset
from train_encoder import predictions_frame

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
SEEDS = (0, 1, 2, 3, 4)
CFG = dict(hidden=256, dropout=0.2, lr=1e-3, weight_decay=1e-4, batch_size=256, max_epochs=200, patience=10,
           w_a=1.0, w_b=1.0)


def embed(df):
    path = os.path.join(OUT, "embeddings", "minilm.npy")
    if os.path.exists(path):
        emb = np.load(path)
        if len(emb) == len(df):
            return emb
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(MINILM, device=pick_device())
    lengths = np.array([len(x) for x in enc.tokenizer(df.text.tolist())["input_ids"]])
    print(f"MiniLM max_seq_length={enc.max_seq_length}; inputs truncated: {(lengths > enc.max_seq_length).mean():.2%}")
    t = time.time()
    emb = enc.encode(df.text.tolist(), batch_size=128, show_progress_bar=True, convert_to_numpy=True)
    print(f"embedded {len(df)} reports in {time.time() - t:.0f}s")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, emb)
    json.dump({"model": MINILM, "max_seq_length": enc.max_seq_length,
               "truncated_frac": float((lengths > enc.max_seq_length).mean())},
              open(os.path.join(OUT, "embeddings", "minilm_meta.json"), "w"), indent=2)
    return emb


def batches(X, ye, yh, bs, shuffle=False, rng=None):
    idx = rng.permutation(len(X)) if shuffle else np.arange(len(X))
    for i in range(0, len(X), bs):
        j = idx[i:i + bs]
        yield {"emb": X[j]}, ye[j], yh[j]


def val_score(model, Xv, yev, yhv, device):
    pred = predict(model, batches(Xv, yev, yhv, 4096), device)
    _, _, ev = event_metrics(yev.numpy(), pred["probs_a"])
    m = yhv.numpy() >= 0
    return {"event_macro_f1": ev["macro_f1"], "event_accuracy": ev["accuracy"],
            "harm_pr_auc": float(average_precision_score(yhv.numpy()[m], pred["p_hurt"][m])),
            "harm_prevalence": float(yhv.numpy()[m].mean())}


def train_one(Xtr, yetr, yhtr, Xv, yev, yhv, use_event_probs, seed, device, fixed_epochs=None):
    """Early-stops on validation (harm PR-AUC + event macro-F1) unless fixed_epochs is given."""
    seed_everything(seed)
    model = EmbeddingMultiTask(Xtr.shape[1], use_event_probs, CFG["hidden"], CFG["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    rng = np.random.RandomState(seed)
    best, best_state, best_epoch, bad, history = -1, None, 0, 0, []
    for epoch in range(fixed_epochs or CFG["max_epochs"]):
        model.train()
        for inputs, ye, yh in batches(Xtr, yetr, yhtr, CFG["batch_size"], True, rng):
            la, lb = model(**inputs)
            loss, _, _ = multitask_loss(la, lb, ye, yh, CFG["w_a"], CFG["w_b"])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if fixed_epochs:
            continue
        vs = val_score(model, Xv, yev, yhv, device)
        vs["epoch"] = epoch
        history.append(vs)
        check_head_balance(history)
        score = vs["harm_pr_auc"] + vs["event_macro_f1"]
        if score > best:
            best, best_epoch, bad = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= CFG["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch


def tensors(df, emb, device):
    X = torch.tensor(emb[df.index.to_numpy()], dtype=torch.float32, device=device)
    return X, torch.tensor(df.event_type_label.to_numpy(), device=device), torch.tensor(df.harm_label.to_numpy(), device=device)


def main():
    device = "cpu"  # small MLP on 384-d vectors: CPU is fast and deterministic
    df = pd.read_parquet(os.path.join(OUT, "processed", "dataset.parquet"))
    assert_no_excluded_columns(df.columns)
    checks = validate_full_dataset(df)
    print("data checks passed:", checks["hurt_prevalence"])
    emb = embed(df)
    tr, va, te = (df[df.split == s] for s in SPLITS)
    for name, d in (("train", tr), ("validation", va), ("test", te)):
        check_head_a_classes(d, name)
    Xtr, yetr, yhtr = tensors(tr, emb, device)
    Xv, yev, yhv = tensors(va, emb, device)
    Xte, yete, yhte = tensors(te, emb, device)
    prior = event_prior_from_labels(tr.event_type_label.to_numpy())

    seed_rows = []
    for variant, use in (("baseline", True), ("baseline_ablation", False)):
        vdir = os.path.join(OUT, variant)
        os.makedirs(vdir, exist_ok=True)
        for seed in SEEDS:
            t = time.time()
            model, hist, best_epoch = train_one(Xtr, yetr, yhtr, Xv, yev, yhv, use, seed, device)
            pv = predict(model, batches(Xv, yev, yhv, 4096), device, prior if use else None)
            pt = predict(model, batches(Xte, yete, yhte, 4096), device, prior if use else None)
            predictions_frame(va, pv).to_parquet(os.path.join(vdir, f"predictions_validation_seed{seed}.parquet"))
            predictions_frame(te, pt).to_parquet(os.path.join(vdir, f"predictions_test_seed{seed}.parquet"))
            vs = hist[best_epoch]
            seed_rows.append({"variant": variant, "seed": seed, "best_epoch": best_epoch, **vs,
                              "sec": round(time.time() - t, 1)})
            print(f"{variant} seed {seed}: best epoch {best_epoch}, val {vs}")
            if seed == SEEDS[0]:
                torch.save(model.state_dict(), os.path.join(vdir, "model_seed0.pt"))
                json.dump({"config": CFG, "history": hist, "best_epoch": best_epoch, "encoder": MINILM,
                           "use_event_probs": use}, open(os.path.join(vdir, "train_log_seed0.json"), "w"), indent=2)
                best_epoch0 = best_epoch

        if use:
            # Supplementary SH numbers need honest train predictions: 5-fold out-of-fold on TRAIN.
            oof_a = np.zeros((len(tr), 11), dtype=np.float32)
            oof_b = np.zeros(len(tr), dtype=np.float32)
            strat = np.where(tr.event_type_label.to_numpy() >= 0, tr.event_type_label.to_numpy(), 11)
            for k, (i_fit, i_out) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, strat)):
                m, _, _ = train_one(Xtr[i_fit], yetr[i_fit], yhtr[i_fit], None, None, None, True, 100 + k, device,
                                    fixed_epochs=best_epoch0 + 1)
                p = predict(m, batches(Xtr[i_out], yetr[i_out], yhtr[i_out], 4096), device)
                oof_a[i_out], oof_b[i_out] = p["probs_a"], p["p_hurt"]
                print(f"  OOF fold {k} done")
            predictions_frame(tr, {"probs_a": oof_a, "p_hurt": oof_b}).to_parquet(
                os.path.join(vdir, "predictions_train_oof_seed0.parquet"))

    pd.DataFrame(seed_rows).to_csv(os.path.join(OUT, "baseline_seed_runs.csv"), index=False)
    print(pd.DataFrame(seed_rows).groupby("variant")[["event_macro_f1", "harm_pr_auc"]].agg(["mean", "std"]))


if __name__ == "__main__":
    sys.exit(main())
