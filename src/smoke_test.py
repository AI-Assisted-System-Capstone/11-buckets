"""Step 2: end-to-end smoke test on a ~500-row stratified slice.

Runs every stage with tiny/fast components, plus negative tests that deliberately
break the data to prove each guardrail hard-fails. Finally runs one real
Bio_ClinicalBERT optimizer step through the same code the Colab notebook uses.

Usage: python src/smoke_test.py [--skip-main-encoder]
Writes outputs/smoke_test_results.json.
"""
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import torch

import pipeline as P
from evaluation import (GuardrailWarning, PlattCalibrator, bucket_sensitivity, calibration_table, choose_threshold,
                        event_metrics, harm_metrics)
from mtl import EmbeddingMultiTask, EncoderMultiTask, multitask_loss, pick_device
from train_encoder import ReportDataset, fit, make_collate, predict_df, train_prior

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY_MODEL = "prajjwal1/bert-tiny"
MAIN_ENCODER = "emilyalsentzer/Bio_ClinicalBERT"  # must match the notebook
MINILM = "sentence-transformers/all-MiniLM-L6-v2"

results = []


def step(name):
    def deco(fn):
        def run(*a, **k):
            t = time.time()
            try:
                detail = fn(*a, **k)
                results.append({"step": name, "status": "PASS", "detail": detail, "sec": round(time.time() - t, 1)})
                print(f"PASS  {name}: {detail}")
                return detail
            except Exception as e:
                results.append({"step": name, "status": "FAIL", "detail": repr(e), "sec": round(time.time() - t, 1)})
                print(f"FAIL  {name}: {e!r}")
                traceback.print_exc()
                return None
        return run
    return deco


def expect_raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return str(e)[:200]
    raise AssertionError(f"expected {exc.__name__}, nothing raised")


@step("1. load + label construction on full CSV, then 500-row stratified slice")
def s_load():
    df = P.load_dataset(os.path.join(ROOT, P.CSV_NAME))
    sl = P.smoke_slice(df, n=500)
    P.check_head_a_classes(sl, "smoke slice")
    assert len(sl) == 500
    assert set(P.EXCLUDED_COLUMNS).isdisjoint(sl.columns)
    s_load.slice = sl
    return {"rows": len(sl), "hurt": int((sl.harm_label == 1).sum()),
            "no_harm_score": int((sl.harm_label == P.UNLABELED).sum()),
            "no_event_code": int((sl.event_type_label == P.UNLABELED).sum()),
            "buckets": sl.event_bucket.value_counts().to_dict()}


@step("2. label rules: 11-bucket prefix + harm cut point")
def s_label_rules():
    assert P.event_bucket("Me - Wrong - Over dosage") == "ME"
    assert P.event_bucket("O-Unanticipated transfer-Intra-facility transfer") == "O"
    assert P.event_bucket("SIGNED OUT AGAINST MEDICAL ADVICE") is None
    assert P.event_bucket("LEFT WITHOUT TREATMENT") is None
    assert [P.harm_label(x) for x in ["A-Unsafe", "B1-Near", "B2-Near", "C-No", "D-No"]] == [0] * 5
    assert [P.harm_label(x) for x in ["E-Harm", "F-Harm", "G-Harm", "H-Harm", "I-Harm"]] == [1] * 5
    assert P.harm_label(np.nan) == P.UNLABELED
    return "ME case-folded; no-dash rows unlabeled; A-D->0, E-I->1, blank->unlabeled"


@step("3. guardrail negative tests (each must hard-fail)")
def s_negative(sl):
    out = {}
    for col in P.EXCLUDED_COLUMNS + P.LABEL_SOURCE_COLUMNS:
        expect_raises(P.LeakageError, P.assert_no_excluded_columns, ["event_comments", col])
    out["leakage: each of 6 excluded + 2 label cols"] = "raised LeakageError"
    feats = pd.DataFrame({c: ["x"] for c in P.FEATURE_COLUMNS})
    feats["manager_comments"] = "leak"
    out["build_input_text with manager_comments"] = expect_raises(P.LeakageError, P.build_input_text, feats)
    bad = sl.copy()
    bad["Level of Invet"] = "RCA"
    out["ReportDataset with Level of Invet"] = expect_raises(P.LeakageError, ReportDataset, bad)
    drop_sh = sl[sl.event_bucket != "SH"]
    out["Head A set missing SH (10 classes)"] = expect_raises(P.DataContractError, P.check_head_a_classes, drop_sh, "t")
    twelve = sl.copy()
    twelve.loc[twelve.index[0], "event_type_label"] = 11
    out["Head A set with a 12th class"] = expect_raises(P.DataContractError, P.check_head_a_classes, twelve, "t")
    full = pd.read_parquet(os.path.join(ROOT, "outputs", "processed", "dataset.parquet"))
    swapped = full.copy()
    swapped["split"] = swapped["split"].map({"TRAIN": "TEST", "TEST": "TRAIN", "VALIDATION": "VALIDATION"})
    out["prevalence check with TRAIN/TEST swapped"] = expect_raises(P.DataContractError, P.check_hurt_prevalence, swapped)
    tmp = tempfile.mkdtemp()
    try:
        raw = pd.read_csv(os.path.join(ROOT, P.CSV_NAME), nrows=20)
        raw.loc[0, P.DATE_COL] = "2024-10-15"  # a TRAIN id dated in the TEST month
        raw.to_csv(os.path.join(tmp, "bad.csv"), index=False)
        out["Event No. split vs Event Date mismatch"] = expect_raises(P.DataContractError, P.load_dataset,
                                                                      os.path.join(tmp, "bad.csv"))
        raw = pd.read_csv(os.path.join(ROOT, P.CSV_NAME), nrows=20)
        raw.loc[0, P.EVENT_TYPE_COL] = "ZZ - Unknown code"
        raw.to_csv(os.path.join(tmp, "bad.csv"), index=False)
        out["unknown event-type code"] = expect_raises(P.DataContractError, P.load_dataset, os.path.join(tmp, "bad.csv"))
    finally:
        shutil.rmtree(tmp)
    return out


@step("4. tokenization (tiny BERT + Bio_ClinicalBERT tokenizers)")
def s_tokenize(sl):
    from transformers import AutoTokenizer
    out = {}
    for name in (TINY_MODEL, MAIN_ENCODER):
        tok = AutoTokenizer.from_pretrained(name)
        batch = make_collate(tok, 512)([(t, 0, 0) for t in sl.text.tolist()[:16]])
        L = tok(sl.text.tolist(), truncation=False)["input_ids"]
        out[name] = {"batch_shape": list(batch[0]["input_ids"].shape),
                     "max_tokens_in_slice": max(map(len, L)), "truncated": int(sum(len(x) > 512 for x in L))}
    return out


@step("5. forward pass through both heads + gradient path Head B -> Head A (tiny encoder)")
def s_forward(sl):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TINY_MODEL)
    inputs, ye, yh = make_collate(tok, 256)(list(zip(sl.text[:32], sl.event_type_label[:32], sl.harm_label[:32])))
    out = {}
    for use in (True, False):
        m = EncoderMultiTask(TINY_MODEL, use_event_probs=use)
        la, lb = m(**inputs)
        assert la.shape == (32, 11) and lb.shape == (32,)
        in_b = m.heads.head_b[-1].in_features
        # BCE alone must reach Head A's weights only when Head B consumes Head A's soft probs.
        yh_all = torch.where(yh < 0, torch.zeros_like(yh), yh)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(lb, yh_all.float())
        bce.backward()
        g = m.heads.head_a[-1].weight.grad
        grad_to_a = float(g.abs().sum()) if g is not None else 0.0
        if use:
            assert grad_to_a > 0, "Head B's loss does not reach Head A: soft probs not wired end-to-end"
        else:
            assert grad_to_a == 0
        out["full" if use else "ablation"] = {"head_a_out": list(la.shape), "head_b_in_features": in_b,
                                             "bce_grad_into_head_a": round(grad_to_a, 6)}
    return out


@step("6. one training step + checkpoint + resume via train_encoder.fit (tiny encoder)")
def s_train_tiny(sl):
    d = tempfile.mkdtemp()
    try:
        cfg = dict(model_name=TINY_MODEL, use_event_probs=True, max_length=256, batch_size=16, eval_batch_size=64,
                   grad_accum=1, lr=1e-4, max_epochs=1, max_steps=1, w_a=1.0, w_b=1.0, seed=0, log_every=1,
                   eval_every=1, patience=3, length_bucketing=True,
                   ckpt_every=1, keep_ckpts=2, ckpt_dir=d, fp16=True)
        tr, va = sl.iloc[:400], sl.iloc[400:]
        dev = pick_device()
        logs = []
        model, tok, hist = fit(tr, va, cfg, dev, log=logs.append)
        cks = sorted(os.listdir(d))
        cfg["max_steps"] = 2
        cfg["max_epochs"] = 2
        logs2 = []
        fit(tr, va, cfg, dev, log=logs2.append)
        resumed = [l for l in logs2 if l.startswith("Resumed")]
        assert resumed, "did not resume from checkpoint"
        pred = predict_df(model, tok, va, cfg, dev, event_prior=train_prior(tr))
        assert pred["probs_a"].shape == (len(va), 11) and "p_hurt_prior" in pred
        return {"device": dev, "first_run_logs": logs[:2], "checkpoint_files": cks, "resume": resumed[0],
                "val_history_keys": sorted(hist[0].keys())}
    finally:
        shutil.rmtree(d)


@step("6b. early stopping: stops after `patience` evals without improvement, restores the best weights")
def s_early_stop(sl):
    import train_encoder as TE
    scores = iter([0.50, 0.45, 0.40, 0.35, 0.30, 0.25] + [0.2] * 50)  # best at the first eval, then worse
    real = TE.val_summary
    TE.val_summary = lambda *a, **k: {"event_macro_f1": next(scores), "event_accuracy": 0.0,
                                      "harm_pr_auc": 0.0, "harm_prevalence": 0.1}
    d = tempfile.mkdtemp()
    try:
        cfg = dict(model_name=TINY_MODEL, use_event_probs=True, max_length=128, batch_size=16, eval_batch_size=64,
                   grad_accum=1, lr=1e-4, max_epochs=3, max_steps=40, w_a=1.0, w_b=1.0, seed=0, log_every=100,
                   eval_every=2, patience=2, length_bucketing=True, ckpt_every=100, keep_ckpts=2, ckpt_dir=d)
        logs = []
        model, _, hist = TE.fit(sl.iloc[:400], sl.iloc[400:], cfg, pick_device(), log=logs.append)
        stop = [l for l in logs if l.startswith("Early stopping")]
        assert stop and "step 6" in stop[0], f"expected early stop at step 6 (evals at 2, 4, 6), got {stop} / {len(hist)} evals"
        assert [h["step"] for h in hist] == [2, 4, 6], [h["step"] for h in hist]
        best = torch.load(os.path.join(d, "best_model.pt"), weights_only=True)
        k = next(iter(best))
        assert torch.equal(model.state_dict()[k].cpu(), best[k].cpu()), "best weights not restored"
        # Resuming a stopped run must not train further.
        logs2 = []
        TE.fit(sl.iloc[:400], sl.iloc[400:], cfg, pick_device(), log=logs2.append)
        assert not any(l.startswith("step ") for l in logs2), "resumed a stopped run and kept training"
        # A checkpoint from a different model must be rejected before any weights load (no state_dict error).
        msg_model = expect_raises(RuntimeError, TE.fit, sl.iloc[:400], sl.iloc[400:], dict(cfg, use_event_probs=False),
                                  pick_device(), log=lambda m: None)
        assert "different model" in msg_model, msg_model
        # Same for a pre-fix checkpoint with no model record (the stale Clinical-Longformer case).
        old = tempfile.mkdtemp()
        try:
            ck = sorted(f for f in os.listdir(d) if f.startswith("step_"))[-1]
            blob = torch.load(os.path.join(d, ck), weights_only=False)
            blob["state"].pop("identity")
            torch.save(blob, os.path.join(old, ck))
            msg_old = expect_raises(RuntimeError, TE.fit, sl.iloc[:400], sl.iloc[400:], dict(cfg, ckpt_dir=old),
                                    pick_device(), log=lambda m: None)
            assert "different model" in msg_old, msg_old
        finally:
            shutil.rmtree(old)
        # Resuming with different batching must hard-fail rather than mis-align the epoch position.
        msg = expect_raises(RuntimeError, TE.fit, sl.iloc[:400], sl.iloc[400:], dict(cfg, batch_size=8), pick_device(),
                            log=lambda m: None)
        return {"evals_at_steps": [h["step"] for h in hist], "stop_log": stop[0], "best_restored": True,
                "resume_after_stop_trains": False, "resume_with_other_model": msg_model[:100], "resume_from_pre_fix_checkpoint": msg_old[:100],
                "resume_with_other_batch_size": msg[:90]}
    finally:
        TE.val_summary = real
        shutil.rmtree(d)


@step("6c. length bucketing: every row once per epoch, reproducible, less padding")
def s_bucketing():
    from train_encoder import epoch_batches
    rng = np.random.RandomState(0)
    lengths = rng.lognormal(4.3, 0.6, 60000).astype(int) + 5
    out = {}
    for bucket in (False, True):
        b1 = epoch_batches(lengths, 32, seed=7, bucket=bucket)
        b2 = epoch_batches(lengths, 32, seed=7, bucket=bucket)
        assert b1 == b2, "not reproducible from the seed"
        flat = np.concatenate(b1)
        assert len(flat) == len(lengths) and len(set(flat)) == len(lengths), "rows missing or repeated"
        padded = sum(len(b) * lengths[b].max() for b in b1)
        out["bucketed" if bucket else "random"] = {"padding_waste": round(1 - lengths.sum() / padded, 3)}
    assert epoch_batches(lengths, 32, seed=7) != epoch_batches(lengths, 32, seed=8), "epochs share an order"
    assert out["bucketed"]["padding_waste"] < out["random"]["padding_waste"] / 3
    return out


@step("6d. batch-size probe runs and reports (tiny model, batch 8/16)")
def s_probe():
    from train_encoder import probe_batch_size
    best, rep = probe_batch_size(TINY_MODEL, 128, pick_device(), candidates=(8, 16), log=lambda m: None)
    assert best == 16, rep
    return {"device": pick_device(), "largest_fitting": best, "report": rep}


@step("7. baseline path: MiniLM embeddings + MLP heads, one step each (full + ablation)")
def s_baseline(sl):
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(MINILM, device=pick_device())
    emb = torch.tensor(enc.encode(sl.text.tolist()[:64], batch_size=64))
    ye = torch.tensor(sl.event_type_label.values[:64])
    yh = torch.tensor(sl.harm_label.values[:64])
    out = {"embedding_dim": emb.shape[1], "max_seq_length": enc.max_seq_length}
    for use in (True, False):
        m = EmbeddingMultiTask(emb.shape[1], use_event_probs=use)
        opt = torch.optim.AdamW(m.parameters(), 1e-3)
        loss, ce, bce = multitask_loss(*m(emb), ye, yh)
        loss.backward()
        opt.step()
        out["full" if use else "ablation"] = {"loss": round(loss.item(), 4), "ce": round(ce.item(), 4),
                                             "bce": round(bce.item(), 4)}
    return out


@step("8. evaluation + guardrail functions on synthetic predictions")
def s_eval():
    rng = np.random.RandomState(0)
    y = (rng.rand(4000) < 0.05).astype(int)
    p = np.clip(0.3 * y + rng.rand(4000) * 0.7, 0, 1)
    thr = choose_threshold(y[:2000], p[:2000])
    assert thr["val_recall"] >= 0.95
    hm = harm_metrics(y[2000:], p[2000:], thr["threshold"])
    cal, ece = calibration_table(y[2000:], PlattCalibrator().fit(y[:2000], p[:2000]).transform(p[2000:]))
    ye = rng.randint(0, 11, 500)
    per, cm, summ = event_metrics(ye, np.eye(11)[ye] * 0.9 + 0.01)
    assert cm.shape == (11, 11)
    buckets = np.array(P.EVENT_BUCKETS)[ye]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bucket_sensitivity(buckets, p[:500], p[:500])
        fired = sum(issubclass(x.category, GuardrailWarning) for x in w)
    assert fired == 11, f"expected the event-type ceiling warning for all 11 buckets on identical inputs, got {fired}"
    return {"threshold_rule": thr["rule"], "val_recall": round(thr["val_recall"], 3),
            "test_recall": round(hm["recall"], 3), "ece_after_platt": round(ece, 4),
            "event-type ceiling warning fired on identical preds": f"{fired}/11 buckets"}


@step("9. ONE real Bio_ClinicalBERT optimizer step (batch 2) via train_encoder.fit")
def s_main_encoder(sl):
    d = tempfile.mkdtemp()
    try:
        cfg = dict(model_name=MAIN_ENCODER, use_event_probs=True, max_length=512, batch_size=2, eval_batch_size=4,
                   grad_accum=1, lr=2e-5, max_epochs=1, max_steps=1, w_a=1.0, w_b=1.0, seed=0, log_every=1,
                   eval_every=1, patience=3, length_bucketing=True,
                   ckpt_every=1, keep_ckpts=1, ckpt_dir=d, fp16=True)
        tr = sl[sl.harm_label >= 0].iloc[:2]
        va = pd.concat([sl[sl.harm_label == 1].iloc[:2], sl[sl.harm_label == 0].iloc[2:4]])
        logs = []
        model, tok, hist = fit(tr, va, cfg, pick_device(), log=logs.append)
        return {"device": pick_device(), "logs": logs, "val_summary": hist[0]}
    finally:
        shutil.rmtree(d)


def main():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")
    s_load()
    sl = getattr(s_load, "slice", None)
    s_label_rules()
    if sl is not None:
        s_negative(sl)
        s_tokenize(sl)
        s_forward(sl)
        s_train_tiny(sl)
        s_early_stop(sl)
        s_bucketing()
        s_probe()
        s_baseline(sl)
    s_eval()
    if sl is not None and "--skip-main-encoder" not in sys.argv:
        s_main_encoder(sl)
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    json.dump(results, open(os.path.join(ROOT, "outputs", "smoke_test_results.json"), "w"), indent=2, default=str)
    n_fail = sum(r["status"] == "FAIL" for r in results)
    print(f"\n{len(results) - n_fail}/{len(results)} smoke steps passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
