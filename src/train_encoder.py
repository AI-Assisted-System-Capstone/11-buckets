"""Fine-tuning loop for the encoder multi-task model.

Used by the Colab notebook and by the local smoke test. Checkpoints every
`ckpt_every` optimizer steps to `ckpt_dir` (point this at Google Drive in Colab)
and resumes from the latest checkpoint if one exists.
"""
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from evaluation import check_head_balance, event_metrics
from mtl import EncoderMultiTask, event_prior_from_labels, linear_warmup_decay, multitask_loss, predict
from pipeline import EVENT_BUCKETS, assert_no_excluded_columns
from sklearn.metrics import average_precision_score


class ReportDataset(Dataset):
    def __init__(self, df):
        # The model only ever sees `text`; hard-fail if anything excluded sneaks in.
        assert_no_excluded_columns(df.columns)
        self.text = df["text"].tolist()
        self.y_event = df["event_type_label"].to_numpy()
        self.y_harm = df["harm_label"].to_numpy()

    def __len__(self):
        return len(self.text)

    def __getitem__(self, i):
        return self.text[i], self.y_event[i], self.y_harm[i]


def make_collate(tokenizer, max_length):
    def collate(batch):
        text, ye, yh = zip(*batch)
        enc = tokenizer(list(text), truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        inputs = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        return inputs, torch.as_tensor(np.array(ye)), torch.as_tensor(np.array(yh))
    return collate


def _ckpts(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return []
    return sorted(f for f in os.listdir(ckpt_dir) if f.startswith("step_") and f.endswith(".pt"))


def _latest_ckpt(ckpt_dir):
    cks = _ckpts(ckpt_dir)
    return os.path.join(ckpt_dir, cks[-1]) if cks else None


def _save_ckpt(path, model, opt, sched, scaler, state):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict() if scaler else None, "state": state}, tmp)
    os.replace(tmp, path)  # atomic, so a disconnect mid-save can't corrupt the latest checkpoint


def token_lengths(tok, texts, max_length):
    return np.array([len(x) for x in tok(list(texts), truncation=True, max_length=max_length)["input_ids"]])


def epoch_batches(lengths, batch_size, seed, bucket=True, chunk_batches=50):
    """Batch index lists for one epoch, reproducible from `seed` (so a resume replays the same order).

    With `bucket`, the shuffled rows are cut into chunks of `chunk_batches` batches and each chunk is
    sorted by token length, so a batch holds similarly sized reports and dynamic padding wastes little.
    Batch order is then shuffled again so training doesn't see a short-to-long curriculum.
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(lengths))
    batches = []
    chunk = batch_size * chunk_batches if bucket else len(idx)
    for i in range(0, len(idx), chunk):
        c = idx[i:i + chunk]
        if bucket:
            c = c[np.argsort(lengths[c], kind="stable")]
        batches += [c[j:j + batch_size].tolist() for j in range(0, len(c), batch_size)]
    return [batches[k] for k in rng.permutation(len(batches))] if bucket else batches


def predict_df(model, tok, df, cfg, device, event_prior=None):
    """Predict in length-sorted order (minimal padding), return results in the original row order."""
    order = np.argsort(token_lengths(tok, df["text"], cfg["max_length"]), kind="stable")
    loader = DataLoader(ReportDataset(df.iloc[order]), batch_size=cfg["eval_batch_size"],
                        collate_fn=make_collate(tok, cfg["max_length"]))
    pred = predict(model, loader, device, event_prior)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return {k: v[inv] for k, v in pred.items()}


def val_summary(model, tok, val_df, cfg, device):
    """Same validation score as the baseline's early stopping: harm PR-AUC + event macro-F1."""
    pred = predict_df(model, tok, val_df, cfg, device)
    _, _, ev = event_metrics(val_df.event_type_label.to_numpy(), pred["probs_a"])
    y = val_df.harm_label.to_numpy()
    m = y >= 0
    return {"event_macro_f1": ev["macro_f1"], "event_accuracy": ev["accuracy"],
            "harm_pr_auc": float(average_precision_score(y[m], pred["p_hurt"][m])),
            "harm_prevalence": float(y[m].mean())}


def fit(train_df, val_df, cfg, device, log=print):
    """Train one model (full or ablation) per `cfg`. Returns (model, tokenizer, history).

    Validation-based early stopping (same score as the baseline: val harm PR-AUC + event macro-F1):
    evaluate every `eval_every` optimizer steps, stop after `patience` evaluations without
    improvement, cap at `max_epochs`, and restore the best weights at the end.
    """
    from transformers import AutoTokenizer

    torch.manual_seed(cfg["seed"])
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    collate = make_collate(tok, cfg["max_length"])
    train_ds = ReportDataset(train_df)
    lengths = token_lengths(tok, train_df["text"], cfg["max_length"])

    model = EncoderMultiTask(cfg["model_name"], use_event_probs=cfg["use_event_probs"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    n_batches = -(-len(train_ds) // cfg["batch_size"])
    steps_per_epoch = n_batches // cfg["grad_accum"]
    total = steps_per_epoch * cfg["max_epochs"]
    if cfg.get("max_steps"):
        total = min(total, cfg["max_steps"])
    sched = linear_warmup_decay(opt, int(0.06 * total), total)
    use_amp = device == "cuda" and cfg.get("fp16", True)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    best_path = os.path.join(cfg["ckpt_dir"], "best_model.pt")

    identity = {"model_name": cfg["model_name"], "use_event_probs": cfg["use_event_probs"]}
    batching = {k: cfg[k] for k in ("batch_size", "grad_accum", "max_length", "seed")}
    batching["length_bucketing"] = cfg.get("length_bucketing", True)
    state = {"step": 0, "epoch": 0, "batch_in_epoch": 0, "history": [], "best_score": -1.0,
             "bad_evals": 0, "stopped": False, "identity": identity, "batching": batching}
    ck = _latest_ckpt(cfg["ckpt_dir"])
    if ck:
        blob = torch.load(ck, map_location=device, weights_only=False)
        saved = blob["state"]
        # Check compatibility before touching any weights, so a stale checkpoint gives a clear message.
        if saved.get("identity") != identity:
            raise RuntimeError(
                f"{cfg['ckpt_dir']} holds a checkpoint from a different model ({saved.get('identity') or 'an older run with no model record, e.g. Clinical-Longformer'}); "
                f"this run is {identity}. Point ckpt_dir at a fresh folder, or move that folder aside if you "
                f"no longer need it. Nothing was loaded.")
        if saved.get("batching") != batching:
            raise RuntimeError(f"Checkpoint {ck} was trained with {saved.get('batching')}, but this run uses "
                               f"{batching}; the saved position in the epoch would not line up. Set the same values "
                               f"(e.g. batch_size/grad_accum from outputs/batch_probe.json) or use a fresh ckpt_dir.")
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        sched.load_state_dict(blob["sched"])
        if scaler and blob["scaler"]:
            scaler.load_state_dict(blob["scaler"])
        state = saved
        log(f"Resumed from {ck} at step {state['step']} (epoch {state['epoch']}, batch {state['batch_in_epoch']})")

    def save_ckpt():
        _save_ckpt(os.path.join(cfg["ckpt_dir"], f"step_{state['step']:07d}.pt"), model, opt, sched, scaler, state)
        for old in _ckpts(cfg["ckpt_dir"])[:-cfg["keep_ckpts"]]:
            os.remove(os.path.join(cfg["ckpt_dir"], old))

    def evaluate():
        vs = val_summary(model, tok, val_df, cfg, device)
        vs.update(step=state["step"], epoch=state["epoch"])
        state["history"].append(vs)
        log(f"validation @ step {state['step']}: {vs}")
        check_head_balance(state["history"])
        score = vs["harm_pr_auc"] + vs["event_macro_f1"]
        if score > state["best_score"]:
            state["best_score"], state["bad_evals"] = score, 0
            torch.save(model.state_dict(), best_path)
            log(f"  new best (val harm PR-AUC + event macro-F1 = {score:.4f}) -> best_model.pt")
        else:
            state["bad_evals"] += 1
            log(f"  no improvement ({state['bad_evals']}/{cfg['patience']})")
            if state["bad_evals"] >= cfg["patience"]:
                state["stopped"] = True
                log(f"Early stopping at step {state['step']}: best score {state['best_score']:.4f}")
        model.train()

    json.dump(cfg, open(os.path.join(cfg["ckpt_dir"], "config.json"), "w"), indent=2)
    log(f"{len(train_ds)} train rows, {steps_per_epoch} optimizer steps/epoch, cap {total} steps "
        f"({cfg['max_epochs']} epochs), eval every {cfg['eval_every']} steps, patience {cfg['patience']}, "
        f"batch {cfg['batch_size']} x accum {cfg['grad_accum']}, fp16={use_amp}, "
        f"length bucketing={cfg.get('length_bucketing', True)}")
    t0, start_step = time.time(), state["step"]
    while not state["stopped"] and state["step"] < total and state["epoch"] < cfg["max_epochs"]:
        model.train()
        batches = epoch_batches(lengths, cfg["batch_size"], cfg["seed"] + state["epoch"],
                                bucket=cfg.get("length_bucketing", True))
        start_b = state["batch_in_epoch"]  # resume mid-epoch: skip batches already trained on
        loader = DataLoader(train_ds, batch_sampler=batches[start_b:], collate_fn=collate)
        opt.zero_grad(set_to_none=True)  # don't carry a partial accumulation across epochs/resumes
        for b, (inputs, ye, yh) in enumerate(loader, start=start_b):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            ye, yh = ye.to(device), yh.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                la, lb = model(**inputs)
            loss, ce, bce = multitask_loss(la.float(), lb.float(), ye, yh, cfg["w_a"], cfg["w_b"])
            loss = loss / cfg["grad_accum"]
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            state["batch_in_epoch"] = b + 1
            if (b + 1) % cfg["grad_accum"]:
                continue
            if scaler:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            state["step"] += 1
            if device == "mps" and state["step"] % 10 == 0:
                torch.mps.empty_cache()  # MPS caches a buffer per batch shape; dynamic padding makes many
            if state["step"] % cfg["log_every"] == 0:
                rate = (state["step"] - start_step) / max(1e-9, time.time() - t0)
                eta_h = (total - state["step"]) / max(rate, 1e-9) / 3600
                log(f"step {state['step']}/{total} loss {loss.item() * cfg['grad_accum']:.4f} "
                    f"(CE {ce.item():.4f}, BCE {bce.item():.4f}) {rate:.2f} steps/s, "
                    f"ETA to the {total}-step cap {eta_h:.2f} h (early stopping may end sooner)")
            if state["step"] % cfg["eval_every"] == 0:
                evaluate()
            if state["step"] % cfg["ckpt_every"] == 0 or state["stopped"]:
                save_ckpt()
            if state["stopped"] or state["step"] >= total:
                break
        else:
            state["epoch"] += 1
            state["batch_in_epoch"] = 0
            save_ckpt()
            continue
        break

    if not state["history"] or state["history"][-1]["step"] != state["step"]:
        evaluate()  # score the final weights too, so the best checkpoint considers them
        save_ckpt()
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    return model, tok, state["history"]


def probe_batch_size(model_name, max_length, device, candidates=(8, 16, 32, 64, 128), headroom=0.85, log=print):
    """Largest batch size whose worst case (every row at max_length) survives forward + backward +
    optimizer step, using the same mixed-precision setup as training. Returns (best, report)."""
    report, best = [], None
    use_amp = device == "cuda"
    if device == "cuda":
        total_mem = torch.cuda.get_device_properties(0).total_memory
    elif device == "mps":
        total_mem = torch.mps.recommended_max_memory()
    else:
        return None, [{"note": "no GPU"}]
    for bs in candidates:
        model = opt = None
        try:
            model = EncoderMultiTask(model_name, use_event_probs=True).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
            scaler = torch.amp.GradScaler("cuda") if use_amp else None
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            ids = torch.randint(1000, 20000, (bs, max_length), device=device)
            mask = torch.ones_like(ids)
            for _ in range(2):  # second step includes the allocated optimizer state
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    la, lb = model(ids, mask)
                loss = la.float().logsumexp(-1).mean() + lb.float().mean()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
                opt.zero_grad(set_to_none=True)
            if device == "cuda":
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_reserved()
            else:
                torch.mps.synchronize()
                peak = torch.mps.driver_allocated_memory()  # current driver allocation (MPS has no peak counter)
            ok = peak < headroom * total_mem
            report.append({"batch_size": bs, "peak_gb": round(peak / 2 ** 30, 2),
                           "device_gb": round(total_mem / 2 ** 30, 2), "fits_with_headroom": ok})
            log(f"batch {bs} x {max_length} tokens: peak {peak / 2 ** 30:.2f} GB of {total_mem / 2 ** 30:.2f} GB"
                f" -> {'OK' if ok else 'too close to the limit'}")
            if not ok:
                break
            best = bs
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            report.append({"batch_size": bs, "oom": True})
            log(f"batch {bs} x {max_length} tokens: OOM")
            break
        finally:
            del model, opt
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()
    return best, report


def predictions_frame(df, pred):
    """Uniform prediction file format consumed by make_report.py."""
    out = df[["event_no", "split", "event_bucket", "event_type_label", "harm_label"]].reset_index(drop=True).copy()
    out["p_hurt"] = pred["p_hurt"]
    if "p_hurt_prior" in pred:
        out["p_hurt_prior"] = pred["p_hurt_prior"]
    for i, b in enumerate(EVENT_BUCKETS):
        out[f"p_event_{b}"] = pred["probs_a"][:, i]
    return out


def train_prior(train_df):
    return event_prior_from_labels(train_df.event_type_label.to_numpy())
