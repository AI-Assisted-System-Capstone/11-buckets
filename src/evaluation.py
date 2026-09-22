"""Evaluation metrics, threshold selection, calibration and guardrails."""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score)

from pipeline import EVENT_BUCKETS, N_EVENT_CLASSES, UNLABELED

TARGET_RECALL = 0.95
SH = EVENT_BUCKETS.index("SH")


class GuardrailWarning(UserWarning):
    pass


def warn(msg):
    warnings.warn(msg, GuardrailWarning, stacklevel=2)
    print(f"[GUARDRAIL WARNING] {msg}")


# ---------- Head B: threshold, metrics, calibration ----------

def choose_threshold(y_val, p_val, target_recall=TARGET_RECALL):
    """Highest threshold whose validation recall on the hurt class is >= target.

    Takes validation data only; never call this with test data.
    """
    y_val, p_val = np.asarray(y_val), np.asarray(p_val)
    order = np.argsort(-p_val)
    tp = np.cumsum(y_val[order])
    recall = tp / y_val.sum()
    k = int(np.argmax(recall >= target_recall))  # first (highest-score) cut reaching target
    thr = float(p_val[order][k])
    pred = p_val >= thr
    return {
        "threshold": thr,
        "val_recall": float(y_val[pred].sum() / y_val.sum()),
        "val_precision": float(y_val[pred].mean()),
        "target_recall": target_recall,
        "rule": f"highest threshold with validation hurt-class recall >= {target_recall:.0%}",
    }


def harm_metrics(y, p, threshold):
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "n_hurt": int(y.sum()), "prevalence": float(y.mean()),
        "threshold": float(threshold),
        "recall": float(rec), "precision": float(prec), "f1": float(f1),
        "pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "flagged_frac": float(pred.mean()),
        "accuracy": float((pred == y).mean()),
        "always_not_hurt_accuracy": float(1 - y.mean()),  # the accuracy trap
    }


def calibration_table(y, p, n_bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append({"bin": f"{edges[b]:.0%}-{edges[b + 1]:.0%}", "n": int(m.sum()),
                     "mean_predicted": float(p[m].mean()) if m.any() else np.nan,
                     "observed_hurt_rate": float(y[m].mean()) if m.any() else np.nan})
    t = pd.DataFrame(rows)
    ece = float(np.nansum(t.n * (t.mean_predicted - t.observed_hurt_rate).abs()) / len(y))
    return t, ece


class PlattCalibrator:
    """Logistic recalibration of the hurt logit, fit on validation.

    Monotonic, so it does not change the ranking or the recall/precision achieved at
    the validation-chosen operating point; it only makes the probabilities honest.
    """

    def fit(self, y_val, p_val):
        self.lr = LogisticRegression().fit(self._logit(p_val), y_val)
        return self

    def transform(self, p):
        return self.lr.predict_proba(self._logit(p))[:, 1]

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p)).reshape(-1, 1)


# ---------- Head A ----------

def event_metrics(y, probs_a):
    """Per-bucket P/R/F1 + 11x11 confusion matrix, on rows that have an event-type label."""
    y = np.asarray(y)
    m = y != UNLABELED
    y, pred = y[m], np.asarray(probs_a)[m].argmax(1)
    labels = list(range(N_EVENT_CLASSES))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    per = pd.DataFrame({"bucket": EVENT_BUCKETS, "precision": p, "recall": r, "f1": f, "support": s})
    per["note"] = ""
    per.loc[per.support < 30, "note"] = "UNRELIABLE: n<30, not statistically meaningful"
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=labels),
                      index=[f"true_{b}" for b in EVENT_BUCKETS], columns=[f"pred_{b}" for b in EVENT_BUCKETS])
    summary = {"accuracy": float((pred == y).mean()),
               "macro_f1": float(f.mean()),
               "weighted_f1": float(np.average(f, weights=s)),
               "n": int(len(y))}
    return per, cm, summary


def sh_metrics(y, probs_a):
    """One-vs-rest P/R/F1 for SH."""
    y = np.asarray(y)
    m = y != UNLABELED
    yt, pred = (y[m] == SH).astype(int), (np.asarray(probs_a)[m].argmax(1) == SH).astype(int)
    p, r, f, _ = precision_recall_fscore_support(yt, pred, average="binary", zero_division=0)
    return {"n_true_SH": int(yt.sum()), "n_pred_SH": int(pred.sum()), "precision": float(p),
            "recall": float(r), "f1": float(f)}


# ---------- Guardrails ----------

def check_rare_bucket_collapse(per_bucket, name="", min_support=10):
    """Rare-bucket collapse: a bucket with real support but ~zero recall."""
    bad = per_bucket[(per_bucket.support >= min_support) & (per_bucket.recall < 0.05)]
    for _, r in bad.iterrows():
        warn(f"rare-bucket collapse [{name}]: {r.bucket} recall {r.recall:.3f} on n={r.support}")
    return bad.bucket.tolist()


def check_suspicious_performance(harm_val, event_val, name=""):
    """Leakage: near-perfect offline numbers usually mean leakage."""
    if harm_val["pr_auc"] > 0.98:
        warn(f"possible label leakage [{name}]: validation hurt PR-AUC {harm_val['pr_auc']:.3f}")
    if event_val["accuracy"] > 0.995:
        warn(f"possible label leakage [{name}]: validation event-type accuracy {event_val['accuracy']:.3f}")


def check_head_balance(history):
    """Loss imbalance: one head near chance while the other improves (history: list of per-epoch val dicts)."""
    if len(history) < 2:
        return
    last, first = history[-1], history[0]
    a_chance = last["event_macro_f1"] < 1.5 / N_EVENT_CLASSES
    b_chance = last["harm_pr_auc"] < 1.5 * last["harm_prevalence"]
    if a_chance and last["harm_pr_auc"] > first["harm_pr_auc"]:
        warn("loss imbalance: Head A near chance while Head B improves -- raise w_a")
    if b_chance and last["event_macro_f1"] > first["event_macro_f1"]:
        warn("loss imbalance: Head B near chance while Head A improves -- raise w_b")


def bucket_sensitivity(buckets, p_with, p_without, tol=0.005, name=""):
    """Event-type ceiling signature: per bucket, mean |p_hurt(with Head A input) - p_hurt(without)|.

    `p_without` is either the ablation model's p_hurt, or the full model's p_hurt with Head A's
    vector replaced by the train prior. Warns for every bucket whose predictions are ~identical.
    """
    buckets = np.asarray(buckets, dtype=object)
    d = np.abs(np.asarray(p_with) - np.asarray(p_without))
    rows = []
    for b in EVENT_BUCKETS:
        m = buckets == b
        if m.any():
            rows.append({"bucket": b, "n": int(m.sum()), "mean_abs_diff": float(d[m].mean()),
                         "max_abs_diff": float(d[m].max())})
    t = pd.DataFrame(rows)
    flat = t[t.mean_abs_diff < tol]
    for _, r in flat.iterrows():
        warn(f"event-type ceiling signature [{name}]: bucket {r.bucket} hurt probabilities ~identical with vs "
             f"without Head A input (mean |diff| {r.mean_abs_diff:.4f} < {tol}, n={r.n})")
    return t
