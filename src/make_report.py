"""Step 6: evaluation report from saved prediction files.

Reads outputs/<variant>/predictions_{validation,test}*.parquet for every variant that
exists (baseline, baseline_ablation, main, main_ablation) and writes
reports/evaluation_report.md plus CSV/PNG/JSON artifacts. Variants that haven't been
run yet (the Colab-only main model) are reported as "not run" -- nothing is estimated.

Usage: python src/make_report.py
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

from evaluation import (PlattCalibrator, bucket_sensitivity, calibration_table, check_rare_bucket_collapse,
                        check_suspicious_performance, choose_threshold, event_metrics, harm_metrics, sh_metrics)
from pipeline import EVENT_BUCKETS, UNLABELED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
REP = os.path.join(ROOT, "reports")
PCOLS = [f"p_event_{b}" for b in EVENT_BUCKETS]
VARIANTS = {
    "baseline": "Baseline: MiniLM (frozen) + MLP heads, Head B gets Head A probs",
    "baseline_ablation": "Baseline ablation: MiniLM + MLP, Head B text-only",
    "main": "Main: Bio_ClinicalBERT fine-tuned, Head B gets Head A probs",
    "main_ablation": "Main ablation: Bio_ClinicalBERT, Head B text-only",
}


def load(variant, split, seed=None):
    d = os.path.join(OUT, variant)
    name = f"predictions_{split}" + (f"_seed{seed}" if seed is not None else "") + ".parquet"
    p = os.path.join(d, name)
    return pd.read_parquet(p) if os.path.exists(p) else None


def primary(variant, split):
    """Seed-0 run for multi-seed variants, else the single run."""
    return load(variant, split, 0) if load(variant, split, 0) is not None else load(variant, split)


def harm_rows(df):
    return df[df.harm_label != UNLABELED]


def evaluate_variant(v):
    val, test = primary(v, "validation"), primary(v, "test")
    if val is None or test is None:
        return None
    hv, ht = harm_rows(val), harm_rows(test)
    thr = choose_threshold(hv.harm_label, hv.p_hurt)
    res = {"threshold": thr,
           "val_harm": harm_metrics(hv.harm_label, hv.p_hurt, thr["threshold"]),
           "test_harm": harm_metrics(ht.harm_label, ht.p_hurt, thr["threshold"])}
    # Calibration: raw, then Platt fit on validation, both checked on test.
    cal_raw, ece_raw = calibration_table(ht.harm_label, ht.p_hurt)
    platt = PlattCalibrator().fit(hv.harm_label.to_numpy(), hv.p_hurt.to_numpy())
    p_cal = platt.transform(ht.p_hurt)
    cal_cal, ece_cal = calibration_table(ht.harm_label, p_cal)
    thr_cal = float(platt.transform([thr["threshold"]])[0])
    res["calibration"] = {"ece_raw": ece_raw, "ece_platt": ece_cal, "threshold_on_calibrated_scale": thr_cal,
                          "table_raw": cal_raw, "table_platt": cal_cal,
                          "test_recall_at_calibrated_threshold": harm_metrics(ht.harm_label, p_cal, thr_cal)["recall"]}
    per, cm, summ = event_metrics(test.event_type_label, test[PCOLS].to_numpy())
    _, _, summ_v = event_metrics(val.event_type_label, val[PCOLS].to_numpy())
    res.update(event_per_bucket=per, event_cm=cm, event_summary=summ, event_summary_val=summ_v)
    res["sh_test"] = sh_metrics(test.event_type_label, test[PCOLS].to_numpy())
    res["sh_val"] = sh_metrics(val.event_type_label, val[PCOLS].to_numpy())
    oof = load(v, "train_oof", 0)
    tr_in = primary(v, "train")
    if oof is not None:
        both = pd.concat([oof, val])
        res["sh_trainval"] = {**sh_metrics(both.event_type_label, both[PCOLS].to_numpy()),
                              "train_source": "5-fold out-of-fold predictions on TRAIN (honest)"}
    elif tr_in is not None:
        both = pd.concat([tr_in, val])
        res["sh_trainval"] = {**sh_metrics(both.event_type_label, both[PCOLS].to_numpy()),
                              "train_source": "IN-SAMPLE predictions on TRAIN (optimistic: model trained on these rows)"}
    res["guardrail_rare_collapse"] = check_rare_bucket_collapse(per, v)
    check_suspicious_performance(res["val_harm"], summ_v, v)
    res["_test"] = test
    return res


def seed_tables(v):
    """Per-seed hurt-class metrics and per-bucket event metrics.

    The threshold is re-chosen on each seed's own validation predictions (nothing reused).
    """
    rows, buckets = [], []
    for seed in range(20):
        val, test = load(v, "validation", seed), load(v, "test", seed)
        if val is None:
            continue
        hv, ht = harm_rows(val), harm_rows(test)
        thr = choose_threshold(hv.harm_label, hv.p_hurt)
        m = harm_metrics(ht.harm_label, ht.p_hurt, thr["threshold"])
        per, _, ev = event_metrics(test.event_type_label, test[PCOLS].to_numpy())
        rows.append({"variant": v, "seed": seed, "threshold": thr["threshold"], "val_recall": thr["val_recall"],
                     "test_recall": m["recall"], "test_precision": m["precision"], "test_f1": m["f1"],
                     "test_pr_auc": m["pr_auc"], "test_event_accuracy": ev["accuracy"],
                     "test_event_macro_f1": ev["macro_f1"]})
        buckets.append(per.assign(variant=v, seed=seed))
    return pd.DataFrame(rows), (pd.concat(buckets, ignore_index=True) if buckets else pd.DataFrame())


def train_configs():
    """Training config + best epochs per baseline variant, to confirm the comparison is like-for-like."""
    out = {}
    runs = os.path.join(OUT, "baseline_seed_runs.csv")
    runs = pd.read_csv(runs) if os.path.exists(runs) else pd.DataFrame()
    for v in ("baseline", "baseline_ablation"):
        p = os.path.join(OUT, v, "train_log_seed0.json")
        if os.path.exists(p):
            cfg = json.load(open(p))["config"]
            eps = runs[runs.variant == v].best_epoch.tolist() if len(runs) else []
            out[v] = {"config": cfg, "best_epochs": eps}
    return out


def pm(series):
    return f"{series.mean():.3f} ± {series.std():.3f}"


def save_cm_png(cm, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rn = cm.div(cm.sum(1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(rn.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(11), EVENT_BUCKETS)
    ax.set_yticks(range(11), EVENT_BUCKETS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(11):
        for j in range(11):
            n = cm.values[i, j]
            if n:
                ax.text(j, i, str(n), ha="center", va="center", fontsize=7,
                        color="white" if rn.values[i, j] > 0.5 else "black")
    ax.set_title(title + "\n(color = row-normalized recall; numbers = counts)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fmt(x, pct=False):
    return f"{x:.1%}" if pct else f"{x:.3f}"


def md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def main():
    os.makedirs(REP, exist_ok=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = {v: evaluate_variant(v) for v in VARIANTS}
    guardrail_msgs = [str(w.message) for w in caught if w.category.__name__ == "GuardrailWarning"]
    ran = [v for v, r in results.items() if r]

    # ----- Guardrail: per-bucket sensitivity of Head B to Head A's input -----
    sens = {}
    for full, abl in (("baseline", "baseline_ablation"), ("main", "main_ablation")):
        if results.get(full):
            t = harm_rows(results[full]["_test"])
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                if "p_hurt_prior" in t:
                    sens[f"{full}: with vs Head A replaced by train prior"] = bucket_sensitivity(
                        t.event_bucket, t.p_hurt, t.p_hurt_prior, name=f"{full}, Head A -> prior")
                if results.get(abl):
                    ta = harm_rows(results[abl]["_test"]).set_index("event_no").loc[t.event_no]
                    sens[f"{full} vs {abl} (separately trained)"] = bucket_sensitivity(
                        t.event_bucket, t.p_hurt.to_numpy(), ta.p_hurt.to_numpy(), name=f"{full} vs {abl}")
            guardrail_msgs += [str(x.message) for x in w if x.category.__name__ == "GuardrailWarning"]

    st = [seed_tables(v) for v in ("baseline", "baseline_ablation")]
    seeds = pd.concat([a for a, _ in st], ignore_index=True)
    seed_buckets = pd.concat([b for _, b in st], ignore_index=True)
    cfgs = train_configs()

    # ----- artifacts -----
    metrics_json = {}
    for v in ran:
        r = results[v]
        r["event_per_bucket"].to_csv(os.path.join(REP, f"per_bucket_{v}.csv"), index=False)
        r["event_cm"].to_csv(os.path.join(REP, f"confusion_matrix_{v}.csv"))
        save_cm_png(r["event_cm"], f"{VARIANTS[v]} - test set", os.path.join(REP, f"confusion_matrix_{v}.png"))
        r["calibration"]["table_raw"].to_csv(os.path.join(REP, f"calibration_{v}_raw.csv"), index=False)
        r["calibration"]["table_platt"].to_csv(os.path.join(REP, f"calibration_{v}_platt.csv"), index=False)
        t = harm_rows(r["_test"])
        t[["event_no", "p_hurt"]].assign(pred_hurt=(t.p_hurt >= r["threshold"]["threshold"]).astype(int)).to_csv(
            os.path.join(REP, f"test_predictions_{v}.csv"), index=False)
        metrics_json[v] = {k: r[k] for k in ("threshold", "val_harm", "test_harm", "event_summary",
                                             "event_summary_val", "sh_test", "sh_val") }
        metrics_json[v]["sh_trainval"] = r.get("sh_trainval")
        metrics_json[v]["calibration"] = {k: r["calibration"][k] for k in
                                          ("ece_raw", "ece_platt", "threshold_on_calibrated_scale",
                                           "test_recall_at_calibrated_threshold")}
    for name, t in sens.items():
        t.assign(comparison=name).to_csv(os.path.join(REP, "bucket_sensitivity.csv"),
                                         mode="a" if name != list(sens)[0] else "w",
                                         header=name == list(sens)[0], index=False)
    if len(seeds):
        seeds.to_csv(os.path.join(REP, "baseline_seed_runs_test.csv"), index=False)
        seed_buckets.to_csv(os.path.join(REP, "baseline_seed_runs_per_bucket_test.csv"), index=False)
    json.dump({"metrics": metrics_json, "guardrail_warnings": guardrail_msgs},
              open(os.path.join(REP, "metrics.json"), "w"), indent=2, default=float)

    # ----- markdown report -----
    L = ["# Evaluation report — harm prediction via 11 event-type buckets", "",
         "Generated by `src/make_report.py` from saved prediction files. All metrics are on the **test set** "
         "(Oct 2024, natural prevalence) unless labeled otherwise; thresholds were chosen on **validation** only.", ""]
    L += ["## Which models ran", "", "| Variant | Status |", "|---|---|"]
    for v, desc in VARIANTS.items():
        L.append(f"| {desc} | {'evaluated' if results[v] else '**not run yet** (Colab-only; run the notebook, copy predictions into `outputs/' + v + '/`, re-run this script)'} |")
    L.append("")

    L += ["## Threshold selection", ""]
    for v in ran:
        th = results[v]["threshold"]
        L.append(f"- **{v}**: threshold **{th['threshold']:.4f}** on P(hurt). Rule: {th['rule']}. "
                 f"On validation that gives recall {fmt(th['val_recall'], True)}, precision {fmt(th['val_precision'], True)}.")
    L.append("")

    L += ["## Head B: hurt-class metrics, test set", ""]
    rows = []
    for v in ran:
        m = results[v]["test_harm"]
        rows.append({"model": v, "recall": fmt(m["recall"], True), "precision": fmt(m["precision"], True),
                     "F1": fmt(m["f1"]), "PR-AUC": fmt(m["pr_auc"]), "ROC-AUC": fmt(m["roc_auc"]),
                     "TP": m["tp"], "FN": m["fn"], "FP": m["fp"], "flagged": fmt(m["flagged_frac"], True)})
    if rows:
        L.append(md_table(pd.DataFrame(rows)))
        m0 = results[ran[0]]["test_harm"]
        L += ["", f"Test hurt prevalence {fmt(m0['prevalence'], True)} (n={m0['n']}, {m0['n_hurt']} hurt). "
              f"PR-AUC of a random model = prevalence. A model that always says \"not hurt\" scores "
              f"{fmt(m0['always_not_hurt_accuracy'], True)} accuracy, which is why accuracy isn't reported here.", ""]

    L += ["## Ablation: does Head A's event-type vector help Head B?", ""]
    for full, abl in (("baseline", "baseline_ablation"), ("main", "main_ablation")):
        if results.get(full) and results.get(abl):
            a, b = results[full]["test_harm"], results[abl]["test_harm"]
            L.append(f"**{full} vs {abl}** (seed 0): PR-AUC {fmt(a['pr_auc'])} vs {fmt(b['pr_auc'])} "
                     f"(Δ {a['pr_auc'] - b['pr_auc']:+.3f}); recall {fmt(a['recall'], True)} vs {fmt(b['recall'], True)}; "
                     f"precision {fmt(a['precision'], True)} vs {fmt(b['precision'], True)}.")
        else:
            L.append(f"**{full} vs {abl}**: not run yet.")
    if len(seeds):
        n_seeds = seeds.groupby("variant").seed.nunique().to_dict()
        L += ["", f"### Baseline across seeds: mean ± sd, test set (seeds per variant: {n_seeds})", "",
              "Each seed's threshold was re-chosen on that seed's own validation predictions, from these "
              "retrained models. No threshold is carried over from the earlier undertrained runs.", ""]
        cols = ["threshold", "val_recall", "test_recall", "test_precision", "test_f1", "test_pr_auc",
                "test_event_accuracy", "test_event_macro_f1"]
        tab = seeds.groupby("variant")[cols].agg(pm).reset_index()
        L.append(md_table(tab))
        f = seeds[seeds.variant == "baseline"].set_index("seed")
        g = seeds[seeds.variant == "baseline_ablation"].set_index("seed")
        L += ["", "Paired difference, full − ablation (same seed):", ""]
        diff = pd.DataFrame({c: [pm((f[c] - g[c]).dropna())] for c in
                             ("test_recall", "test_precision", "test_f1", "test_pr_auc", "test_event_macro_f1")})
        L.append(md_table(diff))
        d = (f.test_pr_auc - g.test_pr_auc).dropna()
        noise = max(seeds.groupby("variant").test_pr_auc.std().max(), 1e-9)
        wins = int((d > 0).sum())
        verdict = ("**meaningful**" if abs(d.mean()) > 2 * noise and abs(d.mean()) > 0.01 else
                   "**not meaningful** (within 2× seed-to-seed noise, or under 0.01 PR-AUC)")
        L.append(f"\nVerdict for the baseline: the PR-AUC gap of {d.mean():+.4f} is {verdict}. "
                 f"Largest per-variant seed-to-seed PR-AUC sd = {noise:.4f}; the full model beat the ablation "
                 f"on {wins}/{len(d)} seeds.")
    if cfgs:
        same = len({json.dumps(c["config"], sort_keys=True) for c in cfgs.values()}) == 1
        L += ["", "### Fairness check: identical training settings", ""]
        for v, c in cfgs.items():
            cap = c["config"]["max_epochs"]
            L.append(f"- **{v}**: {c['config']} — best epochs per seed {c['best_epochs']} "
                     f"({'all below' if c['best_epochs'] and max(c['best_epochs']) < cap - 1 else 'NOT all below'} the {cap}-epoch cap).")
        L.append(f"- Configs identical: **{'yes' if same else 'NO'}**. Same frozen embeddings, same seeds (0–4), same "
                 "train/validation/test rows, same early-stopping rule (validation harm PR-AUC + event macro-F1). "
                 "The only difference is whether Head B receives Head A's 11-way probability vector.")
    L.append("")

    if len(seed_buckets):
        L += ["### Head A per bucket across seeds: mean ± sd, test set", ""]
        agg = seed_buckets.groupby(["bucket", "variant"]).agg(
            precision=("precision", pm), recall=("recall", pm), f1=("f1", pm), support=("support", "first")).reset_index()
        agg["bucket"] = pd.Categorical(agg.bucket, EVENT_BUCKETS, ordered=True)
        agg = agg.sort_values(["bucket", "variant"])
        agg["note"] = np.where(agg.bucket == "SH", "NOT STATISTICALLY MEANINGFUL (n=1 in test)", "")
        L += [md_table(agg), ""]

    L += ["## Head A: event type, test set", ""]
    for v in ran:
        r = results[v]
        s = r["event_summary"]
        L += [f"### {v}", "", f"Accuracy {fmt(s['accuracy'], True)}, macro-F1 {fmt(s['macro_f1'])}, "
              f"weighted-F1 {fmt(s['weighted_f1'])} on n={s['n']} test rows with an event-type code "
              f"(the 7 uncoded test rows are excluded from Head A).", ""]
        per = r["event_per_bucket"].copy()
        per.loc[per.bucket == "SH", "note"] = "NOT STATISTICALLY MEANINGFUL: n=1 in test; see supplementary SH below"
        L.append(md_table(per))
        L += ["", f"Confusion matrix: `reports/confusion_matrix_{v}.csv` / `.png`.", ""]
        sh = r["sh_test"]
        L.append(f"**SH, test (n={sh['n_true_SH']}, NOT statistically meaningful):** precision {fmt(sh['precision'])}, "
                 f"recall {fmt(sh['recall'])}, F1 {fmt(sh['f1'])} ({sh['n_pred_SH']} test rows predicted SH).")
        if r.get("sh_trainval"):
            st = r["sh_trainval"]
            L.append(f"**SH, supplementary, train+val combined (n={st['n_true_SH']}):** precision {fmt(st['precision'])}, "
                     f"recall {fmt(st['recall'])}, F1 {fmt(st['f1'])}. Train part: {st['train_source']}.")
        sv = r["sh_val"]
        L += [f"SH, validation only (n={sv['n_true_SH']}): precision {fmt(sv['precision'])}, recall {fmt(sv['recall'])}, "
              f"F1 {fmt(sv['f1'])}.", ""]

    L += ["## Calibration check", "",
          "Raw probabilities come from a model trained at 15.1% hurt prevalence and are scored at ~4.8%, so they "
          "are expected to over-predict. Platt scaling (logistic on the logit) is fit on validation and checked on "
          "test. It is monotonic, so it doesn't change ranking, PR-AUC, or the recall/precision of the "
          "validation-chosen operating point; it only makes the probabilities readable as risks.", ""]
    for v in ran:
        c = results[v]["calibration"]
        L += [f"### {v}", "", f"ECE raw **{c['ece_raw']:.4f}** → after Platt **{c['ece_platt']:.4f}**. "
              f"Threshold {results[v]['threshold']['threshold']:.4f} raw = {c['threshold_on_calibrated_scale']:.4f} calibrated; "
              f"test recall at it: {fmt(c['test_recall_at_calibrated_threshold'], True)}.", ""]
        tr, tc = c["table_raw"], c["table_platt"]
        comb = pd.DataFrame({"bin": tr.bin, "n (raw)": tr.n, "mean pred (raw)": tr.mean_predicted,
                             "observed (raw)": tr.observed_hurt_rate, "n (Platt)": tc.n,
                             "mean pred (Platt)": tc.mean_predicted, "observed (Platt)": tc.observed_hurt_rate})
        L += [md_table(comb.fillna(np.nan)).replace("nan", "–"), ""]

    L += ["## Guardrails", ""]
    for name, t in sens.items():
        L += [f"**Per-bucket sensitivity to Head A, {name}** (mean |ΔP(hurt)| on test rows with a harm score; warn if < 0.005):", "",
              md_table(t, "{:.4f}"), ""]
    L += ["Guardrail warnings raised while building this report:", ""]
    L += [f"- {m}" for m in guardrail_msgs] or ["- none"]
    L += ["", "Hard checks enforced in code before any training (all passed): split sizes 60k/10k/10k; Event No. "
          "split agrees with Event Date; hurt prevalence within expected bands (train 13.5–16.5%, val 3.0–5.5%, "
          "test 3.5–6.5%); exactly 11 event-type labels in each Head A set; no excluded or label-source "
          "column in the model's features (`LeakageError`).", ""]

    L += ["## Comparison against the partner's model", "",
          "Not done: the partner's predictions aren't available here. Per-row test predictions for each model are in "
          "`reports/test_predictions_<variant>.csv` (Event No., P(hurt), 0/1 at the validation-chosen threshold), "
          "ready to join to the partner's output on `Event No.`. Before comparing, confirm the partner's 97% figure "
          "used these test rows, the A–D/E–I cut, and excluded rows with no harm score.", ""]

    open(os.path.join(REP, "evaluation_report.md"), "w").write("\n".join(L))
    print(f"wrote {os.path.join(REP, 'evaluation_report.md')} (variants: {ran})")


if __name__ == "__main__":
    sys.exit(main())
