"""Data pipeline for the harm / event-type model.

This file is the source of truth for field names, the harm cut point,
the 11-bucket rule and the exclusion list. Every check here raises (hard-fails)
rather than logging, and uses explicit `raise` so it survives `python -O`.
"""
import re

import numpy as np
import pandas as pd

CSV_NAME = "LOCAL_ONLY_student_facing_candidate (1).csv"

ID_COL = "Event No."
DATE_COL = "Event Date"
EVENT_TYPE_COL = "Event Type"
HARM_COL = "Significance (PSRS Harm score)"
TEXT_COL = "event_comments"

# Post-investigation fields: must never reach the model.
EXCLUDED_COLUMNS = (
    "manager_comments",
    "unit_actions_taken",
    "shareable_lessons",
    "HPI Designation... Name",
    "Analyst-Report Type*",
    "Level of Invet",
)
# Label sources are also forbidden as features (they are the answers).
LABEL_SOURCE_COLUMNS = (EVENT_TYPE_COL, HARM_COL)

# Intake fields used as model input: narrative plus a short structured prefix.
PREFIX_FIELDS = (
    ("Unit", ("Location Name",)),
    ("Service", ("Encounter Service",)),
    ("Age", ("Age at Encounter",)),
    ("Prescribed", ("ME - Prescribed - Name *... Name", "ME - Prescribed - Dose *")),
    ("Administered", ("ME - Admin - Name", "ME - Admin - Dose")),
    ("ADR suspect medication", ("ADR - Suspect Med Name", "ADR - Dose *")),
)
FEATURE_COLUMNS = (TEXT_COL,) + tuple(c for _, cols in PREFIX_FIELDS for c in cols)

# Index in this tuple = class id for event_type_label.
EVENT_BUCKETS = ("ADR", "C", "E", "EQ", "FALL", "I", "ME", "O", "SH", "SI", "T")
N_EVENT_CLASSES = len(EVENT_BUCKETS)

SPLITS = ("TRAIN", "VALIDATION", "TEST")
SPLIT_MONTHS = {"TRAIN": set(range(1, 9)), "VALIDATION": {9}, "TEST": {10}}
# Audit: 15.11% / 4.13% / 4.85% hurt among scored rows (split / distribution-shift guardrail).
EXPECTED_HURT_PREVALENCE = {
    "TRAIN": (0.135, 0.165),
    "VALIDATION": (0.030, 0.055),
    "TEST": (0.035, 0.065),
}

UNLABELED = -1


class DataContractError(RuntimeError):
    """Raised when the data violates a data-contract guarantee."""


class LeakageError(DataContractError):
    """Raised when an excluded or label-source column reaches the feature set."""


def assert_no_excluded_columns(columns):
    """Hard-fail if any excluded or label-source column is in `columns`."""
    leaked = sorted(set(columns) & (set(EXCLUDED_COLUMNS) | set(LABEL_SOURCE_COLUMNS)))
    if leaked:
        raise LeakageError(f"Excluded/label columns in model features: {leaked}")


def event_bucket(event_type):
    """Code before the first '-', uppercased. None if there is no '-'."""
    if not isinstance(event_type, str) or "-" not in event_type:
        return None
    return event_type.split("-", 1)[0].strip().upper()


def harm_label(score):
    """A-D -> 0, E-I -> 1, blank -> UNLABELED."""
    if not isinstance(score, str) or not score.strip():
        return UNLABELED
    letter = score.strip()[0].upper()
    if letter in "ABCD":
        return 0
    if letter in "EFGHI":
        return 1
    raise DataContractError(f"Unexpected harm score: {score!r}")


def split_from_event_no(event_no):
    m = re.match(r"^SYNPROD-(TRAIN|VALIDATION|TEST)-\d+$", str(event_no))
    if not m:
        raise DataContractError(f"Event No. without a split prefix: {event_no!r}")
    return m.group(1)


def build_input_text(features):
    """'Unit: X. Service: Y. ...' prefix + narrative. Event type is never included."""
    assert_no_excluded_columns(features.columns)
    missing = set(FEATURE_COLUMNS) - set(features.columns)
    if missing:
        raise DataContractError(f"Feature columns missing: {sorted(missing)}")

    def fmt(v):
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v).strip()

    parts = []
    for label, cols in PREFIX_FIELDS:
        vals = features[list(cols)]
        joined = vals.apply(lambda r: " ".join(fmt(v) for v in r if pd.notna(v) and str(v).strip()), axis=1)
        parts.append(np.where(joined != "", label + ": " + joined + ". ", ""))
    prefix = pd.Series(["".join(p) for p in zip(*parts)], index=features.index)
    return (prefix.str.strip() + "\n" + features[TEXT_COL].astype(str)).str.lstrip()


def load_dataset(csv_path, nrows=None):
    """Load the CSV and return one row per report with labels and model input text.

    Columns: event_no, split, event_bucket, event_type_label, harm_label, text.
    Excluded columns are never read from disk.
    """
    usecols = [ID_COL, DATE_COL, EVENT_TYPE_COL, HARM_COL, *FEATURE_COLUMNS]
    raw = pd.read_csv(csv_path, usecols=usecols, nrows=nrows, low_memory=False)

    split = raw[ID_COL].map(split_from_event_no)
    month = pd.to_datetime(raw[DATE_COL], errors="raise").dt.month
    bad = [(s, m) for s, m in zip(split, month) if m not in SPLIT_MONTHS[s]]
    if bad:
        raise DataContractError(f"{len(bad)} rows where Event No. split disagrees with Event Date, e.g. {bad[:3]}")

    bucket = raw[EVENT_TYPE_COL].map(event_bucket)
    unknown = sorted(set(bucket.dropna()) - set(EVENT_BUCKETS))
    if unknown:
        raise DataContractError(f"Event-type codes outside the 11 buckets: {unknown}")

    features = raw[list(FEATURE_COLUMNS)]
    assert_no_excluded_columns(features.columns)

    out = pd.DataFrame({
        "event_no": raw[ID_COL],
        "split": split,
        "event_bucket": bucket,
        "event_type_label": bucket.map({b: i for i, b in enumerate(EVENT_BUCKETS)}).fillna(UNLABELED).astype(int),
        "harm_label": raw[HARM_COL].map(harm_label).astype(int),
        "text": build_input_text(features),
    })
    return out


def check_head_a_classes(df, name):
    """Head A sets must contain exactly the 11 buckets, no more, no fewer."""
    labels = df.loc[df.event_type_label != UNLABELED, "event_type_label"].unique()
    if len(labels) != N_EVENT_CLASSES or set(labels) != set(range(N_EVENT_CLASSES)):
        present = sorted(EVENT_BUCKETS[i] if 0 <= i < N_EVENT_CLASSES else f"<unknown id {i}>" for i in labels)
        raise DataContractError(f"{name}: expected exactly {N_EVENT_CLASSES} event-type labels, got {len(labels)}: {present}")


def check_hurt_prevalence(df):
    """Guardrail: a broken split shows up as wrong prevalence."""
    report = {}
    for s in SPLITS:
        y = df.loc[(df.split == s) & (df.harm_label != UNLABELED), "harm_label"]
        prev = float(y.mean())
        lo, hi = EXPECTED_HURT_PREVALENCE[s]
        if not lo <= prev <= hi:
            raise DataContractError(f"{s} hurt prevalence {prev:.4f} outside expected [{lo}, {hi}]")
        report[s] = prev
    return report


def validate_full_dataset(df):
    """All hard checks for the full 80k file. Returns a summary dict."""
    counts = df.split.value_counts().to_dict()
    if counts != {"TRAIN": 60000, "VALIDATION": 10000, "TEST": 10000}:
        raise DataContractError(f"Unexpected split sizes: {counts}")
    for s in SPLITS:
        check_head_a_classes(df[df.split == s], s)
    prevalence = check_hurt_prevalence(df)
    rare = {EVENT_BUCKETS[i]: int(n) for i, n in
            df[(df.split == "TRAIN") & (df.event_type_label >= 0)].event_type_label.value_counts().items() if n < 50}
    return {
        "split_sizes": counts,
        "hurt_prevalence": prevalence,
        "no_event_code_rows": int((df.event_type_label == UNLABELED).sum()),
        "no_harm_score_rows": int((df.harm_label == UNLABELED).sum()),
        "rare_train_buckets_lt50": rare,
    }


def smoke_slice(df, n=500, min_per_bucket=3, min_hurt=40, seed=0):
    """Small TRAIN slice with every bucket and enough hurt cases for the smoke test."""
    rng = np.random.RandomState(seed)
    tr = df[df.split == "TRAIN"]
    picks = set()
    for i in range(N_EVENT_CLASSES):
        idx = tr.index[tr.event_type_label == i]
        picks.update(rng.choice(idx, min(min_per_bucket, len(idx)), replace=False))
    hurt = tr.index[tr.harm_label == 1].difference(list(picks))
    picks.update(rng.choice(hurt, min_hurt, replace=False))
    picks.update(rng.choice(tr.index[tr.event_type_label == UNLABELED], 2, replace=False))
    rest = tr.index.difference(list(picks))
    picks.update(rng.choice(rest, n - len(picks), replace=False))
    return df.loc[sorted(picks)]
