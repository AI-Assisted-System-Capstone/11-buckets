"""Step 1: build the processed dataset and run every data-contract check.

Usage: python src/prepare_data.py
Writes outputs/processed/dataset.parquet and outputs/processed/data_checks.json.
"""
import json
import os
import sys

from pipeline import CSV_NAME, EVENT_BUCKETS, SPLITS, UNLABELED, load_dataset, validate_full_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "processed")


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load_dataset(os.path.join(ROOT, CSV_NAME))
    checks = validate_full_dataset(df)

    counts = {}
    for s in SPLITS:
        d = df[df.split == s]
        counts[s] = {
            "rows": len(d),
            "head_a_rows": int((d.event_type_label != UNLABELED).sum()),
            "head_b_rows": int((d.harm_label != UNLABELED).sum()),
            "hurt": int((d.harm_label == 1).sum()),
            "buckets": {b: int((d.event_bucket == b).sum()) for b in EVENT_BUCKETS},
        }
    checks["per_split"] = counts
    df.to_parquet(os.path.join(OUT, "dataset.parquet"), index=False)
    json.dump(checks, open(os.path.join(OUT, "data_checks.json"), "w"), indent=2)
    print(json.dumps(checks, indent=2))
    print(f"\nExample model input:\n{df.text.iloc[0][:400]}")
    print("\nAll data-contract checks passed.")


if __name__ == "__main__":
    sys.exit(main())
