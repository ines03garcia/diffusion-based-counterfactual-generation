#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf", required=True, help="CSV with CF results")
    ap.add_argument("--no_cf", required=True, help="CSV with no-CF results")
    ap.add_argument("--key", default="img_path", help="Column used to pair rows (default: img_path)")
    ap.add_argument(
        "--cols",
        nargs="*",
        default=None,
        help="Metric columns to test (default: all columns starting with 'iou_score_')",
    )
    ap.add_argument("--out", default=None, help="Optional output CSV for p-values table")
    args = ap.parse_args()

    df_cf = pd.read_csv(args.cf)
    df_no = pd.read_csv(args.no_cf)

    key = args.key
    if key not in df_cf.columns or key not in df_no.columns:
        raise ValueError(f"Key column '{key}' must exist in both files.")

    # Choose metric columns
    if args.cols is None:
        metric_cols = [c for c in df_cf.columns if c.startswith("iou_score_")]
    else:
        metric_cols = args.cols

    missing_cf = [c for c in metric_cols if c not in df_cf.columns]
    missing_no = [c for c in metric_cols if c not in df_no.columns]
    if missing_cf or missing_no:
        raise ValueError(f"Missing columns. In CF missing: {missing_cf}. In no-CF missing: {missing_no}.")

    # Pair rows by key
    merged = df_cf[[key] + metric_cols].merge(
        df_no[[key] + metric_cols],
        on=key,
        suffixes=("_cf", "_no_cf"),
        how="inner",
    )

    if len(merged) == 0:
        raise ValueError("No paired rows found after merging. Check that keys match across files.")

    results = []
    for col in metric_cols:
        x = merged[f"{col}_cf"].to_numpy(dtype=float)
        y = merged[f"{col}_no_cf"].to_numpy(dtype=float)
        diff = x - y

        n_pairs = diff.size
        n_nonzero = int(np.count_nonzero(diff))

        # Pratt is a good default when many paired differences are exactly zero
        stat, p = wilcoxon(
            x, y,
            zero_method="pratt",
            alternative="two-sided",
            mode="auto"
        )

        results.append({
            "metric": col,
            "n_pairs": int(n_pairs),
            "n_nonzero_diffs": n_nonzero,
            "wilcoxon_statistic": float(stat),
            "p_value_two_sided": float(p),
            "median_cf": float(np.median(x)),
            "median_no_cf": float(np.median(y)),
            "median_diff_cf_minus_no_cf": float(np.median(diff)),
            "mean_diff_cf_minus_no_cf": float(np.mean(diff)),
        })

    out_df = pd.DataFrame(results).sort_values("p_value_two_sided")
    out_df["p_fdr_bh"] = multipletests(out_df["p_value_two_sided"], method="fdr_bh")[1]
    print(out_df.to_string(index=False))

    if args.out:
        out_df.to_csv(args.out, index=False)
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()