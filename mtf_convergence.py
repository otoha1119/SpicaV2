#ROI収束閾値識別用スクリプト
#実行コマンド
"""
python3 mtf_convergence.py \
  --results_dir /workspace/results \
  --series LR SR HR \
  --metrics mtf50 mtf10 auc \
  --rel_thresh 0.05 \
  --abs_thresh 0.02 \
  --boots 2000

"""

#!/usr/bin/env python3
"""
mtf_convergence.py
------------------
Estimate how many ROIs you need until summary metrics (MTF50 / MTF10 / AUC)
stabilize, using bootstrap 95% confidence intervals as a function of ROI count N.

It reads the ROI metrics CSVs produced by your pipeline, e.g.:
  - results/lr_roi_metrics.csv
  - results/sr_roi_metrics.csv
  - results/hr_roi_metrics.csv

Outputs:
  - <results_dir>/convergence_<series>_<metric>.png    (CI width vs N plot)
  - <results_dir>/mtf_convergence_summary.csv          (table of N* for each series/metric)
  - Console table preview

Usage example:
  python3 mtf_convergence.py \
    --results_dir /workspace/results \
    --series LR SR HR \
    --metrics mtf50 mtf10 auc \
    --rel_thresh 0.05 --abs_thresh 0.02 \
    --boots 2000

Notes:
- Uses bootstrap over means with replacement. For each target N we sample N items
  from the full set (with replacement), repeat `--boots` times, and compute 95% CI of the means.
- We define convergence if either (CI_width / mean) <= rel_thresh  OR CI_width <= abs_thresh.
- Absolute-width thresholds are most interpretable for MTF fractions (0..1). Adjust as you like.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------- Bootstrap helper -----------------------------
def bootstrap_ci_of_mean(data: np.ndarray, sample_size: int, n_boot: int = 2000, ci: float = 95.0, rng: np.random.Generator | None = None):
    """
    Bootstrap the mean for a given sample size 'sample_size' by resampling with replacement
    from the full array 'data'. Returns (mean_estimate, ci_lower, ci_upper).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    # Handle edge cases
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    if data.size == 0:
        return np.nan, np.nan, np.nan

    means = np.empty(n_boot, dtype=float)
    n = data.size
    for b in range(n_boot):
        idx = rng.integers(0, n, size=sample_size, endpoint=False)
        means[b] = data[idx].mean()

    lower = np.percentile(means, (100.0 - ci) / 2.0)
    upper = np.percentile(means, 100.0 - (100.0 - ci) / 2.0)
    mean_est = means.mean()
    return mean_est, float(lower), float(upper)

# ----------------------------- Main computation -----------------------------
def analyze_series(file_path: Path, metrics: list[str], Ns: list[int], n_boot: int, ci: float, rel_thresh: float, abs_thresh: float, out_dir: Path, series_label: str):
    """
    For a single series CSV, compute CI width vs N for specified metrics.
    Returns a DataFrame with rows for each N and metric.
    """
    df = pd.read_csv(file_path)
    out_rows = []
    rng = np.random.default_rng(12345)

    # Sanity: available count
    n_avail = len(df)
    Ns = [int(n) for n in Ns if n > 0]

    # Build plots per metric
    for metric in metrics:
        if metric not in df.columns:
            print(f"[WARN] metric '{metric}' not found in {file_path.name} (available: {list(df.columns)})")
            continue

        data = df[metric].to_numpy(dtype=float)
        data = data[~np.isnan(data)]
        if data.size == 0:
            print(f"[WARN] metric '{metric}' has no valid (non-NaN) data in {file_path.name}")
            continue

        xs, rel_w, abs_w = [], [], []
        best_N = None

        for N in Ns:
            mean_est, lo, hi = bootstrap_ci_of_mean(data, sample_size=N, n_boot=n_boot, ci=ci, rng=rng)
            width = hi - lo
            rel = (width / abs(mean_est)) if (mean_est is not None and not np.isnan(mean_est) and mean_est != 0.0) else np.nan

            out_rows.append({
                "series": series_label,
                "metric": metric,
                "N": N,
                "mean": mean_est,
                "ci_lower": lo,
                "ci_upper": hi,
                "ci_width": width,
                "rel_width": rel,
                "meets_rel": (rel_thresh is not None and np.isfinite(rel) and rel <= rel_thresh),
                "meets_abs": (abs_thresh is not None and width <= abs_thresh),
            })

            xs.append(N)
            rel_w.append(rel)
            abs_w.append(width)

            # First N that satisfies either threshold => convergence point
            if best_N is None:
                ok_rel = (rel_thresh is not None and np.isfinite(rel) and rel <= rel_thresh)
                ok_abs = (abs_thresh is not None and width <= abs_thresh)
                if ok_rel or ok_abs:
                    best_N = N

        # Plot relative CI width vs N
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, rel_w, marker="o", linewidth=1.5)
        if rel_thresh is not None:
            ax.axhline(rel_thresh, linestyle="--", linewidth=1.0)
        ax.set_xlabel("Number of ROIs (N)")
        ax.set_ylabel("Relative CI width (95%)")
        ax.set_title(f"Convergence: {series_label} / {metric}")
        ax.grid(True, alpha=0.3)
        if best_N is not None and np.all(np.isfinite(rel_w)):
            y_at = np.interp(best_N, xs, rel_w)
            ax.annotate(f"N* = {best_N}", xy=(best_N, y_at),
                        xytext=(best_N, max(rel_w)*0.9 if len(rel_w) else y_at),
                        arrowprops=dict(arrowstyle="->"))
        fig.tight_layout()
        fig.savefig(out_dir / f"convergence_{series_label.lower()}_{metric}.png", dpi=200)
        plt.close(fig)

    return pd.DataFrame(out_rows)

def build_N_grid(max_available: int, user_max: int | None):
    # Reasonable exponentially-spaced grid up to max
    maxN = min(max_available, user_max) if user_max is not None else max_available
    grid = sorted(set([25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000, 3000]))
    grid = [n for n in grid if n <= maxN]
    if maxN not in grid and maxN > 0:
        grid.append(maxN)
    return grid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="./results", help="Directory containing *_roi_metrics.csv files")
    ap.add_argument("--series", type=str, nargs="+", default=["LR","SR","HR"], help="Which series to analyze (labels must match file prefixes)")
    ap.add_argument("--metrics", type=str, nargs="+", default=["mtf50","mtf10","auc"], help="Which metrics to analyze")
    ap.add_argument("--boots", type=int, default=2000, help="Bootstrap repeats per (N, metric)")
    ap.add_argument("--ci", type=float, default=95.0, help="Confidence level, e.g., 95")
    ap.add_argument("--rel_thresh", type=float, default=0.05, help="Relative CI width threshold (e.g., 0.05 = 5%). Set -1 to disable")
    ap.add_argument("--abs_thresh", type=float, default=0.02, help="Absolute CI width threshold. Set -1 to disable")
    ap.add_argument("--max_n", type=int, default=None, help="Optional cap on N for the grid")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rel_thresh = None if (args.rel_thresh is not None and args.rel_thresh < 0) else args.rel_thresh
    abs_thresh = None if (args.abs_thresh is not None and args.abs_thresh < 0) else args.abs_thresh

    # Determine max available N per series by reading the CSV lengths
    lengths = {}
    for s in args.series:
        fp = results_dir / f"{s.lower()}_roi_metrics.csv"
        if not fp.exists():
            print(f"[WARN] Missing file: {fp}")
            continue
        try:
            n_rows = len(pd.read_csv(fp))
            lengths[s] = n_rows
        except Exception:
            print(f"[WARN] Could not read {fp}")
            lengths[s] = 0

    if not lengths:
        print("[WARN] No valid series files found. Nothing to do.")
        return

    max_available = max(lengths.values())
    Ns = build_N_grid(max_available=max_available, user_max=args.max_n)

    all_rows = []
    for s in args.series:
        fp = results_dir / f"{s.lower()}_roi_metrics.csv"
        if not fp.exists():
            print(f"[WARN] Skip {s}: file not found {fp}")
            continue
        df_series = analyze_series(
            file_path=fp,
            metrics=args.metrics,
            Ns=Ns,
            n_boot=args.boots,
            ci=args.ci,
            rel_thresh=rel_thresh,
            abs_thresh=abs_thresh,
            out_dir=out_dir,
            series_label=s
        )
        if not df_series.empty:
            all_rows.append(df_series)

    if all_rows:
        summary = pd.concat(all_rows, ignore_index=True)
        # Compute first N meeting thresholds per (series, metric)
        def first_ok(group: pd.DataFrame):
            ok = group[(group["meets_rel"]) | (group["meets_abs"])]
            return pd.Series({
                "N_star": int(ok["N"].iloc[0]) if not ok.empty else np.nan,
                "mean_at_N_star": float(ok["mean"].iloc[0]) if not ok.empty else np.nan,
                "CI_width_at_N_star": float(ok["ci_width"].iloc[0]) if not ok.empty else np.nan,
                "Rel_width_at_N_star": float(ok["rel_width"].iloc[0]) if not ok.empty else np.nan
            })

        summary_sorted = summary.sort_values(["series","metric","N"])
        report = summary_sorted.groupby(["series","metric"], as_index=False).apply(first_ok)
        if isinstance(report.columns, pd.MultiIndex):
            report.columns = ['_'.join([str(c) for c in col if c != '']) for col in report.columns.values]

        out_csv = out_dir / "mtf_convergence_summary.csv"
        report.to_csv(out_csv, index=False)
        print("\n=== Convergence summary (first N meeting threshold) ===")
        print(report.to_string(index=False))
        print(f"\nSaved summary: {out_csv}")
        print(f"Saved plots under: {out_dir}")
    else:
        print("[WARN] No data to summarize.")

if __name__ == "__main__":
    main()
