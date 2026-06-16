"""Print a per-method summary of held-out metrics from an all_metrics CSV.

Usage:
    python scripts/summarize_metrics.py
    python scripts/summarize_metrics.py --metrics outputs/predictions/my_run_all_metrics.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

DEFAULT_METRICS = "outputs/predictions/synthetic_v0_all_metrics.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Summarize per-method metrics from an all_metrics CSV."
    )
    parser.add_argument(
        "--metrics",
        default=DEFAULT_METRICS,
        help=f"Path to the all_metrics CSV (default: {DEFAULT_METRICS})",
    )
    args = parser.parse_args()

    path = Path(args.metrics)
    if not path.exists():
        sys.exit(f"File not found: {path}")

    df = pd.read_csv(path)

    summary = (
        df.groupby("method")
        .agg(
            mean_mse=("mse", "mean"),
            mean_mae=("mae", "mean"),
            mean_pearson_r=("pearson_r", "mean"),
            median_pearson_r=("pearson_r", "median"),
        )
        .sort_values("mean_pearson_r", ascending=False)
    )

    print(f"\nMetrics file: {path}")
    print(f"Genes per method: {df.groupby('method')['gene'].count().iloc[0]}\n")
    print(
        summary.to_string(
            float_format="{:.4f}".format,
            index_names=True,
        )
    )
    print()


if __name__ == "__main__":
    main()
