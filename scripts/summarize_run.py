"""Print a human-readable summary of a MALACHY metrics CSV.

Usage:
    python scripts/summarize_run.py --metrics outputs/predictions/breast_tma_stripe_mlp_metrics.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize a MALACHY per-gene metrics CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--metrics", required=True,
                   help="Path to the metrics CSV (method, gene, mse, mae, pearson_r).")
    return p.parse_args()


def hr(char: str = "─", width: int = 62) -> str:
    return char * width


def print_method_block(name: str, df: pd.DataFrame) -> None:
    valid = df.dropna(subset=["pearson_r"])
    n_genes = len(df)
    n_valid = len(valid)

    print()
    print(f"  Method: {name}")
    print(hr())
    print(f"  Genes evaluated : {n_genes}  (Pearson computable: {n_valid})")
    print(f"  Mean  MSE       : {df['mse'].mean():.5f}")
    print(f"  Mean  MAE       : {df['mae'].mean():.5f}")
    print(f"  Mean  Pearson r : {valid['pearson_r'].mean():.4f}")
    print(f"  Median Pearson r: {valid['pearson_r'].median():.4f}")

    # Top 10
    top = valid.nlargest(10, "pearson_r")[["gene", "pearson_r", "mse"]]
    print()
    print(f"  Top-10 genes by Pearson r:")
    print(f"  {'Gene':<20}  {'Pearson r':>9}  {'MSE':>9}")
    print(f"  {hr('-', 42)}")
    for _, row in top.iterrows():
        print(f"  {row['gene']:<20}  {row['pearson_r']:>9.4f}  {row['mse']:>9.5f}")

    # Bottom 10
    bot = valid.nsmallest(10, "pearson_r")[["gene", "pearson_r", "mse"]]
    print()
    print(f"  Bottom-10 genes by Pearson r:")
    print(f"  {'Gene':<20}  {'Pearson r':>9}  {'MSE':>9}")
    print(f"  {hr('-', 42)}")
    for _, row in bot.iterrows():
        print(f"  {row['gene']:<20}  {row['pearson_r']:>9.4f}  {row['mse']:>9.5f}")


def print_comparison_table(df: pd.DataFrame, methods: list[str]) -> None:
    """Side-by-side mean metrics across methods."""
    print()
    print(hr("═"))
    print("  Method comparison (mean across genes)")
    print(hr())
    header = f"  {'Method':<16}  {'Mean MSE':>10}  {'Mean MAE':>10}  {'Mean r':>8}  {'Med r':>8}"
    print(header)
    print(f"  {hr('-', len(header) - 2)}")
    for m in methods:
        sub = df[df["method"] == m].dropna(subset=["pearson_r"])
        print(
            f"  {m:<16}  "
            f"{df[df['method']==m]['mse'].mean():>10.5f}  "
            f"{df[df['method']==m]['mae'].mean():>10.5f}  "
            f"{sub['pearson_r'].mean():>8.4f}  "
            f"{sub['pearson_r'].median():>8.4f}"
        )
    print(hr("═"))


def main():
    args = parse_args()
    path = Path(args.metrics)
    if not path.exists():
        sys.exit(f"File not found: {path}")

    df = pd.read_csv(path)
    required = {"method", "gene", "mse", "mae", "pearson_r"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Metrics CSV is missing columns: {missing}")

    methods = list(df["method"].unique())

    print()
    print(hr("═"))
    print(f"  MALACHY run summary")
    print(f"  File   : {path}")
    print(f"  Genes  : {df['gene'].nunique()}")
    print(f"  Methods: {', '.join(methods)}")
    print(hr("═"))

    for m in methods:
        print_method_block(m, df[df["method"] == m].copy())

    print_comparison_table(df, methods)
    print()


if __name__ == "__main__":
    main()
