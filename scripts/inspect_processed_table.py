"""Print a concise summary of any processed spot table.

Works for single-section files produced by preprocess_visium.py and
combined multi-section files produced by combine_sections.py.

Usage:
    python scripts/inspect_processed_table.py --input data/processed/synthetic_sections.csv
    python scripts/inspect_processed_table.py --input data/processed/combined_visium_sections.parquet
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

META_COLS = {"section_id", "x", "y", "z"}


def load(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix in (".csv", ".tsv"):
        return pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    else:
        sys.exit(f"Unsupported file format: {path.suffix}. Expected .parquet or .csv.")


def hr(char: str = "─", width: int = 58) -> str:
    return char * width


def main():
    p = argparse.ArgumentParser(
        description="Inspect a processed spatial transcriptomics spot table."
    )
    p.add_argument("--input", required=True, help="Path to .parquet or .csv file.")
    args = p.parse_args()

    path = Path(args.input)
    if not path.exists():
        sys.exit(f"File not found: {path}")

    df = load(path)

    gene_cols = [c for c in df.columns if c not in META_COLS]
    has_section = "section_id" in df.columns

    print()
    print(hr("═"))
    print(f"  File    : {path}")
    print(hr())

    # ── Shape & columns ───────────────────────────────────────────────────
    print(f"  Shape   : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns : {list(df.columns)}")
    print()

    # ── Sections ──────────────────────────────────────────────────────────
    if has_section:
        section_summary = (
            df.groupby("section_id", sort=False)
            .agg(n_spots=("x", "count"), z=("z", "first"))
            .reset_index()
        )
        print(f"  Sections: {len(section_summary)}")
        print()
        col_w = max(len(s) for s in section_summary["section_id"]) + 2
        header = f"  {'section_id':<{col_w}}  {'z':>6}  {'spots':>7}"
        print(header)
        print("  " + hr("-", len(header) - 2))
        for _, row in section_summary.iterrows():
            print(
                f"  {row['section_id']:<{col_w}}  {row['z']:>6.2f}  {row['n_spots']:>7,}"
            )
        print()
    else:
        print("  (no section_id column found)")
        print()

    # ── Genes ─────────────────────────────────────────────────────────────
    print(f"  Gene columns : {len(gene_cols)}")
    if gene_cols:
        preview = gene_cols[:8]
        suffix = " …" if len(gene_cols) > 8 else ""
        print(f"  Gene names   : {', '.join(preview)}{suffix}")
    print()

    # ── Coordinate ranges ─────────────────────────────────────────────────
    coord_rows = []
    for col in ["x", "y", "z"]:
        if col in df.columns:
            coord_rows.append(
                f"  {col} : [{df[col].min():.4f}, {df[col].max():.4f}]"
            )
    print("\n".join(coord_rows))
    print()

    # ── Expression summary ────────────────────────────────────────────────
    if gene_cols:
        expr = df[gene_cols]
        print(f"  Expression (all genes):")
        print(f"    min    {expr.values.min():.4f}")
        print(f"    mean   {expr.values.mean():.4f}")
        print(f"    median {pd.Series(expr.values.ravel()).median():.4f}")
        print(f"    max    {expr.values.max():.4f}")
        print()

    # ── First 5 rows ──────────────────────────────────────────────────────
    print(hr())
    print("  First 5 rows:")
    print()
    preview_cols = list(META_COLS & set(df.columns)) + gene_cols[:4]
    preview_cols = [c for c in df.columns if c in preview_cols]  # preserve order
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 120,
        "display.float_format", "{:.4f}".format,
    ):
        for line in df[preview_cols].head(5).to_string(index=False).splitlines():
            print("  " + line)
    if len(gene_cols) > 4:
        print(f"  … ({len(gene_cols) - 4} more gene columns not shown)")
    print()
    print(hr("═"))
    print()


if __name__ == "__main__":
    main()
