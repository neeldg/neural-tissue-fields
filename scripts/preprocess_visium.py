"""Preprocess a single 10x Visium section for the neural field pipeline.

Reads a Space Ranger output directory, normalizes expression, selects a gene
panel, and writes a flat parquet/CSV file compatible with train_v0.py.

Usage — HVG selection (default):
    python scripts/preprocess_visium.py \\
        --visium-dir /path/to/spaceranger/outs/ \\
        --section-id patient01_A1 \\
        --z 0 \\
        --n-top-genes 50

Usage — explicit gene list (comma-separated):
    python scripts/preprocess_visium.py \\
        --visium-dir /path/to/spaceranger/outs/ \\
        --section-id patient01_A1 \\
        --gene-list MALAT1,PTPRC,COL1A1,EPCAM

Usage — explicit gene list (from file, one gene per line):
    python scripts/preprocess_visium.py \\
        --visium-dir /path/to/spaceranger/outs/ \\
        --section-id patient01_A1 \\
        --gene-list-file configs/gene_panel.txt

The output file is written to:
    {output_dir}/{section_id}_visium.parquet   (or .csv if pyarrow is missing)

To combine multiple sections for training, run this script once per section
then concatenate the output files before running train_v0.py.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.load_visium import load_visium_section, load_scalefactors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess a 10x Visium section into a flat spot table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument(
        "--visium-dir", required=True,
        help="Path to the Space Ranger output directory (contains "
             "filtered_feature_bc_matrix.h5 and spatial/).",
    )
    p.add_argument(
        "--section-id", required=True,
        help="Unique identifier for this section (e.g. 'patient01_A1'). "
             "Used as the section_id column value and in the output filename.",
    )

    # Spatial depth
    p.add_argument(
        "--z", type=float, default=0.0,
        help="Z coordinate to assign to this section.  Use the section index "
             "(0, 1, 2, …) when combining multiple sections later.",
    )

    # Gene selection (mutually exclusive)
    gene_group = p.add_mutually_exclusive_group()
    gene_group.add_argument(
        "--n-top-genes", type=int, default=50,
        help="Number of highly variable genes to select (ignored if "
             "--gene-list or --gene-list-file is given).",
    )
    gene_group.add_argument(
        "--gene-list",
        help="Comma-separated gene symbols to include, e.g. MALAT1,PTPRC,COL1A1.",
    )
    gene_group.add_argument(
        "--gene-list-file",
        help="Text file with one gene symbol per line.",
    )

    # Coordinate type
    p.add_argument(
        "--coord-type", choices=["pixel", "array"], default="pixel",
        help="'pixel': full-resolution pixel positions (pxl_col/pxl_row). "
             "'array': Visium hexagonal grid indices (array_col/array_row).",
    )

    # Quality filter
    p.add_argument(
        "--min-counts", type=int, default=100,
        help="Minimum total UMI counts per spot; lower-count spots are dropped.",
    )

    # Output
    p.add_argument(
        "--output-dir", default="data/processed",
        help="Directory to write the processed spot table.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Gene list helpers
# ---------------------------------------------------------------------------

def resolve_gene_list(args: argparse.Namespace) -> list[str] | None:
    """Return the explicit gene list from CLI args, or None to trigger HVG selection."""
    if args.gene_list:
        genes = [g.strip() for g in args.gene_list.split(",") if g.strip()]
        log.info("Using %d genes from --gene-list.", len(genes))
        return genes

    if args.gene_list_file:
        path = Path(args.gene_list_file)
        if not path.exists():
            sys.exit(f"Gene list file not found: {path}")
        genes = [line.strip() for line in path.read_text().splitlines()
                 if line.strip() and not line.startswith("#")]
        log.info("Loaded %d genes from %s.", len(genes), path)
        return genes

    return None   # use HVG selection


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_output(df, output_dir: Path, section_id: str) -> Path:
    """Save to parquet; fall back to CSV if pyarrow is not installed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{section_id}_visium.parquet"

    try:
        df.to_parquet(parquet_path, index=False)
        log.info("Saved → %s", parquet_path)
        return parquet_path
    except ImportError:
        csv_path = output_dir / f"{section_id}_visium.csv"
        df.to_csv(csv_path, index=False)
        log.info(
            "pyarrow not installed; saved as CSV → %s", csv_path
        )
        return csv_path


def print_summary(df, out_path: Path, section_id: str, coord_type: str) -> None:
    """Print a concise summary of the preprocessed section."""
    gene_cols = [c for c in df.columns
                 if c not in ("section_id", "x", "y", "z")]

    print("\n" + "─" * 56)
    print(f"  Section  : {section_id}")
    print(f"  Spots    : {len(df):,}")
    print(f"  Genes    : {len(gene_cols)}")
    print(f"  Coords   : {coord_type}  "
          f"x=[{df['x'].min():.1f}, {df['x'].max():.1f}]  "
          f"y=[{df['y'].min():.1f}, {df['y'].max():.1f}]  "
          f"z={df['z'].iloc[0]:.1f}")

    expr = df[gene_cols]
    print(f"  Expression (log-norm):")
    print(f"    mean   {expr.values.mean():.4f}")
    print(f"    median {expr.stack().median():.4f}")
    print(f"    max    {expr.values.max():.4f}")
    print(f"  Genes    : {', '.join(gene_cols[:8])}"
          + (" …" if len(gene_cols) > 8 else ""))
    print(f"  Output   : {out_path}")
    print("─" * 56 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    visium_dir = Path(args.visium_dir)
    if not visium_dir.is_dir():
        sys.exit(f"Visium directory not found: {visium_dir}")

    gene_list = resolve_gene_list(args)

    # Report scale factors if available (informational only).
    sf = load_scalefactors(visium_dir)
    if sf:
        log.info(
            "Scale factors: spot_diameter=%.2f  hires=%.4f  lowres=%.4f",
            sf.get("spot_diameter_fullres", float("nan")),
            sf.get("tissue_hires_scalef", float("nan")),
            sf.get("tissue_lowres_scalef", float("nan")),
        )

    df = load_visium_section(
        visium_dir=visium_dir,
        section_id=args.section_id,
        z=args.z,
        n_top_genes=args.n_top_genes,
        gene_list=gene_list,
        coord_type=args.coord_type,
        min_counts=args.min_counts,
    )

    out_path = save_output(df, Path(args.output_dir), args.section_id)
    print_summary(df, out_path, args.section_id, args.coord_type)


if __name__ == "__main__":
    main()
