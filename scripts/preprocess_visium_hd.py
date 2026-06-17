"""Preprocess a single 10x Visium HD section for the MALACHY pipeline.

Reads one bin level from a Space Ranger HD binned_outputs/ directory,
normalizes expression, selects a gene panel, and writes a flat table
compatible with train_v0.py and combine_sections.py.

Usage — HVG selection (default):
    python scripts/preprocess_visium_hd.py \\
        --binned-dir /path/to/spaceranger_hd/outs/binned_outputs \\
        --bin-level square_016um \\
        --section-id patient01_A1 \\
        --z 0

Usage — explicit gene list (comma-separated):
    python scripts/preprocess_visium_hd.py \\
        --binned-dir /path/to/spaceranger_hd/outs/binned_outputs \\
        --bin-level square_008um \\
        --section-id patient01_A1 \\
        --gene-list MALAT1,PTPRC,COL1A1,EPCAM

Usage — explicit gene list (from file, one gene per line):
    python scripts/preprocess_visium_hd.py \\
        --binned-dir /path/to/spaceranger_hd/outs/binned_outputs \\
        --bin-level square_016um \\
        --section-id patient01_A1 \\
        --gene-list-file configs/gene_panel.txt

Output path default:
    data/processed/{section_id}_{bin_level}_visium_hd.parquet
    (falls back to .csv if pyarrow is not installed)
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.load_visium_hd import KNOWN_BIN_LEVELS, load_visium_hd_section

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

META_COLS = {"section_id", "x", "y", "z"}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess a 10x Visium HD bin level into a flat spot table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument(
        "--binned-dir", required=True,
        help="Path to the Space Ranger HD binned_outputs/ directory. "
             f"Expected subdirectories: {KNOWN_BIN_LEVELS}.",
    )
    p.add_argument(
        "--bin-level", required=True,
        help="Bin-level folder to process, e.g. 'square_016um' or 'square_008um'.",
    )
    p.add_argument(
        "--section-id", required=True,
        help="Unique identifier for this section (e.g. 'patient01_A1'). "
             "Written to the section_id column and used in the output filename.",
    )

    # Spatial depth
    p.add_argument(
        "--z", type=float, default=0.0,
        help="Z coordinate to assign. Use the section index (0, 1, 2, …) "
             "when combining multiple sections later.",
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
        help="Text file with one gene symbol per line (lines starting with # ignored).",
    )

    # Quality filter
    p.add_argument(
        "--min-counts", type=int, default=10,
        help="Minimum total UMI counts per bin.  HD bins are small (8–16 µm) "
             "and sparse, so this is lower than the standard Visium default.",
    )

    # Output
    p.add_argument(
        "--output",
        help="Full output path for the processed table (.parquet or .csv). "
             "Default: data/processed/{section_id}_{bin_level}_visium_hd.parquet",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Gene list resolution (identical pattern to preprocess_visium.py)
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
        genes = [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        log.info("Loaded %d genes from %s.", len(genes), path)
        return genes

    return None


# ---------------------------------------------------------------------------
# Output path resolution
# ---------------------------------------------------------------------------

def resolve_output_path(args: argparse.Namespace) -> Path:
    """Return the output path, using the default naming scheme if not specified."""
    if args.output:
        return Path(args.output)
    return Path("data/processed") / f"{args.section_id}_{args.bin_level}_visium_hd.parquet"


# ---------------------------------------------------------------------------
# Save and summary
# ---------------------------------------------------------------------------

def save_output(df, path: Path) -> Path:
    """Write to parquet; fall back to CSV if pyarrow is not installed."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            log.info("Saved → %s", path)
            return path
        except ImportError:
            path = path.with_suffix(".csv")
            log.warning("pyarrow not installed; saving as CSV instead.")

    df.to_csv(path, index=False)
    log.info("Saved → %s", path)
    return path


def print_summary(df, out_path: Path, section_id: str, bin_level: str) -> None:
    gene_cols = [c for c in df.columns if c not in META_COLS]
    expr = df[gene_cols]

    print("\n" + "─" * 58)
    print(f"  Section  : {section_id}")
    print(f"  Bin level: {bin_level}")
    print(f"  Bins     : {len(df):,}")
    print(f"  Genes    : {len(gene_cols)}")
    print(
        f"  Coords   : pixel  "
        f"x=[{df['x'].min():.0f}, {df['x'].max():.0f}]  "
        f"y=[{df['y'].min():.0f}, {df['y'].max():.0f}]  "
        f"z={df['z'].iloc[0]:.1f}"
    )
    print(f"  Expression (log-norm):")
    print(f"    mean   {expr.values.mean():.4f}")
    print(f"    median {expr.stack().median():.4f}")
    print(f"    max    {expr.values.max():.4f}")
    print(f"  Genes    : {', '.join(gene_cols[:8])}"
          + (" …" if len(gene_cols) > 8 else ""))
    print(f"  Output   : {out_path}")
    print("─" * 58 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    binned_dir = Path(args.binned_dir)
    if not binned_dir.is_dir():
        sys.exit(f"binned_outputs directory not found: {binned_dir}")

    gene_list = resolve_gene_list(args)
    out_path = resolve_output_path(args)

    df = load_visium_hd_section(
        binned_dir=binned_dir,
        bin_level=args.bin_level,
        section_id=args.section_id,
        z=args.z,
        n_top_genes=args.n_top_genes,
        gene_list=gene_list,
        min_counts=args.min_counts,
    )

    out_path = save_output(df, out_path)
    print_summary(df, out_path, args.section_id, args.bin_level)


if __name__ == "__main__":
    main()
