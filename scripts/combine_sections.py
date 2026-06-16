"""Combine multiple preprocessed Visium section files into one training table.

Each input file is produced by scripts/preprocess_visium.py and contains
columns: section_id, x, y, z, <gene columns>.

Only genes present in ALL sections are kept (inner join on gene columns).
The section_id and z columns in each file are overridden by the values
supplied on the command line, so the caller has full control over naming
and depth ordering.

Usage — z positions inferred from file order (0, 1, 2, …):
    python scripts/combine_sections.py \\
        --sections data/processed/A1_visium.parquet \\
                   data/processed/A2_visium.parquet \\
                   data/processed/A3_visium.parquet \\
        --section-ids patient01_A1 patient01_A2 patient01_A3

Usage — explicit z positions:
    python scripts/combine_sections.py \\
        --sections data/processed/A1_visium.parquet \\
                   data/processed/A2_visium.parquet \\
        --section-ids patient01_A1 patient01_A2 \\
        --z-values 0.0 1.5

Usage — keep only a specific gene panel:
    python scripts/combine_sections.py \\
        --sections data/processed/A1_visium.parquet \\
                   data/processed/A2_visium.parquet \\
        --section-ids patient01_A1 patient01_A2 \\
        --gene-list MALAT1,PTPRC,COL1A1

Output is written to data/processed/combined_visium_sections.parquet
(or .csv if pyarrow is not installed).  Override with --output.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

META_COLS = {"section_id", "x", "y", "z"}
DEFAULT_OUTPUT = "data/processed/combined_visium_sections.parquet"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_section_file(path: Path) -> pd.DataFrame:
    """Read a parquet or CSV section file."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix in (".csv", ".tsv"):
        sep = "\t" if path.suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    else:
        raise ValueError(
            f"Unsupported file format: {path.suffix}. Expected .parquet or .csv."
        )


def save_output(df: pd.DataFrame, path: Path) -> Path:
    """Write to parquet; fall back to CSV if pyarrow is not installed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except ImportError:
            path = path.with_suffix(".csv")
            log.warning("pyarrow not installed; saving as CSV instead.")
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def gene_cols_of(df: pd.DataFrame) -> list[str]:
    """Return the gene expression columns (everything except metadata columns)."""
    return [c for c in df.columns if c not in META_COLS]


def find_shared_genes(
    frames: list[pd.DataFrame],
    section_ids: list[str],
) -> list[str]:
    """Return sorted list of gene columns present in every section."""
    gene_sets = [set(gene_cols_of(df)) for df in frames]
    shared = sorted(set.intersection(*gene_sets))

    for i, (sid, df) in enumerate(zip(section_ids, frames)):
        n_genes = len(gene_cols_of(df))
        n_shared = len(shared)
        if n_genes != n_shared:
            only_here = set(gene_cols_of(df)) - set(shared)
            log.warning(
                "Section '%s': %d gene(s) not shared across all sections "
                "and will be dropped: %s%s",
                sid,
                len(only_here),
                sorted(only_here)[:6],
                " …" if len(only_here) > 6 else "",
            )

    if not shared:
        raise ValueError(
            "No genes are shared across all sections. "
            "Check that sections were preprocessed with the same gene panel, "
            "or use --gene-list to specify a common panel."
        )

    log.info("Shared genes across %d sections: %d", len(frames), len(shared))
    return shared


def filter_to_gene_list(
    shared_genes: list[str],
    gene_list: list[str],
) -> list[str]:
    """Intersect shared genes with a user-supplied panel, preserving panel order."""
    shared_set = set(shared_genes)
    missing = [g for g in gene_list if g not in shared_set]
    if missing:
        log.warning(
            "%d gene(s) from --gene-list not found in all sections (skipped): %s%s",
            len(missing),
            missing[:8],
            " …" if len(missing) > 8 else "",
        )
    kept = [g for g in gene_list if g in shared_set]
    if not kept:
        raise ValueError(
            "None of the genes in --gene-list are shared across all sections."
        )
    log.info("Using %d / %d genes from --gene-list.", len(kept), len(gene_list))
    return kept


def combine(
    section_files: list[Path],
    section_ids: list[str],
    z_values: list[float],
    gene_list: list[str] | None = None,
) -> pd.DataFrame:
    """Load, align, and concatenate section DataFrames.

    Args:
        section_files: Paths to the preprocessed section files.
        section_ids:   Replacement section identifiers (one per file).
        z_values:      Z coordinate for each section (one per file).
        gene_list:     Optional explicit gene panel; must be a subset of the
                       shared genes across all sections.

    Returns:
        Combined DataFrame with columns: section_id, x, y, z, <gene columns>.
    """
    # ── Load ──────────────────────────────────────────────────────────────
    frames: list[pd.DataFrame] = []
    for path, sid in zip(section_files, section_ids):
        log.info("Loading  %s  (section_id='%s')", path.name, sid)
        df = load_section_file(path)

        missing_meta = META_COLS - set(df.columns)
        if missing_meta:
            raise ValueError(
                f"File {path.name} is missing required columns: {missing_meta}. "
                "Is it a valid preprocess_visium.py output?"
            )

        frames.append(df)
        log.info("  → %d spots, %d genes", len(df), len(gene_cols_of(df)))

    # ── Gene intersection ──────────────────────────────────────────────────
    shared_genes = find_shared_genes(frames, section_ids)
    if gene_list is not None:
        shared_genes = filter_to_gene_list(shared_genes, gene_list)

    # ── Override section_id and z, then concatenate ────────────────────────
    output_frames: list[pd.DataFrame] = []
    for df, sid, z in zip(frames, section_ids, z_values):
        out = df[["x", "y"] + shared_genes].copy()
        out.insert(0, "section_id", sid)
        out.insert(3, "z", float(z))
        # x and y are kept from the file (caller normalises later).
        output_frames.append(out)

    combined = pd.concat(output_frames, ignore_index=True)

    # Enforce final column order.
    combined = combined[["section_id", "x", "y", "z"] + shared_genes]
    return combined


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Combine preprocessed Visium sections into one training table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--sections", nargs="+", required=True, metavar="FILE",
        help="Preprocessed section files (.parquet or .csv), one per section.",
    )
    p.add_argument(
        "--section-ids", nargs="+", required=True, metavar="ID",
        help="Section identifiers, one per file (must match --sections length).",
    )
    p.add_argument(
        "--z-values", nargs="+", type=float, metavar="Z",
        help="Z coordinate for each section. If omitted, uses 0, 1, 2, … "
             "in the order supplied to --sections.",
    )
    p.add_argument(
        "--gene-list", metavar="GENES",
        help="Optional comma-separated gene panel to keep.  Must be a subset "
             "of genes shared across all sections.",
    )
    p.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help="Output path for the combined table.",
    )

    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if len(args.sections) != len(args.section_ids):
        sys.exit(
            f"--sections ({len(args.sections)}) and "
            f"--section-ids ({len(args.section_ids)}) must have the same length."
        )
    if args.z_values is not None and len(args.z_values) != len(args.sections):
        sys.exit(
            f"--z-values ({len(args.z_values)}) must have the same length as "
            f"--sections ({len(args.sections)})."
        )
    for path in args.sections:
        if not Path(path).exists():
            sys.exit(f"Section file not found: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    validate_args(args)

    section_files = [Path(p) for p in args.sections]
    section_ids = args.section_ids
    z_values = (
        args.z_values
        if args.z_values is not None
        else list(range(len(section_files)))
    )
    gene_list = (
        [g.strip() for g in args.gene_list.split(",") if g.strip()]
        if args.gene_list
        else None
    )

    combined = combine(
        section_files=section_files,
        section_ids=section_ids,
        z_values=z_values,
        gene_list=gene_list,
    )

    out_path = save_output(combined, Path(args.output))
    log.info("Saved combined table → %s", out_path)

    # ── Summary ───────────────────────────────────────────────────────────
    gene_cols = [c for c in combined.columns if c not in META_COLS]
    print("\n" + "─" * 56)
    print(f"  Sections : {len(section_ids)}")
    for sid, z in zip(section_ids, z_values):
        n = (combined["section_id"] == sid).sum()
        print(f"    {sid:<28} z={z:<6.1f}  {n:>5} spots")
    print(f"  Total    : {len(combined):,} spots")
    print(f"  Genes    : {len(gene_cols)}")
    print(f"    {', '.join(gene_cols[:8])}" + (" …" if len(gene_cols) > 8 else ""))
    print(f"  Output   : {out_path}")
    print("─" * 56 + "\n")


if __name__ == "__main__":
    main()
