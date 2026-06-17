"""Load and preprocess a single 10x Visium HD section at a chosen bin level.

Visium HD Space Ranger output structure assumed:
  binned_outputs/
    square_002um/
    square_008um/
    square_016um/          ← one of these is the --bin-level
      filtered_feature_bc_matrix.h5
      spatial/
        tissue_positions.parquet      (primary format for Visium HD)
        tissue_positions.csv          (fallback)
        tissue_positions_list.csv     (legacy fallback)
        scalefactors_json.json
        tissue_hires_image.png

Key difference from standard Visium:
  - Positions are stored as parquet, not CSV.
  - There is no hexagonal array grid; bins live on a regular square grid.
  - sc.read_visium() does not understand this directory layout, so the H5 is
    loaded with sc.read_10x_h5() and coordinates are joined manually.
  - Bin sizes are much smaller (8–16 µm), so per-bin counts are lower;
    the default min_counts is set accordingly.

Public API
----------
load_visium_hd_section(binned_dir, bin_level, section_id, ...)  →  pd.DataFrame
find_bin_level_dir(binned_dir, bin_level)  →  Path
load_hd_positions(bin_level_dir)  →  pd.DataFrame
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# normalize() and select_genes() are identical to standard Visium; import
# rather than duplicate.
from src.data.load_visium import normalize, select_genes

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scanpy lazy import (same guard used across the data subpackage)
# ---------------------------------------------------------------------------

def _require_scanpy():
    try:
        import scanpy as sc
        return sc
    except ImportError:
        raise ImportError(
            "scanpy is required for Visium HD preprocessing.\n"
            "Install it with:  pip install scanpy"
        )


# ---------------------------------------------------------------------------
# Bin-level directory resolution
# ---------------------------------------------------------------------------

# Known bin-level folder names produced by Space Ranger HD.
KNOWN_BIN_LEVELS = ["square_002um", "square_008um", "square_016um"]


def find_bin_level_dir(binned_dir: str | Path, bin_level: str) -> Path:
    """Return the path to the requested bin-level subdirectory.

    Args:
        binned_dir: Path to the ``binned_outputs/`` directory.
        bin_level:  Subfolder name, e.g. ``'square_016um'``.

    Raises:
        FileNotFoundError: If the directory or the requested bin level is
            absent, with a listing of what *is* available to help diagnose.
    """
    binned_dir = Path(binned_dir)
    if not binned_dir.is_dir():
        raise FileNotFoundError(f"binned_outputs directory not found: {binned_dir}")

    target = binned_dir / bin_level
    if target.is_dir():
        return target

    available = sorted(d.name for d in binned_dir.iterdir() if d.is_dir())
    raise FileNotFoundError(
        f"Bin level '{bin_level}' not found in {binned_dir}.\n"
        f"Available subdirectories: {available}\n"
        f"Standard bin level names: {KNOWN_BIN_LEVELS}"
    )


# ---------------------------------------------------------------------------
# Spatial coordinate loading
# ---------------------------------------------------------------------------

# Priority order for position files inside the bin-level spatial/ folder.
_HD_POSITIONS_CANDIDATES = [
    ("tissue_positions.parquet", "parquet"),
    ("tissue_positions.csv",     "csv"),
    ("tissue_positions_list.csv", "csv_noheader"),
]

_POSITIONS_COLS = [
    "barcode", "in_tissue",
    "array_row", "array_col",
    "pxl_row_in_fullres", "pxl_col_in_fullres",
]


def load_hd_positions(bin_level_dir: str | Path) -> pd.DataFrame:
    """Load bin spatial coordinates from the bin-level spatial/ directory.

    Tries tissue_positions.parquet first (the native Visium HD format), then
    falls back to CSV variants.

    Coordinate columns returned:
        pxl_col_in_fullres  →  x (horizontal, left-right)
        pxl_row_in_fullres  →  y (vertical,   top-bottom)

    Args:
        bin_level_dir: Path to the bin-level directory (e.g. ``square_016um/``).

    Returns:
        DataFrame indexed by barcode with at least ``pxl_row_in_fullres`` and
        ``pxl_col_in_fullres`` columns.  If an ``in_tissue`` column is present,
        only in-tissue bins are returned; otherwise all bins are returned.

    Raises:
        FileNotFoundError: If no positions file is found, with a directory
            listing to help diagnose.
    """
    bin_level_dir = Path(bin_level_dir)
    spatial_dir = bin_level_dir / "spatial"

    if not spatial_dir.is_dir():
        found_top = sorted(f.name for f in bin_level_dir.iterdir())
        raise FileNotFoundError(
            f"No 'spatial/' subdirectory found in {bin_level_dir}.\n"
            f"Contents of {bin_level_dir}: {found_top}"
        )

    for fname, fmt in _HD_POSITIONS_CANDIDATES:
        path = spatial_dir / fname
        if not path.exists():
            continue

        if fmt == "parquet":
            df = pd.read_parquet(path)
            log.info("Loaded positions from %s (%d bins)", fname, len(df))
        elif fmt == "csv":
            first_field = path.read_text().split(",", 1)[0].strip()
            has_header = first_field == "barcode"
            df = pd.read_csv(path, header=0 if has_header else None)
            if not has_header:
                # Legacy headerless format: assign standard column names.
                df.columns = _POSITIONS_COLS[: len(df.columns)]
            log.info("Loaded positions from %s (%d bins)", fname, len(df))
        else:  # csv_noheader
            df = pd.read_csv(path, header=None)
            df.columns = _POSITIONS_COLS[: len(df.columns)]
            log.info("Loaded positions from %s (%d bins)", fname, len(df))

        # Normalise index to barcode.
        if "barcode" in df.columns:
            df = df.set_index("barcode")
        elif df.index.name != "barcode":
            df.index.name = "barcode"

        # Filter to in-tissue bins if the column is present.
        if "in_tissue" in df.columns:
            n_total = len(df)
            df = df[df["in_tissue"] == 1]
            log.info("Retained %d / %d in-tissue bins", len(df), n_total)

        # Verify required coordinate columns exist.
        for required in ("pxl_row_in_fullres", "pxl_col_in_fullres"):
            if required not in df.columns:
                raise ValueError(
                    f"Positions file {fname} is missing column '{required}'.\n"
                    f"Columns found: {list(df.columns)}"
                )

        return df

    # Nothing found — show what is actually in the spatial directory.
    found = sorted(f.name for f in spatial_dir.iterdir())
    raise FileNotFoundError(
        f"No tissue positions file found in {spatial_dir}.\n"
        f"Files present: {found}\n"
        f"Expected one of: tissue_positions.parquet, tissue_positions.csv"
    )


# ---------------------------------------------------------------------------
# Expression loading
# ---------------------------------------------------------------------------

_H5_CANDIDATES = [
    "filtered_feature_bc_matrix.h5",
    "raw_feature_bc_matrix.h5",
]


def read_hd_expression(
    bin_level_dir: str | Path,
    min_counts: int = 10,
) -> "sc.AnnData":
    """Load the count matrix and attach spatial coordinates.

    Uses ``sc.read_10x_h5()`` (not ``sc.read_visium()``) because scanpy does
    not understand the Visium HD bin-level directory layout.  Spatial
    coordinates are loaded separately via :func:`load_hd_positions` and
    attached as ``adata.obsm['spatial']``.

    Args:
        bin_level_dir: Path to the bin-level directory.
        min_counts:    Minimum total UMI count per bin; lower-count bins are
                       dropped.  Default is 10 (much lower than standard Visium
                       because HD bins are far smaller and sparser).

    Returns:
        Raw (un-normalised) AnnData with:
          - ``adata.obsm['spatial']`` : (N, 2) pixel coords ``[col, row]``
          - obs_names                 : barcodes
    """
    sc = _require_scanpy()
    bin_level_dir = Path(bin_level_dir)

    # Locate the H5 file.
    h5_path: Path | None = None
    for candidate in _H5_CANDIDATES:
        p = bin_level_dir / candidate
        if p.exists():
            h5_path = p
            if candidate.startswith("raw"):
                log.warning(
                    "filtered_feature_bc_matrix.h5 not found; using raw matrix."
                )
            break

    if h5_path is None:
        found = sorted(f.name for f in bin_level_dir.iterdir() if f.is_file())
        raise FileNotFoundError(
            f"No feature-barcode matrix (.h5) found in {bin_level_dir}.\n"
            f"Files present in that directory: {found}\n"
            f"Expected one of: {_H5_CANDIDATES}"
        )

    log.info("Reading %s …", h5_path.name)
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()
    log.info("Loaded: %d bins × %d genes", adata.n_obs, adata.n_vars)

    # Load spatial coordinates and inner-join on barcode.
    positions = load_hd_positions(bin_level_dir)
    common = adata.obs_names.intersection(positions.index)

    if len(common) == 0:
        raise ValueError(
            "No barcode overlap between the expression matrix and the positions file.\n"
            f"  Expression barcodes (first 3): {list(adata.obs_names[:3])}\n"
            f"  Positions barcodes  (first 3): {list(positions.index[:3])}\n"
            "Check that both files belong to the same bin level."
        )
    if len(common) < adata.n_obs:
        n_dropped = adata.n_obs - len(common)
        log.warning(
            "%d bin(s) in the expression matrix have no position entry "
            "and will be dropped.",
            n_dropped,
        )

    adata = adata[common].copy()
    pos = positions.loc[common]

    # Attach coordinates: obsm['spatial'] = [pxl_col (x), pxl_row (y)]
    adata.obsm["spatial"] = np.column_stack([
        pos["pxl_col_in_fullres"].values.astype(np.float64),
        pos["pxl_row_in_fullres"].values.astype(np.float64),
    ])

    # Apply count filter after coordinate join.
    sc.pp.filter_cells(adata, min_counts=min_counts)
    log.info("After min_counts=%d filter: %d bins remain", min_counts, adata.n_obs)

    return adata


# ---------------------------------------------------------------------------
# DataFrame assembly
# ---------------------------------------------------------------------------

def build_hd_dataframe(
    adata: "sc.AnnData",
    section_id: str,
    z: float,
    gene_names: list[str],
) -> pd.DataFrame:
    """Assemble the final bin DataFrame from a processed AnnData.

    Coordinates are always full-resolution pixel positions:
        x = pxl_col_in_fullres   (horizontal, left-right)
        y = pxl_row_in_fullres   (vertical,   top-bottom)

    Args:
        adata:       Normalized AnnData produced by read_hd_expression + normalize.
        section_id:  Value written to the section_id column.
        z:           Z coordinate for this section.
        gene_names:  Genes to include; must be in adata.var_names.

    Returns:
        DataFrame with columns: section_id, x, y, z, <gene_names…>
    """
    import scipy.sparse

    if "spatial" not in adata.obsm:
        raise KeyError(
            "adata.obsm['spatial'] not found.  "
            "Ensure read_hd_expression() ran successfully."
        )

    xy = adata.obsm["spatial"]
    x = xy[:, 0].astype(np.float32)   # pxl_col = horizontal
    y = xy[:, 1].astype(np.float32)   # pxl_row = vertical

    X = adata[:, gene_names].X
    if scipy.sparse.issparse(X):
        X = X.toarray()
    expr = X.astype(np.float32)

    meta = pd.DataFrame({
        "section_id": section_id,
        "x": x,
        "y": y,
        "z": np.float32(z),
    })
    genes_df = pd.DataFrame(expr, columns=gene_names)

    return pd.concat([meta, genes_df], axis=1)


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def load_visium_hd_section(
    binned_dir: str | Path,
    bin_level: str,
    section_id: str,
    *,
    z: float = 0.0,
    n_top_genes: int = 50,
    gene_list: list[str] | None = None,
    min_counts: int = 10,
) -> pd.DataFrame:
    """Preprocess a single Visium HD section at one bin level.

    Pipeline:
        1. Resolve the bin-level directory under binned_outputs/.
        2. Load filtered_feature_bc_matrix.h5 with sc.read_10x_h5().
        3. Join spatial coordinates from tissue_positions.parquet (or CSV).
        4. Filter bins with fewer than min_counts UMIs.
        5. Normalize to 10k counts per bin + log1p.
        6. Select genes (HVG or user-supplied list).
        7. Assemble a flat DataFrame ready for SpatialExpressionDataset.

    Args:
        binned_dir:  Path to the ``binned_outputs/`` directory.
        bin_level:   Bin-level folder name, e.g. ``'square_016um'``.
        section_id:  Identifier for this section (stored in section_id column).
        z:           Z coordinate to assign.  Use the section index when
                     combining multiple sections later.
        n_top_genes: Number of HVGs to select (ignored when gene_list given).
        gene_list:   Explicit list of gene symbols to include.
        min_counts:  Minimum UMI count per bin.

    Returns:
        DataFrame with columns: section_id, x, y, z, <gene columns>
    """
    log.info("── Preprocessing HD section '%s' [%s] ──", section_id, bin_level)

    bin_level_dir = find_bin_level_dir(binned_dir, bin_level)
    adata = read_hd_expression(bin_level_dir, min_counts=min_counts)
    adata = normalize(adata)
    genes = select_genes(adata, n_top_genes=n_top_genes, gene_list=gene_list)
    df = build_hd_dataframe(adata, section_id=section_id, z=z, gene_names=genes)

    log.info(
        "Section '%s' [%s]: %d bins, %d genes",
        section_id, bin_level, len(df), len(genes),
    )
    return df
