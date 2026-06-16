"""Load and preprocess a single 10x Visium section.

Public API
----------
load_visium_section(visium_dir, section_id, ...)  →  pd.DataFrame
    Full pipeline: read → filter → normalize → select genes → assemble.

load_tissue_positions(visium_dir)  →  pd.DataFrame
    Standalone utility: returns the raw spot-coordinate table.

All scanpy imports are deferred so the rest of the codebase does not require
scanpy unless preprocessing is actually being run.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scanpy lazy import
# ---------------------------------------------------------------------------

def _require_scanpy():
    """Import and return scanpy, raising a helpful error if it is missing."""
    try:
        import scanpy as sc
        return sc
    except ImportError:
        raise ImportError(
            "scanpy is required for Visium preprocessing.\n"
            "Install it with:  pip install scanpy"
        )


# ---------------------------------------------------------------------------
# Spatial coordinate loading (standalone utility)
# ---------------------------------------------------------------------------

# Space Ranger < 2.0 writes a headerless CSV; >= 2.0 adds a header row.
_POSITIONS_FILES = [
    "tissue_positions.csv",         # Space Ranger >= 2.0  (has header)
    "tissue_positions_list.csv",    # Space Ranger <  2.0  (no header)
]
_POSITIONS_COLS = [
    "barcode", "in_tissue",
    "array_row", "array_col",
    "pxl_row_in_fullres", "pxl_col_in_fullres",
]


def load_tissue_positions(visium_dir: str | Path) -> pd.DataFrame:
    """Read spot coordinates from the Space Ranger spatial directory.

    Handles both tissue_positions.csv (SR >= 2.0, has header) and
    tissue_positions_list.csv (SR < 2.0, no header).

    Args:
        visium_dir: Path to the Space Ranger output directory.

    Returns:
        DataFrame with columns: barcode, in_tissue, array_row, array_col,
        pxl_row_in_fullres, pxl_col_in_fullres.
        Rows are filtered to in-tissue spots only.
    """
    visium_dir = Path(visium_dir)
    spatial_dir = visium_dir / "spatial"
    if not spatial_dir.is_dir():
        raise FileNotFoundError(
            f"Expected a 'spatial/' subdirectory in {visium_dir}"
        )

    for fname in _POSITIONS_FILES:
        path = spatial_dir / fname
        if not path.exists():
            continue

        # Detect header by inspecting the first field of the first line.
        first_field = path.read_text().split(",", 1)[0].strip()
        has_header = first_field == "barcode"

        df = pd.read_csv(path, header=0 if has_header else None)
        if not has_header:
            df.columns = _POSITIONS_COLS

        log.info("Loaded tissue positions from %s (%d spots total)", fname, len(df))

        n_total = len(df)
        df = df[df["in_tissue"] == 1].reset_index(drop=True)
        log.info("Retained %d / %d in-tissue spots", len(df), n_total)
        return df

    raise FileNotFoundError(
        f"No tissue positions file found under {spatial_dir}.\n"
        f"Expected one of: {_POSITIONS_FILES}"
    )


def load_scalefactors(visium_dir: str | Path) -> dict:
    """Return the scalefactors JSON as a dict (empty dict if the file is absent)."""
    path = Path(visium_dir) / "spatial" / "scalefactors_json.json"
    if path.exists():
        return json.loads(path.read_text())
    log.warning("scalefactors_json.json not found in %s/spatial/", visium_dir)
    return {}


# ---------------------------------------------------------------------------
# Expression loading
# ---------------------------------------------------------------------------

def read_expression(
    visium_dir: str | Path,
    min_counts: int = 100,
) -> "sc.AnnData":
    """Load the count matrix and spatial metadata via scanpy.

    Uses filtered_feature_bc_matrix.h5 by default; falls back to the raw matrix
    if the filtered one is absent.  Spots with fewer than min_counts total UMIs
    are removed.

    The returned AnnData object has:
      - adata.obsm['spatial']   : (N, 2) full-res pixel coordinates [col, row]
      - adata.obs['array_row']  : Visium array row index
      - adata.obs['array_col']  : Visium array column index

    Args:
        visium_dir: Space Ranger output directory.
        min_counts: Minimum total UMI count per spot.

    Returns:
        Raw (un-normalised) AnnData with spatial info embedded.
    """
    sc = _require_scanpy()
    visium_dir = Path(visium_dir)

    # Prefer the filtered matrix; fall back to raw.
    for candidate in ("filtered_feature_bc_matrix.h5", "raw_feature_bc_matrix.h5"):
        h5 = visium_dir / candidate
        if h5.exists():
            count_file = candidate
            if candidate.startswith("raw"):
                log.warning(
                    "filtered_feature_bc_matrix.h5 not found; using raw matrix."
                )
            break
    else:
        raise FileNotFoundError(
            f"No feature-barcode matrix (.h5) found in {visium_dir}."
        )

    log.info("Reading %s via scanpy …", count_file)
    adata = sc.read_visium(
        path=visium_dir,
        count_file=count_file,
        load_images=False,
    )
    adata.var_names_make_unique()
    log.info("Loaded: %d spots × %d genes", adata.n_obs, adata.n_vars)

    sc.pp.filter_cells(adata, min_counts=min_counts)
    log.info(
        "After min_counts=%d filter: %d spots remain", min_counts, adata.n_obs
    )
    return adata


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(adata: "sc.AnnData") -> "sc.AnnData":
    """Normalize each spot to 10 000 counts and apply log1p.

    Modifies adata in place; also returns it for chaining.
    """
    sc = _require_scanpy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    log.info("Normalization: total-count → log1p done.")
    return adata


# ---------------------------------------------------------------------------
# Gene selection
# ---------------------------------------------------------------------------

def select_genes(
    adata: "sc.AnnData",
    n_top_genes: int = 50,
    gene_list: list[str] | None = None,
) -> list[str]:
    """Return the gene names to include in the processed output.

    Args:
        adata:        Normalized AnnData.
        n_top_genes:  Number of highly variable genes to select when gene_list
                      is not provided.
        gene_list:    Explicit list of gene symbols.  Genes absent from the data
                      are silently dropped (with a warning).

    Returns:
        List of gene symbol strings.
    """
    sc = _require_scanpy()

    if gene_list is not None:
        available = set(adata.var_names)
        missing = [g for g in gene_list if g not in available]
        if missing:
            log.warning(
                "%d requested genes not in data (skipped): %s%s",
                len(missing),
                missing[:8],
                " …" if len(missing) > 8 else "",
            )
        selected = [g for g in gene_list if g in available]
        if not selected:
            raise ValueError(
                "None of the requested genes were found in the expression matrix.\n"
                f"First 10 available genes: {list(adata.var_names[:10])}"
            )
        log.info("Using %d / %d requested genes.", len(selected), len(gene_list))
        return selected

    # Highly variable gene selection.
    n_select = min(n_top_genes, adata.n_vars)
    if adata.n_obs < 2:
        raise ValueError(
            f"Only {adata.n_obs} spot(s) remain after filtering — "
            "too few for HVG selection.  Lower --min-counts or check data quality."
        )

    sc.pp.highly_variable_genes(adata, n_top_genes=n_select, flavor="seurat")
    selected = adata.var_names[adata.var["highly_variable"]].tolist()
    log.info(
        "Selected %d highly variable genes (requested %d).",
        len(selected), n_top_genes,
    )
    return selected


# ---------------------------------------------------------------------------
# DataFrame assembly
# ---------------------------------------------------------------------------

def build_dataframe(
    adata: "sc.AnnData",
    section_id: str,
    z: float,
    gene_names: list[str],
    coord_type: str = "pixel",
) -> pd.DataFrame:
    """Assemble the final spot DataFrame from a processed AnnData.

    Coordinate conventions
    ----------------------
    pixel (default): uses full-resolution pixel positions from the Space Ranger
        output.  adata.obsm['spatial'] stores [col_pixel, row_pixel], so:
            x = pxl_col_in_fullres  (horizontal / left-right)
            y = pxl_row_in_fullres  (vertical   / top-bottom)

    array: uses the Visium hexagonal array grid indices stored in adata.obs:
            x = array_col
            y = array_row

    Args:
        adata:       Normalized AnnData produced by read_expression + normalize.
        section_id:  Value written to the section_id column.
        z:           Z coordinate for this section (use section index when
                     combining multiple sections later).
        gene_names:  Genes to include; must be present in adata.var_names.
        coord_type:  'pixel' or 'array'.

    Returns:
        DataFrame with columns: section_id, x, y, z, <gene_names…>
    """
    import scipy.sparse

    if coord_type == "pixel":
        if "spatial" not in adata.obsm:
            raise KeyError(
                "adata.obsm['spatial'] is missing.  "
                "Ensure read_expression() was called with sc.read_visium()."
            )
        # obsm['spatial'] column 0 = pxl_col (x), column 1 = pxl_row (y).
        xy = adata.obsm["spatial"]
        x = xy[:, 0].astype(np.float32)
        y = xy[:, 1].astype(np.float32)

    elif coord_type == "array":
        for col in ("array_col", "array_row"):
            if col not in adata.obs.columns:
                raise KeyError(
                    f"adata.obs['{col}'] is missing.  "
                    "Ensure read_expression() was called with sc.read_visium()."
                )
        x = adata.obs["array_col"].values.astype(np.float32)
        y = adata.obs["array_row"].values.astype(np.float32)

    else:
        raise ValueError(
            f"Unknown coord_type '{coord_type}'. Choose 'pixel' or 'array'."
        )

    # Extract expression for the requested genes.
    X = adata[:, gene_names].X
    if scipy.sparse.issparse(X):
        X = X.toarray()
    expr = X.astype(np.float32)

    meta = pd.DataFrame(
        {"section_id": section_id, "x": x, "y": y, "z": np.float32(z)},
    )
    genes_df = pd.DataFrame(expr, columns=gene_names)

    return pd.concat([meta, genes_df], axis=1)


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def load_visium_section(
    visium_dir: str | Path,
    section_id: str,
    *,
    z: float = 0.0,
    n_top_genes: int = 50,
    gene_list: list[str] | None = None,
    coord_type: str = "pixel",
    min_counts: int = 100,
) -> pd.DataFrame:
    """Preprocess a single 10x Visium section into a spot DataFrame.

    Pipeline:
        1. Read filtered_feature_bc_matrix.h5 via scanpy.
        2. Filter spots with fewer than min_counts UMIs.
        3. Normalize to 10k counts per spot + log1p.
        4. Select genes (HVG or user-supplied list).
        5. Assemble a flat DataFrame ready for SpatialExpressionDataset.

    Args:
        visium_dir:  Path to the Space Ranger output directory.
        section_id:  Identifier for this section (stored in the section_id column
                     and used in the output filename).
        z:           Z coordinate to assign.  Set to the section index when
                     combining multiple sections; leave at 0.0 for a single section.
        n_top_genes: Number of highly variable genes (ignored when gene_list given).
        gene_list:   Explicit list of gene symbols to include.
        coord_type:  'pixel' (full-res pixel positions) or 'array' (grid indices).
        min_counts:  Minimum UMI count per spot.

    Returns:
        DataFrame with columns: section_id, x, y, z, <gene columns>
    """
    visium_dir = Path(visium_dir)
    if not visium_dir.is_dir():
        raise FileNotFoundError(f"Visium directory not found: {visium_dir}")

    log.info("── Preprocessing section '%s' ──", section_id)

    adata = read_expression(visium_dir, min_counts=min_counts)
    adata = normalize(adata)
    genes = select_genes(adata, n_top_genes=n_top_genes, gene_list=gene_list)
    df = build_dataframe(adata, section_id=section_id, z=z,
                         gene_names=genes, coord_type=coord_type)

    log.info(
        "Section '%s': %d spots, %d genes, coord_type='%s'",
        section_id, len(df), len(genes), coord_type,
    )
    return df
