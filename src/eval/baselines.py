"""Simple spatial baselines for benchmarking the neural field.

All three baselines share a common call signature:

    predict(df_train, df_test, coord_cols, gene_cols, **kwargs) -> np.ndarray

    Returns an (N_test, G) float32 array of predicted expression values,
    in the same row order as df_test.

Coordinate columns are expected to be raw (un-normalised) values.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _xy_cols(coord_cols: list[str]) -> tuple[str, str]:
    """Return the first two coord columns as (x_col, y_col)."""
    return coord_cols[0], coord_cols[1]


def _z_col(coord_cols: list[str]) -> str:
    """Return the third coord column as the z / section-depth column."""
    if len(coord_cols) < 3:
        raise ValueError(
            f"linear_z_interpolation_baseline requires at least 3 coord columns; "
            f"got {coord_cols}"
        )
    return coord_cols[2]


def _nearest_xy(
    query_xy: np.ndarray,
    ref_xy: np.ndarray,
    ref_expr: np.ndarray,
) -> np.ndarray:
    """For each query point, return expression of the nearest point in ref.

    Args:
        query_xy:  (N, 2) query x/y coordinates.
        ref_xy:    (M, 2) reference x/y coordinates.
        ref_expr:  (M, G) expression values for the reference points.

    Returns:
        (N, G) expression values from the nearest reference point.
    """
    tree = cKDTree(ref_xy)
    _, indices = tree.query(query_xy, k=1, workers=-1)
    return ref_expr[indices]


# ---------------------------------------------------------------------------
# 1. Nearest-section baseline
# ---------------------------------------------------------------------------

def nearest_section_baseline(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    coord_cols: list[str],
    gene_cols: list[str],
) -> np.ndarray:
    """Predict held-out expression by copying from the z-closest training section.

    For each test point:
      1. Find the training section whose mean z is closest to the test z.
      2. If an exact (x, y) match exists in that section, use it directly.
      3. Otherwise find the nearest (x, y) neighbour within that section.

    Args:
        df_train:    Training DataFrame (all sections except the held-out one).
        df_test:     Held-out section DataFrame.
        coord_cols:  [x_col, y_col, z_col] column names.
        gene_cols:   Gene expression column names.

    Returns:
        (N_test, G) float32 predictions.
    """
    x_col, y_col = _xy_cols(coord_cols)
    z_col = _z_col(coord_cols)

    test_z = df_test[z_col].iloc[0]   # all rows share the same z in a section

    # Find the single closest training section by z distance.
    section_z = df_train.groupby(z_col)[z_col].first()
    nearest_z = section_z.iloc[(section_z - test_z).abs().argsort().iloc[0]]
    df_ref = df_train[df_train[z_col] == nearest_z].reset_index(drop=True)

    test_xy = df_test[[x_col, y_col]].values
    ref_xy = df_ref[[x_col, y_col]].values
    ref_expr = df_ref[gene_cols].values.astype(np.float32)

    # Build a lookup dict for O(1) exact-match queries.
    exact: dict[tuple, int] = {
        (float(row[x_col]), float(row[y_col])): i
        for i, row in df_ref[[x_col, y_col]].iterrows()
    }

    preds = np.empty((len(df_test), len(gene_cols)), dtype=np.float32)
    no_match_mask = np.ones(len(df_test), dtype=bool)

    for i, (tx, ty) in enumerate(test_xy):
        idx = exact.get((float(tx), float(ty)))
        if idx is not None:
            preds[i] = ref_expr[df_ref.index.get_loc(idx)]
            no_match_mask[i] = False

    # Batch nearest-neighbour for all points without an exact match.
    if no_match_mask.any():
        preds[no_match_mask] = _nearest_xy(
            test_xy[no_match_mask], ref_xy, ref_expr
        )

    return preds


# ---------------------------------------------------------------------------
# 2. Linear z-interpolation baseline
# ---------------------------------------------------------------------------

def linear_z_interpolation_baseline(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    coord_cols: list[str],
    gene_cols: list[str],
) -> np.ndarray:
    """Predict by linearly interpolating between the bracketing training sections.

    For each test point:
      1. Find the nearest lower-z section (z_lo) and upper-z section (z_hi).
      2. Look up expression at the same (x, y) in each — or nearest if not exact.
      3. Linearly interpolate: expr = expr_lo + t * (expr_hi - expr_lo)
         where t = (test_z - z_lo) / (z_hi - z_lo).

    Edge cases:
      - If the test z is at or below the minimum training z, uses that section directly.
      - If the test z is at or above the maximum training z, uses that section directly.

    Args:
        df_train:    Training DataFrame.
        df_test:     Held-out section DataFrame.
        coord_cols:  [x_col, y_col, z_col].
        gene_cols:   Gene expression column names.

    Returns:
        (N_test, G) float32 predictions.
    """
    x_col, y_col = _xy_cols(coord_cols)
    z_col = _z_col(coord_cols)

    test_z = float(df_test[z_col].iloc[0])
    train_zs = sorted(df_train[z_col].unique())

    # Identify bracketing z values.
    lower_zs = [z for z in train_zs if z <= test_z]
    upper_zs = [z for z in train_zs if z >= test_z]

    if not lower_zs:
        # Extrapolate below: use the lowest training section.
        z_lo = z_hi = train_zs[0]
    elif not upper_zs:
        # Extrapolate above: use the highest training section.
        z_lo = z_hi = train_zs[-1]
    else:
        z_lo = max(lower_zs)
        z_hi = min(upper_zs)

    def _lookup(z_val: float) -> np.ndarray:
        """Get expression for each test (x,y) from the section at z_val."""
        df_ref = df_train[df_train[z_col] == z_val].reset_index(drop=True)
        ref_xy = df_ref[[x_col, y_col]].values
        ref_expr = df_ref[gene_cols].values.astype(np.float32)
        test_xy = df_test[[x_col, y_col]].values

        exact: dict[tuple, int] = {
            (float(r[x_col]), float(r[y_col])): i
            for i, r in df_ref[[x_col, y_col]].iterrows()
        }

        out = np.empty((len(df_test), len(gene_cols)), dtype=np.float32)
        no_match = np.ones(len(df_test), dtype=bool)

        for i, (tx, ty) in enumerate(test_xy):
            idx = exact.get((float(tx), float(ty)))
            if idx is not None:
                out[i] = ref_expr[df_ref.index.get_loc(idx)]
                no_match[i] = False

        if no_match.any():
            out[no_match] = _nearest_xy(test_xy[no_match], ref_xy, ref_expr)

        return out

    expr_lo = _lookup(z_lo)

    if z_lo == z_hi:
        return expr_lo

    expr_hi = _lookup(z_hi)
    t = (test_z - z_lo) / (z_hi - z_lo)   # interpolation weight in [0, 1]
    return (expr_lo + t * (expr_hi - expr_lo)).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. k-NN baseline in (x, y, z) space
# ---------------------------------------------------------------------------

def knn_xyz_baseline(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    coord_cols: list[str],
    gene_cols: list[str],
    k: int = 10,
) -> np.ndarray:
    """Predict by averaging expression over the k nearest training points in xyz.

    Coordinates are normalised to [0, 1] per axis (using training data bounds)
    before computing distances, so that x, y, and z contribute on equal footing.

    Args:
        df_train:    Training DataFrame.
        df_test:     Held-out section DataFrame.
        coord_cols:  Coordinate column names (at least 2; all are used for distance).
        gene_cols:   Gene expression column names.
        k:           Number of nearest neighbours.

    Returns:
        (N_test, G) float32 predictions.
    """
    train_coords = df_train[coord_cols].values.astype(np.float64)
    test_coords = df_test[coord_cols].values.astype(np.float64)

    # Normalise each axis to [0, 1] using training bounds so axes are commensurate.
    lo = train_coords.min(axis=0)
    hi = train_coords.max(axis=0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)   # avoid division by zero

    train_norm = (train_coords - lo) / span
    test_norm = (test_coords - lo) / span

    k_clamped = min(k, len(df_train))
    tree = cKDTree(train_norm)
    _, indices = tree.query(test_norm, k=k_clamped, workers=-1)   # (N_test, k)

    train_expr = df_train[gene_cols].values.astype(np.float32)

    # indices shape is (N_test,) when k_clamped == 1, so normalise.
    if k_clamped == 1:
        indices = indices[:, None]

    # Average over the k neighbours.
    preds = train_expr[indices].mean(axis=1)   # (N_test, G)
    return preds.astype(np.float32)
