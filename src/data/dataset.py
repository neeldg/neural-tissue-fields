"""SpatialExpressionDataset: wraps a DataFrame of spatial transcriptomics spots."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SpatialExpressionDataset(Dataset):
    """Map-style dataset for (x, y, z) -> gene expression.

    Coordinates are normalized to [-1, 1] per axis using the min/max of
    the provided DataFrame (or externally supplied bounds, useful when fitting
    bounds on the training split and reusing them on a held-out split).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        coord_cols: list[str],
        gene_cols: list[str],
        coord_bounds: dict | None = None,
    ):
        """
        Args:
            df:           DataFrame with at least coord_cols + gene_cols.
            coord_cols:   Ordered list of column names for spatial coordinates,
                          e.g. ['x', 'y', 'z'].
            gene_cols:    Column names for gene expression values.
            coord_bounds: Optional dict {col: (min, max)} to normalize against.
                          If None, bounds are computed from `df` itself.
        """
        self.coord_cols = coord_cols
        self.gene_cols = gene_cols

        coords_raw = df[coord_cols].values.astype(np.float32)
        exprs_raw = df[gene_cols].values.astype(np.float32)

        # Build or store coordinate bounds so callers can reuse them.
        if coord_bounds is None:
            self.coord_bounds = {
                col: (float(df[col].min()), float(df[col].max()))
                for col in coord_cols
            }
        else:
            self.coord_bounds = coord_bounds

        # Normalize each axis to [-1, 1].
        coords_norm = np.empty_like(coords_raw)
        for i, col in enumerate(coord_cols):
            lo, hi = self.coord_bounds[col]
            span = hi - lo
            if span == 0:
                # Degenerate axis (all values identical) → map to 0.
                coords_norm[:, i] = 0.0
            else:
                coords_norm[:, i] = 2.0 * (coords_raw[:, i] - lo) / span - 1.0

        self.coords = torch.from_numpy(coords_norm)    # (N, len(coord_cols))
        self.exprs = torch.from_numpy(exprs_raw)       # (N, len(gene_cols))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.coords[idx], self.exprs[idx]

    # ------------------------------------------------------------------
    @property
    def n_coords(self) -> int:
        return len(self.coord_cols)

    @property
    def n_genes(self) -> int:
        return len(self.gene_cols)
