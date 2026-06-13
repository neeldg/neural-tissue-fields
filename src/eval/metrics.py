"""Evaluation metrics for neural field predictions."""

import numpy as np
import pandas as pd
import torch


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_metrics(
    true: torch.Tensor | np.ndarray,
    pred: torch.Tensor | np.ndarray,
    gene_names: list[str] | None = None,
) -> pd.DataFrame:
    """Compute per-gene MSE, MAE, and Pearson r between true and predicted expression.

    Args:
        true:        (N, G) array of ground-truth expression values.
        pred:        (N, G) array of predicted expression values.
        gene_names:  Optional list of G gene names for the output index.

    Returns:
        DataFrame with columns ['mse', 'mae', 'pearson_r'] and gene names as index.
    """
    true = _to_numpy(true).astype(np.float64)
    pred = _to_numpy(pred).astype(np.float64)

    assert true.shape == pred.shape, (
        f"Shape mismatch: true {true.shape} vs pred {pred.shape}"
    )

    n_genes = true.shape[1] if true.ndim == 2 else 1
    if true.ndim == 1:
        true = true[:, None]
        pred = pred[:, None]

    mse = np.mean((true - pred) ** 2, axis=0)
    mae = np.mean(np.abs(true - pred), axis=0)

    pearson_r = np.empty(n_genes)
    for g in range(n_genes):
        t, p = true[:, g], pred[:, g]
        # Guard against constant vectors (std == 0).
        if t.std() < 1e-8 or p.std() < 1e-8:
            pearson_r[g] = float("nan")
        else:
            pearson_r[g] = float(np.corrcoef(t, p)[0, 1])

    index = gene_names if gene_names is not None else [f"gene_{g}" for g in range(n_genes)]
    return pd.DataFrame({"mse": mse, "mae": mae, "pearson_r": pearson_r}, index=index)


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.Series:
    """Return mean MSE, MAE, and Pearson r across genes (ignores NaN genes)."""
    return metrics_df.mean(skipna=True).rename("mean")
