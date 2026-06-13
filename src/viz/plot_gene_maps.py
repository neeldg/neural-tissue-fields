"""Visualize true vs predicted gene expression on 2D spatial coordinates."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_gene_maps(
    df: pd.DataFrame,
    gene_names: list[str],
    true_cols: list[str],
    pred_cols: list[str],
    x_col: str = "x",
    y_col: str = "y",
    ncols: int = 2,
    point_size: float = 4.0,
    cmap: str = "viridis",
    save_path: Path | str | None = None,
    figsize_per_panel: tuple[float, float] = (3.5, 3.0),
) -> plt.Figure:
    """Plot true and predicted expression side-by-side for each gene.

    Each row in the figure corresponds to one gene:
        left panel  – true expression
        right panel – predicted expression

    Args:
        df:               DataFrame containing spatial coordinates and expression columns.
        gene_names:       Display names for each gene (used as subplot titles).
        true_cols:        Column names in `df` for true expression, one per gene.
        pred_cols:        Column names in `df` for predicted expression, one per gene.
        x_col:            Column name for x spatial coordinate.
        y_col:            Column name for y spatial coordinate.
        ncols:            Number of column pairs (true+pred counts as 2 columns total).
                          Currently fixed at 2 (true | pred); reserved for future use.
        point_size:       Scatter point size.
        cmap:             Matplotlib colormap name.
        save_path:        If provided, save the figure to this path.
        figsize_per_panel: (width, height) per individual panel in inches.

    Returns:
        The matplotlib Figure object.
    """
    assert len(gene_names) == len(true_cols) == len(pred_cols), (
        "gene_names, true_cols, and pred_cols must have the same length."
    )

    n_genes = len(gene_names)
    n_rows = n_genes
    n_cols = 2  # true | pred

    fig_w = figsize_per_panel[0] * n_cols
    fig_h = figsize_per_panel[1] * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    x = df[x_col].values
    y = df[y_col].values

    for row, (gene, tcol, pcol) in enumerate(zip(gene_names, true_cols, pred_cols)):
        t_vals = df[tcol].values.astype(float)
        p_vals = df[pcol].values.astype(float)

        # Share colour scale between true and predicted for fair comparison.
        vmin = min(t_vals.min(), p_vals.min())
        vmax = max(t_vals.max(), p_vals.max())

        for col_idx, (vals, title) in enumerate(
            [(t_vals, "True"), (p_vals, "Predicted")]
        ):
            ax = axes[row, col_idx]
            sc = ax.scatter(x, y, c=vals, s=point_size, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{gene} – {title}", fontsize=9)
            ax.set_xlabel(x_col, fontsize=7)
            ax.set_ylabel(y_col, fontsize=7)
            ax.tick_params(labelsize=6)
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_pearson_barplot(
    pearson_series: pd.Series,
    title: str = "Per-gene Pearson r (held-out section)",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Horizontal bar chart of per-gene Pearson correlation."""
    fig, ax = plt.subplots(figsize=(6, max(2.5, len(pearson_series) * 0.35)))
    pearson_series = pearson_series.sort_values()
    colors = ["#e74c3c" if v < 0 else "#2980b9" for v in pearson_series]
    ax.barh(pearson_series.index, pearson_series.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Pearson r")
    ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
