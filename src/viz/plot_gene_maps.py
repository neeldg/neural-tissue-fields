"""Visualize true vs predicted gene expression on 2D spatial coordinates."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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


def plot_method_comparison(
    x: np.ndarray,
    y: np.ndarray,
    true_expr: np.ndarray,
    method_preds: list[tuple[str, np.ndarray]],
    gene_names: list[str],
    x_label: str = "x",
    y_label: str = "y",
    point_size: float = 4.0,
    cmap: str = "viridis",
    error_cmap: str = "Reds",
    figsize_per_panel: tuple[float, float] = (2.5, 2.2),
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Side-by-side comparison of all methods for selected genes.

    Layout — one row per gene, columns in this fixed order:
        True | method_preds[0] | ... | method_preds[-1] | method_preds[-1] Abs. Error

    The expression panels (True + all predictions) share a colour scale per gene row
    so magnitudes are directly comparable.  The absolute-error panel uses a separate
    sequential colourmap anchored at zero.

    Args:
        x:             (N,) x spatial coordinates.
        y:             (N,) y spatial coordinates.
        true_expr:     (N, G) true expression values for the G plotted genes.
        method_preds:  Ordered list of (display_label, (N, G) predictions).
                       The LAST entry is assumed to be the neural field and is used
                       for the absolute-error panel.
        gene_names:    G gene display names (row labels).
        x_label:       Axis label for x coordinate.
        y_label:       Axis label for y coordinate.
        point_size:    Scatter marker size.
        cmap:          Colourmap for expression panels.
        error_cmap:    Colourmap for the absolute-error panel.
        figsize_per_panel: (width, height) of each subplot panel in inches.
        save_path:     If provided, save the figure here.

    Returns:
        The matplotlib Figure.
    """
    n_genes = len(gene_names)
    n_methods = len(method_preds)
    # Columns: True + each method + error panel
    n_cols = 1 + n_methods + 1

    assert true_expr.shape == (len(x), n_genes), (
        f"true_expr shape {true_expr.shape} does not match "
        f"(N={len(x)}, G={n_genes})"
    )

    last_label = method_preds[-1][0]
    col_labels = (
        ["True"]
        + [label for label, _ in method_preds]
        + [f"{last_label}\nAbs. Error"]
    )

    fig_w = figsize_per_panel[0] * n_cols
    fig_h = figsize_per_panel[1] * n_genes
    fig, axes = plt.subplots(
        n_genes, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    for row, gene in enumerate(gene_names):
        true_g = true_expr[:, row]
        preds_g = [arr[:, row] for _, arr in method_preds]
        error_g = np.abs(true_g - preds_g[-1])

        # Shared expression colour scale across True + all predictions.
        all_expr = np.concatenate([true_g] + preds_g)
        vmin, vmax = float(all_expr.min()), float(all_expr.max())
        # Error scale anchored at zero.
        err_max = float(error_g.max()) or 1.0

        panels = (
            [(true_g, vmin, vmax, cmap)]
            + [(p, vmin, vmax, cmap) for p in preds_g]
            + [(error_g, 0.0, err_max, error_cmap)]
        )

        for col, (vals, v0, v1, cm) in enumerate(panels):
            ax = axes[row, col]
            sc = ax.scatter(
                x, y, c=vals, s=point_size,
                cmap=cm, vmin=v0, vmax=v1,
                rasterized=True, linewidths=0,
            )

            # Column headers only on the first row.
            if row == 0:
                ax.set_title(col_labels[col], fontsize=8, fontweight="bold", pad=4)

            # Gene name as row label on the leftmost column.
            if col == 0:
                ax.set_ylabel(
                    gene, fontsize=8, rotation=0,
                    labelpad=48, va="center", ha="right",
                )

            # x-axis label only on the bottom row.
            if row == n_genes - 1:
                ax.set_xlabel(x_label, fontsize=7)

            ax.tick_params(labelsize=5)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(3))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(3))

            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=5)

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
