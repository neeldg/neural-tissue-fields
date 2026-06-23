"""Rank genes by spatial autocorrelation using KNN-smoothed expression.

For each gene, we measure how well its expression at each spot is predicted
by the mean expression of its k nearest spatial neighbours.  Genes with
strong spatial structure score highly; housekeeping genes and noise score low.

Spatial score
-------------
  smooth[i]  = mean expression of k nearest neighbours of spot i
  pearson    = Pearson r between y (raw) and smooth (KNN-averaged)
  score      = pearson * log1p(variance) * nonzero_fraction

The variance and nonzero-fraction terms down-weight genes that are very
low-expressed or expressed in almost no spots, even if they show local
smoothness in the spots where they are expressed.

Usage:
    python scripts/select_spatial_genes.py \\
        --input data/processed/breast_tma_hd_0_square016.parquet \\
        --top-n 30 --plot

    python scripts/select_spatial_genes.py \\
        --input data/processed/breast_tma_hd_0_square016.parquet \\
        --top-n 50 \\
        --output outputs/spatial_genes/breast_tma_spatial_genes.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Columns that are never gene columns in a MALACHY table or predictions CSV.
_NON_GENE_COLS = {"section_id", "x", "y", "z", "split", "method", "gene", "true", "pred"}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rank genes by spatial autocorrelation (KNN-smooth Pearson).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True,
                   help="MALACHY spot table (.parquet or .csv).")
    p.add_argument("--output",
                   help="Output CSV path.  "
                        "Default: outputs/spatial_genes/{stem}_spatial_genes.csv")
    p.add_argument("--top-n", type=int, default=100,
                   help="Number of top genes to write to the companion .txt file.")
    p.add_argument("--k", type=int, default=20,
                   help="Number of spatial nearest neighbours for smoothing.")
    p.add_argument("--min-mean", type=float, default=0.01,
                   help="Minimum mean expression; genes below this are excluded.")
    p.add_argument("--max-genes", type=int, default=None,
                   help="Limit to first N gene columns (for quick debugging).")
    p.add_argument("--plot", action="store_true",
                   help="Save a bar chart of the top 30 genes to "
                        "outputs/figures/spatial_gene_ranking.png.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in (".csv", ".tsv"):
        return pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    sys.exit(f"Unsupported format: {path.suffix}")


def resolve_output_path(args: argparse.Namespace, input_path: Path) -> Path:
    if args.output:
        return Path(args.output)
    stem = input_path.stem
    return Path("outputs/spatial_genes") / f"{stem}_spatial_genes.csv"


# ---------------------------------------------------------------------------
# Core: KNN graph + per-gene spatial scoring
# ---------------------------------------------------------------------------

def build_knn_smooth_matrix(
    coords: np.ndarray,
    k: int,
) -> "csr_matrix":
    """Return a (N, N) sparse row-stochastic matrix that averages k neighbours.

    Excludes self from the neighbourhood (k+1 query, drop nearest).
    """
    N = coords.shape[0]
    log.info("Building KNN graph (N=%d, k=%d) …", N, k)
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)   # (N, k+1); first col is self
    idx = idx[:, 1:]                        # (N, k) — exclude self

    row = np.repeat(np.arange(N), k)
    col = idx.ravel()
    data = np.ones(N * k, dtype=np.float32) / k
    return csr_matrix((data, (row, col)), shape=(N, N), dtype=np.float32)


def score_genes(
    expr: np.ndarray,       # (N, G)  float32
    W: "csr_matrix",        # (N, N)  sparse
    gene_cols: list[str],
) -> pd.DataFrame:
    """Compute spatial scores for all genes simultaneously."""
    log.info("Smoothing expression with KNN weights …")
    smooth = W @ expr                       # (N, G)

    log.info("Computing per-gene Pearson correlations …")
    e = expr  - expr.mean(axis=0)
    s = smooth - smooth.mean(axis=0)

    num   = (e * s).sum(axis=0)
    denom = np.sqrt((e ** 2).sum(axis=0) * (s ** 2).sum(axis=0))
    pearson = np.where(denom > 1e-10, num / denom, 0.0)

    variance      = expr.var(axis=0)
    means         = expr.mean(axis=0)
    nonzero_frac  = (expr > 0).mean(axis=0)

    spatial_score = pearson * np.log1p(variance) * nonzero_frac

    return pd.DataFrame({
        "gene":               gene_cols,
        "spatial_score":      spatial_score,
        "pearson_knn_smooth": pearson,
        "variance":           variance,
        "mean":               means,
        "nonzero_fraction":   nonzero_frac,
    }).sort_values("spatial_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(
    results: pd.DataFrame,
    csv_path: Path,
    top_n: int,
) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(csv_path, index=False)
    log.info("Ranking saved → %s  (%d genes)", csv_path, len(results))

    txt_path = csv_path.with_suffix(".txt")
    top_genes = results.head(top_n)["gene"].tolist()
    txt_path.write_text(",".join(top_genes) + "\n")
    log.info("Top-%d gene list → %s", top_n, txt_path)
    return txt_path


def make_plot(results: pd.DataFrame, save_path: Path, n: int = 30) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = results.head(n)
    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.28)))

    genes  = top["gene"].tolist()[::-1]
    scores = top["spatial_score"].tolist()[::-1]
    colors = ["#e05c5c" if s < 0 else "#3a86c8" for s in scores]

    ax.barh(genes, scores, color=colors, edgecolor="none", height=0.7)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Spatial score  (Pearson × log1p(var) × nonzero_frac)")
    ax.set_title(f"Top {n} spatially variable genes")
    ax.tick_params(axis="y", labelsize=8)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Plot saved → %s", save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Input not found: {input_path}")

    # ── Load ──────────────────────────────────────────────────────────────
    df = load_table(input_path)
    log.info("Loaded %d rows × %d columns from %s", len(df), len(df.columns), input_path.name)

    # If multiple sections, score on first section only.
    if "section_id" in df.columns:
        sections = df["section_id"].unique()
        if len(sections) > 1:
            log.warning(
                "Table has %d sections; scoring spatial structure on '%s' only.",
                len(sections), sections[0],
            )
            df = df[df["section_id"] == sections[0]].reset_index(drop=True)

    gene_cols = [c for c in df.columns if c not in _NON_GENE_COLS]
    if args.max_genes:
        gene_cols = gene_cols[: args.max_genes]
        log.info("--max-genes: limiting to %d genes.", args.max_genes)
    log.info("%d gene columns identified.", len(gene_cols))

    # ── Filter low-expressed genes ────────────────────────────────────────
    expr_all = df[gene_cols].values.astype(np.float32)   # (N, G)
    means = expr_all.mean(axis=0)
    keep = means >= args.min_mean
    n_dropped = (~keep).sum()
    if n_dropped:
        log.info("Dropped %d genes with mean < %.4f  (%d remain).",
                 n_dropped, args.min_mean, keep.sum())
    gene_cols = [g for g, k in zip(gene_cols, keep) if k]
    expr = expr_all[:, keep]

    if len(gene_cols) == 0:
        sys.exit("No genes passed the --min-mean filter.")

    # ── Build KNN graph ───────────────────────────────────────────────────
    coords = df[["x", "y"]].values.astype(np.float64)
    W = build_knn_smooth_matrix(coords, k=args.k)

    # ── Score genes ───────────────────────────────────────────────────────
    results = score_genes(expr, W, gene_cols)

    # ── Print top 20 ──────────────────────────────────────────────────────
    print()
    print("─" * 70)
    print(f"  Top 20 spatially variable genes")
    print(f"  {'Gene':<20}  {'Score':>8}  {'Pearson':>8}  {'Var':>8}  {'Nz%':>6}")
    print("─" * 70)
    for _, row in results.head(20).iterrows():
        print(
            f"  {row['gene']:<20}  {row['spatial_score']:>8.4f}  "
            f"{row['pearson_knn_smooth']:>8.4f}  "
            f"{row['variance']:>8.4f}  "
            f"{100*row['nonzero_fraction']:>5.1f}%"
        )
    print("─" * 70)
    print()

    # ── Save ──────────────────────────────────────────────────────────────
    csv_path = resolve_output_path(args, input_path)
    txt_path = save_results(results, csv_path, top_n=args.top_n)

    if args.plot:
        make_plot(results, save_path=Path("outputs/figures/spatial_gene_ranking.png"))

    # ── Hint for next step ────────────────────────────────────────────────
    top_genes_preview = ",".join(results.head(6)["gene"].tolist())
    print(f"  Pass top genes to train_2d_holdout.py with:")
    print(f"    --plot-genes {top_genes_preview}")
    print(f"  Or read the full list from: {txt_path}")
    print()


if __name__ == "__main__":
    main()
