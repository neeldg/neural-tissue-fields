"""Train the CoordinateMLP on a single-section 2D table (x, y → genes).

This is a real-data sanity check before 3D serial-section training.
The model uses only x and y; z is ignored.  A held-out spatial region
tests whether the neural field has learned the underlying expression landscape
or merely memorised training points.

Three hold-out modes:
  random   – uniformly random fraction of spots
  stripe   – central vertical stripe (~holdout_fraction of spots by x quantile)
  quadrant – upper-right quadrant (x > median AND y > median, ≈25 % of spots)

Usage:
    python scripts/train_2d_holdout.py \\
        --input data/processed/patient01_A1_visium.parquet \\
        --holdout-mode stripe \\
        --output-prefix patient01_A1_stripe

    python scripts/train_2d_holdout.py \\
        --input data/processed/synthetic_sections.csv \\
        --holdout-mode random --holdout-fraction 0.2 --epochs 200 \\
        --output-prefix synth_2d_random \\
        --plot-genes gene_0,gene_4,gene_12,gene_14
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import SpatialExpressionDataset
from src.eval.baselines import knn_xyz_baseline
from src.eval.metrics import compute_metrics, summarize_metrics
from src.models.mlp_field import CoordinateMLP
from src.training.train import train as train_model
from src.viz.plot_gene_maps import plot_method_comparison

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

META_COLS = {"section_id", "x", "y", "z"}
COORD_COLS = ["x", "y"]   # this script always uses 2-D coordinates


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="2-D spatial holdout training and evaluation for MALACHY.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True,
                   help="Processed spot table (.parquet or .csv).")
    p.add_argument("--holdout-mode", choices=["random", "stripe", "quadrant"],
                   default="random",
                   help="Spatial holdout strategy.")
    p.add_argument("--holdout-fraction", type=float, default=0.2,
                   help="Fraction of spots to hold out.  "
                        "For 'quadrant' mode the fraction is always ~0.25.")
    p.add_argument("--epochs", type=int, default=100,
                   help="Number of training epochs.")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dims", type=str, default="256,256,256",
                   help="Comma-separated MLP hidden layer widths.")
    p.add_argument("--n-freqs", type=int, default=6,
                   help="Positional encoding frequency bands.")
    p.add_argument("--knn-k", type=int, default=10,
                   help="Number of neighbours for the KNN baseline.")
    p.add_argument("--output-prefix", required=True,
                   help="Prefix for all output filenames.")
    p.add_argument("--plot-genes",
                   help="Comma-separated gene names to plot (default: first 4).")
    p.add_argument("--device", default="auto",
                   help="Torch device: 'auto', 'cpu', 'cuda', or 'mps'.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def resolve_device(device_str: str) -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix in (".csv", ".tsv"):
        return pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    else:
        sys.exit(f"Unsupported file format: {path.suffix}")


def extract_gene_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


# ---------------------------------------------------------------------------
# Hold-out splitting
# ---------------------------------------------------------------------------

def split_random(df: pd.DataFrame, fraction: float, rng: np.random.Generator
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_test = max(1, int(len(df) * fraction))
    test_idx = rng.choice(len(df), size=n_test, replace=False)
    mask = np.zeros(len(df), dtype=bool)
    mask[test_idx] = True
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def split_stripe(df: pd.DataFrame, fraction: float
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out a central vertical stripe spanning the middle `fraction` of x."""
    lo_q = 0.5 - fraction / 2
    hi_q = 0.5 + fraction / 2
    x_lo = float(np.quantile(df["x"], lo_q))
    x_hi = float(np.quantile(df["x"], hi_q))
    mask = (df["x"] >= x_lo) & (df["x"] <= x_hi)
    log.info(
        "Stripe holdout: x ∈ [%.2f, %.2f]  (%d / %d spots held out)",
        x_lo, x_hi, mask.sum(), len(df),
    )
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def split_quadrant(df: pd.DataFrame, fraction: float
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the upper-right quadrant (x > median AND y > median, ≈25 %)."""
    if abs(fraction - 0.25) > 0.05:
        log.warning(
            "Quadrant mode always holds out ~25%% of spots; "
            "--holdout-fraction %.2f is ignored.", fraction,
        )
    x_mid = float(df["x"].median())
    y_mid = float(df["y"].median())
    mask = (df["x"] > x_mid) & (df["y"] > y_mid)
    log.info(
        "Quadrant holdout: x > %.2f, y > %.2f  (%d / %d spots held out)",
        x_mid, y_mid, mask.sum(), len(df),
    )
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def split(df: pd.DataFrame, mode: str, fraction: float,
          rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mode == "random":
        return split_random(df, fraction, rng)
    if mode == "stripe":
        return split_stripe(df, fraction)
    if mode == "quadrant":
        return split_quadrant(df, fraction)
    raise ValueError(f"Unknown holdout mode: {mode!r}")


# ---------------------------------------------------------------------------
# Neural field training + inference
# ---------------------------------------------------------------------------

def run_neural_field(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    gene_cols: list[str],
    *,
    hidden_dims: list[int],
    n_freqs: int,
    n_epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> np.ndarray:
    """Train the CoordinateMLP on (x, y) and return (N_test, G) predictions."""
    train_ds = SpatialExpressionDataset(df_train, COORD_COLS, gene_cols)
    test_ds = SpatialExpressionDataset(
        df_test, COORD_COLS, gene_cols,
        coord_bounds=train_ds.coord_bounds,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
    )

    model = CoordinateMLP(
        n_genes=len(gene_cols),
        coord_dim=2,
        hidden_dims=hidden_dims,
        use_positional_encoding=True,
        n_freqs=n_freqs,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model: %d trainable parameters", n_params)

    log_every = max(1, n_epochs // 10)
    train_model(
        model=model,
        train_loader=train_loader,
        n_epochs=n_epochs,
        lr=lr,
        device=device,
        log_every=log_every,
    )

    model.eval()
    model = model.to(device)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 4, shuffle=False)
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for coords, _ in test_loader:
            chunks.append(model(coords.to(device)).cpu())

    return torch.cat(chunks, dim=0).numpy()   # (N_test, G)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_predictions(
    df_test: pd.DataFrame,
    gene_cols: list[str],
    nf_preds: np.ndarray,
    knn_preds: np.ndarray,
    path: Path,
) -> None:
    out = df_test[["x", "y"]].copy()
    if "section_id" in df_test.columns:
        out.insert(0, "section_id", df_test["section_id"].values)
    out[gene_cols] = df_test[gene_cols].values
    for g, col in enumerate(gene_cols):
        out[f"{col}_nf_pred"]  = nf_preds[:, g]
        out[f"{col}_knn_pred"] = knn_preds[:, g]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    log.info("Predictions saved → %s", path)


def save_metrics(
    gene_cols: list[str],
    true_arr: np.ndarray,
    nf_preds: np.ndarray,
    knn_preds: np.ndarray,
    path: Path,
) -> pd.DataFrame:
    frames = []
    for method, preds in [("neural_field", nf_preds), ("knn", knn_preds)]:
        m = compute_metrics(true_arr, preds, gene_names=gene_cols)
        summary = summarize_metrics(m)
        log.info(
            "%-14s  MSE=%.4f  MAE=%.4f  Pearson_r=%.4f",
            method, summary["mse"], summary["mae"], summary["pearson_r"],
        )
        m = m.reset_index().rename(columns={"index": "gene"})
        m.insert(0, "method", method)
        frames.append(m)
    combined = pd.concat(frames, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    log.info("Metrics saved → %s", path)
    return combined


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figure(
    df_test: pd.DataFrame,
    gene_cols: list[str],
    plot_genes: list[str],
    nf_preds: np.ndarray,
    knn_preds: np.ndarray,
    save_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    g_idx = [gene_cols.index(g) for g in plot_genes]

    fig = plot_method_comparison(
        x=df_test["x"].values,
        y=df_test["y"].values,
        true_expr=df_test[plot_genes].values,
        method_preds=[
            ("KNN",          knn_preds[:, g_idx]),
            ("Neural field", nf_preds[:, g_idx]),
        ],
        gene_names=plot_genes,
        x_label="x",
        y_label="y",
        save_path=save_path,
    )
    plt.close(fig)
    log.info("Figure saved → %s", save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    log.info("Device: %s", device)

    # ── Load ──────────────────────────────────────────────────────────────
    path = Path(args.input)
    if not path.exists():
        sys.exit(f"Input file not found: {path}")
    df = load_table(path)
    log.info("Loaded %d rows × %d columns from %s", len(df), len(df.columns), path.name)

    # If the table has multiple sections, use only the first one.
    if "section_id" in df.columns:
        sections = df["section_id"].unique()
        if len(sections) > 1:
            log.warning(
                "Table contains %d sections; using only '%s'. "
                "For multi-section training use train_v0.py.",
                len(sections), sections[0],
            )
            df = df[df["section_id"] == sections[0]].reset_index(drop=True)

    gene_cols = extract_gene_cols(df)
    if not gene_cols:
        sys.exit("No gene columns found. Expected all columns except section_id, x, y, z.")
    log.info("%d gene columns identified.", len(gene_cols))

    # ── Split ─────────────────────────────────────────────────────────────
    df_train, df_test = split(df, args.holdout_mode, args.holdout_fraction, rng)
    log.info(
        "Split [%s]: train=%d  test=%d  (%.1f%% held out)",
        args.holdout_mode, len(df_train), len(df_test),
        100 * len(df_test) / len(df),
    )

    true_arr = df_test[gene_cols].values.astype(np.float32)

    # ── Neural field ──────────────────────────────────────────────────────
    hidden_dims = [int(w) for w in args.hidden_dims.split(",")]
    log.info("Training neural field …")
    nf_preds = run_neural_field(
        df_train, df_test, gene_cols,
        hidden_dims=hidden_dims,
        n_freqs=args.n_freqs,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    # ── KNN baseline ──────────────────────────────────────────────────────
    log.info("Computing KNN baseline (k=%d) …", args.knn_k)
    knn_preds = knn_xyz_baseline(
        df_train, df_test, COORD_COLS, gene_cols, k=args.knn_k,
    )

    # ── Save ──────────────────────────────────────────────────────────────
    pred_dir = Path("outputs/predictions")
    fig_dir  = Path("outputs/figures")
    prefix   = args.output_prefix

    save_predictions(
        df_test, gene_cols, nf_preds, knn_preds,
        path=pred_dir / f"{prefix}_predictions.csv",
    )
    save_metrics(
        gene_cols, true_arr, nf_preds, knn_preds,
        path=pred_dir / f"{prefix}_metrics.csv",
    )

    # ── Figure ────────────────────────────────────────────────────────────
    if args.plot_genes:
        plot_genes = [g.strip() for g in args.plot_genes.split(",") if g.strip()]
        missing = [g for g in plot_genes if g not in gene_cols]
        if missing:
            log.warning("Requested plot genes not found and will be skipped: %s", missing)
            plot_genes = [g for g in plot_genes if g in gene_cols]
    else:
        plot_genes = gene_cols[:4]

    if plot_genes:
        make_figure(
            df_test, gene_cols, plot_genes, nf_preds, knn_preds,
            save_path=fig_dir / f"{prefix}_gene_maps.png",
        )

    log.info("Done.")


if __name__ == "__main__":
    main()
