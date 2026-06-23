"""MALACHY v1 — 2D spatial holdout training and evaluation.

Train a coordinate neural field on one Visium HD section using (x, y) coordinates
and evaluate held-out spatial prediction against a KNN baseline.

Three hold-out modes:
  random   – uniformly random fraction of spots
  stripe   – central vertical stripe (~holdout_fraction of the x range)
  quadrant – upper-right quadrant (x > median AND y > median, ≈25% of spots)

Two model options:
  mlp        – CoordinateMLP with sinusoidal Fourier encoding
  gridfield  – multi-resolution learnable grid encoding + MLP decoder

Usage:
    python scripts/train_2d_holdout.py \\
        --input data/processed/breast_tma_hd_0_square016.parquet \\
        --holdout-mode stripe \\
        --model mlp \\
        --output-prefix breast_tma_stripe_mlp

    python scripts/train_2d_holdout.py \\
        --input data/processed/breast_tma_hd_0_square016.parquet \\
        --holdout-mode quadrant \\
        --model gridfield \\
        --output-prefix breast_tma_quadrant_grid \\
        --plot-genes MALAT1,EPCAM,COL1A1
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import SpatialExpressionDataset
from src.eval.baselines import knn_xyz_baseline
from src.eval.metrics import compute_metrics, summarize_metrics
from src.models.coordinate_encodings import GridField
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
COORD_COLS = ["x", "y"]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MALACHY v1: 2D spatial holdout training and evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True,
                   help="Processed spot table (.parquet or .csv).")
    p.add_argument("--holdout-mode", choices=["random", "stripe", "quadrant"],
                   default="random",
                   help="Spatial holdout strategy.")
    p.add_argument("--holdout-fraction", type=float, default=0.2,
                   help="Fraction of spots to hold out "
                        "(quadrant mode is always ~25%%).")
    p.add_argument("--model", choices=["mlp", "gridfield"], default="mlp",
                   help="Model architecture: 'mlp' (Fourier-encoded MLP) or "
                        "'gridfield' (multi-resolution grid + MLP).")
    p.add_argument("--epochs", type=int, default=50,
                   help="Training epochs.")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-prefix", required=True,
                   help="Prefix for all output filenames.")
    p.add_argument("--gene-list",
                   help="Path to a text file with genes to train on — "
                        "comma-separated on one line or one gene per line "
                        "(e.g. output of select_spatial_genes.py).  "
                        "Genes absent from the input table are skipped with a warning.")
    p.add_argument("--plot-genes",
                   help="Comma-separated gene names to visualise (default: first 4).")
    p.add_argument("--device", default="auto",
                   help="Torch device: 'auto', 'cpu', 'cuda', or 'mps'.")
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
    if path.suffix in (".csv", ".tsv"):
        return pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    sys.exit(f"Unsupported file format: {path.suffix}")


def extract_gene_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def load_gene_list(path: str) -> list[str]:
    """Parse a gene-list file: comma-separated or one gene per line."""
    text = Path(path).read_text().strip()
    # Try comma-separated first; fall back to newline-separated.
    if "," in text:
        genes = [g.strip() for g in text.split(",") if g.strip()]
    else:
        genes = [line.strip() for line in text.splitlines()
                 if line.strip() and not line.startswith("#")]
    return genes


# ---------------------------------------------------------------------------
# Hold-out splitting
# ---------------------------------------------------------------------------

def split_random(df: pd.DataFrame, fraction: float,
                 rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_test = max(1, int(len(df) * fraction))
    test_idx = rng.choice(len(df), size=n_test, replace=False)
    mask = np.zeros(len(df), dtype=bool)
    mask[test_idx] = True
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def split_stripe(df: pd.DataFrame,
                 fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    lo_q = 0.5 - fraction / 2
    hi_q = 0.5 + fraction / 2
    x_lo = float(np.quantile(df["x"], lo_q))
    x_hi = float(np.quantile(df["x"], hi_q))
    mask = (df["x"] >= x_lo) & (df["x"] <= x_hi)
    log.info("Stripe holdout: x ∈ [%.1f, %.1f]  (%d / %d spots)",
             x_lo, x_hi, mask.sum(), len(df))
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def split_quadrant(df: pd.DataFrame,
                   fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if abs(fraction - 0.25) > 0.05:
        log.warning("Quadrant mode holds out ~25%% of spots; "
                    "--holdout-fraction %.2f is ignored.", fraction)
    x_mid = float(df["x"].median())
    y_mid = float(df["y"].median())
    mask = (df["x"] > x_mid) & (df["y"] > y_mid)
    log.info("Quadrant holdout: x > %.1f, y > %.1f  (%d / %d spots)",
             x_mid, y_mid, mask.sum(), len(df))
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
# Model construction
# ---------------------------------------------------------------------------

def build_model(n_genes: int, model_name: str) -> nn.Module:
    if model_name == "mlp":
        return CoordinateMLP(
            n_genes=n_genes,
            coord_dim=2,
            hidden_dims=[256, 256, 256],
            use_positional_encoding=True,
            n_freqs=6,
        )
    return GridField(
        n_genes=n_genes,
        resolutions=(16, 32, 64, 128),
        n_features=8,
        hidden_dims=(256, 256, 256),
    )


# ---------------------------------------------------------------------------
# Neural field training + inference
# ---------------------------------------------------------------------------

def run_neural_field(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    gene_cols: list[str],
    model: nn.Module,
    *,
    n_epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> np.ndarray:
    """Train model on (x, y) → expression and return (N_test, G) predictions."""
    train_ds = SpatialExpressionDataset(df_train, COORD_COLS, gene_cols)
    test_ds = SpatialExpressionDataset(
        df_test, COORD_COLS, gene_cols,
        coord_bounds=train_ds.coord_bounds,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model: %s  |  %d trainable parameters", type(model).__name__, n_params)

    train_model(
        model=model,
        train_loader=train_loader,
        n_epochs=n_epochs,
        lr=lr,
        device=device,
        log_every=max(1, n_epochs // 10),
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
    preds_by_method: dict[str, np.ndarray],
    path: Path,
) -> None:
    """Save predictions in long format: one row per (spot, gene, method)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_spots = len(df_test)
    n_genes = len(gene_cols)

    xs   = df_test["x"].values
    ys   = df_test["y"].values
    sids = (df_test["section_id"].values if "section_id" in df_test.columns
            else np.full(n_spots, ""))
    true_vals = df_test[gene_cols].values.astype(np.float32)   # (N, G)

    frames = []
    for method, preds in preds_by_method.items():
        frames.append(pd.DataFrame({
            "x":          np.tile(xs,        n_genes),
            "y":          np.tile(ys,        n_genes),
            "section_id": np.tile(sids,      n_genes),
            "split":      "test",
            "method":     method,
            "gene":       np.repeat(gene_cols, n_spots),
            "true":       true_vals.T.ravel(),
            "pred":       preds.astype(np.float32).T.ravel(),
        }))

    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    log.info("Predictions saved → %s  (%d rows)",
             path, n_spots * n_genes * len(preds_by_method))


def save_metrics(
    gene_cols: list[str],
    true_arr: np.ndarray,
    preds_by_method: dict[str, np.ndarray],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for method, preds in preds_by_method.items():
        m = compute_metrics(true_arr, preds, gene_names=gene_cols)
        summary = summarize_metrics(m)
        log.info("%-14s  MSE=%.4f  MAE=%.4f  Pearson_r=%.4f",
                 method, summary["mse"], summary["mae"], summary["pearson_r"])
        m = m.reset_index().rename(columns={"index": "gene"})
        m.insert(0, "method", method)
        frames.append(m)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    log.info("Metrics saved → %s", path)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(
    df_test: pd.DataFrame,
    gene_cols: list[str],
    plot_genes: list[str],
    preds_by_method: dict[str, np.ndarray],
    save_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    save_path.parent.mkdir(parents=True, exist_ok=True)
    g_idx = [gene_cols.index(g) for g in plot_genes]
    method_preds = [(label, arr[:, g_idx]) for label, arr in preds_by_method.items()]

    fig = plot_method_comparison(
        x=df_test["x"].values,
        y=df_test["y"].values,
        true_expr=df_test[plot_genes].values,
        method_preds=method_preds,
        gene_names=plot_genes,
        x_label="x (pixels)",
        y_label="y (pixels)",
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

    if "section_id" in df.columns:
        sections = df["section_id"].unique()
        if len(sections) > 1:
            log.warning("Table has %d sections; using only '%s'. "
                        "Use train_v0.py for multi-section 3D training.",
                        len(sections), sections[0])
            df = df[df["section_id"] == sections[0]].reset_index(drop=True)

    gene_cols = extract_gene_cols(df)
    if not gene_cols:
        sys.exit("No gene columns found (expected all columns except section_id, x, y, z).")
    log.info("%d gene columns in table", len(gene_cols))

    # ── Gene-list filter ──────────────────────────────────────────────────
    if args.gene_list:
        if not Path(args.gene_list).exists():
            sys.exit(f"Gene list file not found: {args.gene_list}")
        requested = load_gene_list(args.gene_list)
        available = set(gene_cols)
        missing = [g for g in requested if g not in available]
        if missing:
            log.warning("Genes in --gene-list not found in table (%d): %s",
                        len(missing), missing)
        gene_cols = [g for g in requested if g in available]
        if not gene_cols:
            sys.exit("No requested genes are present in the input table.")
        preview = ", ".join(gene_cols[:10])
        suffix = f" … (+{len(gene_cols)-10} more)" if len(gene_cols) > 10 else ""
        log.info("Using %d genes from --gene-list: %s%s", len(gene_cols), preview, suffix)

    # ── Split ─────────────────────────────────────────────────────────────
    df_train, df_test = split(df, args.holdout_mode, args.holdout_fraction, rng)
    log.info("Split [%s]: train=%d  test=%d  (%.1f%% held out)",
             args.holdout_mode, len(df_train), len(df_test),
             100 * len(df_test) / len(df))

    true_arr = df_test[gene_cols].values.astype(np.float32)

    # ── Neural field ──────────────────────────────────────────────────────
    model = build_model(len(gene_cols), args.model)

    # F.grid_sample backward is not implemented on MPS; fall back to CPU.
    train_device = device
    if device.type == "mps" and isinstance(model, GridField):
        log.warning("grid_sampler_2d_backward is not implemented on MPS — "
                    "falling back to CPU for GridField training.")
        train_device = torch.device("cpu")

    log.info("Training %s for %d epochs …", type(model).__name__, args.epochs)
    nf_preds = run_neural_field(
        df_train, df_test, gene_cols, model,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=train_device,
    )

    # ── KNN baseline ──────────────────────────────────────────────────────
    log.info("Computing KNN baseline (k=10) …")
    knn_preds = knn_xyz_baseline(df_train, df_test, COORD_COLS, gene_cols, k=10)

    preds_by_method = {
        "knn":           knn_preds,
        "neural_field":  nf_preds,
    }

    # ── Save ──────────────────────────────────────────────────────────────
    prefix   = args.output_prefix
    pred_dir = Path("outputs/predictions")
    fig_dir  = Path("outputs/figures")

    save_predictions(
        df_test, gene_cols, preds_by_method,
        path=pred_dir / f"{prefix}_predictions.csv",
    )
    save_metrics(
        gene_cols, true_arr, preds_by_method,
        path=pred_dir / f"{prefix}_metrics.csv",
    )

    # ── Figure ────────────────────────────────────────────────────────────
    if args.plot_genes:
        plot_genes = [g.strip() for g in args.plot_genes.split(",") if g.strip()]
        missing = [g for g in plot_genes if g not in gene_cols]
        if missing:
            log.warning("Plot genes not in table, skipping: %s", missing)
            plot_genes = [g for g in plot_genes if g in gene_cols]
    else:
        plot_genes = gene_cols[:4]

    if plot_genes:
        make_figure(
            df_test, gene_cols, plot_genes, preds_by_method,
            save_path=fig_dir / f"{prefix}_gene_maps.png",
        )

    log.info("Done.  Outputs written with prefix '%s'.", prefix)


if __name__ == "__main__":
    main()
