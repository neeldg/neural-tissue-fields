"""End-to-end training script: hold out one section, train, predict, save.

Usage:
    python scripts/train_v0.py --config configs/visium_v0.yaml

Override any config key on the command line with dot-notation, e.g.:
    python scripts/train_v0.py --config configs/visium_v0.yaml \\
        training.n_epochs=50 training.device=cuda
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

# Make sure the repo root is on sys.path when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import SpatialExpressionDataset
from src.eval.metrics import compute_metrics, summarize_metrics
from src.models.mlp_field import CoordinateMLP
from src.training.train import train
from src.viz.plot_gene_maps import plot_gene_maps, plot_pearson_barplot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply 'key.subkey=value' overrides from the CLI to a nested config dict."""
    for item in overrides:
        key_path, _, raw_value = item.partition("=")
        keys = key_path.strip().split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        # Try to cast to int/float/bool, otherwise keep as string.
        for cast in (int, float):
            try:
                raw_value = cast(raw_value)
                break
            except ValueError:
                pass
        if raw_value in ("true", "True"):
            raw_value = True
        elif raw_value in ("false", "False"):
            raw_value = False
        node[keys[-1]] = raw_value
    return cfg


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataframe(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    elif p.suffix in (".csv", ".tsv"):
        sep = "\t" if p.suffix == ".tsv" else ","
        return pd.read_csv(p, sep=sep)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}. Use .parquet or .csv.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train neural field on spatial transcriptomics.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, overrides)

    # ---- Data ----
    log.info("Loading data from %s", cfg["data"]["input_path"])
    df = load_dataframe(cfg["data"]["input_path"])
    log.info("Loaded %d spots, %d columns", len(df), len(df.columns))

    coord_cols: list[str] = cfg["data"]["coord_cols"]
    section_col: str = cfg["data"]["section_col"]
    held_out: str = str(cfg["data"]["held_out_section"])

    gene_cols: list[str] = cfg["data"].get("gene_cols") or []
    if not gene_cols:
        # Use all columns that are not coords or the section column.
        skip = set(coord_cols) | {section_col}
        gene_cols = [c for c in df.columns if c not in skip]
        log.info("No gene_cols specified; using all %d gene columns.", len(gene_cols))

    # Split
    mask_heldout = df[section_col].astype(str) == held_out
    df_train = df[~mask_heldout].reset_index(drop=True)
    df_test = df[mask_heldout].reset_index(drop=True)
    log.info(
        "Train: %d spots  |  Held-out (%s): %d spots",
        len(df_train), held_out, len(df_test),
    )

    # Build datasets (fit normalisation bounds on training data only).
    train_dataset = SpatialExpressionDataset(df_train, coord_cols, gene_cols)
    test_dataset = SpatialExpressionDataset(
        df_test, coord_cols, gene_cols,
        coord_bounds=train_dataset.coord_bounds,  # reuse training bounds
    )

    batch_size: int = cfg["data"].get("batch_size", 1024)
    num_workers: int = cfg["data"].get("num_workers", 0)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
    )

    # ---- Model ----
    model_cfg = cfg["model"]
    model = CoordinateMLP(
        n_genes=len(gene_cols),
        coord_dim=len(coord_cols),
        hidden_dims=model_cfg.get("hidden_dims", [256, 256, 256]),
        use_positional_encoding=model_cfg.get("use_positional_encoding", True),
        n_freqs=model_cfg.get("n_freqs", 6),
        dropout=model_cfg.get("dropout", 0.0),
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model: %d trainable parameters", n_params)

    # ---- Train ----
    train_cfg = cfg["training"]
    out_cfg = cfg["output"]

    ckpt_dir = Path(out_cfg["checkpoint_dir"])
    ckpt_path = ckpt_dir / out_cfg["checkpoint_name"]

    history = train(
        model=model,
        train_loader=train_loader,
        n_epochs=train_cfg.get("n_epochs", 100),
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
        checkpoint_path=ckpt_path,
        device=train_cfg.get("device", "cpu"),
        log_every=train_cfg.get("log_every", 10),
    )
    log.info("Final train loss: %.6f", history["train_loss"][-1])

    # ---- Predict on held-out section ----
    device = torch.device(train_cfg.get("device", "cpu"))
    model.eval()
    model = model.to(device)

    all_preds = []
    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size * 4, shuffle=False
        )
        for coords, _ in test_loader:
            coords = coords.to(device)
            preds = model(coords).cpu()
            all_preds.append(preds)

    preds_tensor = torch.cat(all_preds, dim=0)  # (N_test, G)
    true_tensor = test_dataset.exprs             # (N_test, G)

    # ---- Metrics ----
    metrics_df = compute_metrics(true_tensor, preds_tensor, gene_names=gene_cols)
    summary = summarize_metrics(metrics_df)
    log.info(
        "Held-out metrics  MSE=%.4f  MAE=%.4f  Pearson_r=%.4f",
        summary["mse"], summary["mae"], summary["pearson_r"],
    )

    # ---- Save predictions ----
    pred_dir = Path(out_cfg["predictions_dir"])
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / out_cfg["predictions_name"]

    pred_cols = [f"{g}_pred" for g in gene_cols]
    pred_df = df_test[coord_cols + [section_col]].copy()
    pred_df[gene_cols] = true_tensor.numpy()
    pred_df[pred_cols] = preds_tensor.numpy()
    try:
        pred_df.to_parquet(pred_path, index=False)
    except ImportError:
        pred_path = pred_path.with_suffix(".csv")
        pred_df.to_csv(pred_path, index=False)
    log.info("Predictions saved → %s", pred_path)

    # Save metrics CSV alongside predictions.
    metrics_path = pred_dir / (Path(out_cfg["predictions_name"]).stem + "_metrics.csv")
    metrics_df.to_csv(metrics_path)
    log.info("Metrics saved → %s", metrics_path)

    # ---- Figures ----
    fig_dir = Path(out_cfg["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_genes: list[str] = out_cfg.get("plot_genes") or gene_cols[:4]
    plot_genes = [g for g in plot_genes if g in gene_cols]  # guard against typos

    if plot_genes:
        pred_cols_to_plot = [f"{g}_pred" for g in plot_genes]
        fig = plot_gene_maps(
            df=pred_df,
            gene_names=plot_genes,
            true_cols=plot_genes,
            pred_cols=pred_cols_to_plot,
            x_col=coord_cols[0],
            y_col=coord_cols[1],
            save_path=fig_dir / "gene_maps.png",
        )
        import matplotlib.pyplot as plt
        plt.close(fig)
        log.info("Gene maps saved → %s/gene_maps.png", fig_dir)

    fig2 = plot_pearson_barplot(
        metrics_df["pearson_r"],
        save_path=fig_dir / "pearson_r.png",
    )
    import matplotlib.pyplot as plt
    plt.close(fig2)
    log.info("Pearson barplot saved → %s/pearson_r.png", fig_dir)

    log.info("Done.")


if __name__ == "__main__":
    main()
