"""End-to-end training script: hold out one section, train, predict, save.

Produces per-method prediction CSVs and a combined metrics file.

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

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import SpatialExpressionDataset
from src.eval.baselines import (
    knn_xyz_baseline,
    linear_z_interpolation_baseline,
    nearest_section_baseline,
)
from src.eval.metrics import compute_metrics, summarize_metrics
from src.models.mlp_field import CoordinateMLP
from src.training.train import train
from src.viz.plot_gene_maps import plot_gene_maps, plot_method_comparison, plot_pearson_barplot

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
# Saving helpers
# ---------------------------------------------------------------------------

def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_predictions(
    df_test: pd.DataFrame,
    coord_cols: list[str],
    section_col: str,
    gene_cols: list[str],
    pred_array: np.ndarray,
    path: Path,
) -> None:
    """Write a prediction CSV with metadata columns + true + predicted expression."""
    pred_cols = [f"{g}_pred" for g in gene_cols]
    out = df_test[coord_cols + [section_col]].copy()
    out[gene_cols] = df_test[gene_cols].values
    out[pred_cols] = pred_array
    save_csv(out, path)
    log.info("Predictions saved → %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train neural field on spatial transcriptomics."
    )
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
        skip = set(coord_cols) | {section_col}
        gene_cols = [c for c in df.columns if c not in skip]
        log.info("No gene_cols specified; using all %d gene columns.", len(gene_cols))

    mask_heldout = df[section_col].astype(str) == held_out
    df_train = df[~mask_heldout].reset_index(drop=True)
    df_test = df[mask_heldout].reset_index(drop=True)
    log.info(
        "Train: %d spots  |  Held-out (%s): %d spots",
        len(df_train), held_out, len(df_test),
    )

    train_dataset = SpatialExpressionDataset(df_train, coord_cols, gene_cols)
    test_dataset = SpatialExpressionDataset(
        df_test, coord_cols, gene_cols,
        coord_bounds=train_dataset.coord_bounds,
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

    run_name: str = out_cfg.get(
        "run_name",
        Path(out_cfg.get("checkpoint_name", "run")).stem,
    )

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

    # ---- Neural field predictions ----
    device = torch.device(train_cfg.get("device", "cpu"))
    model.eval()
    model = model.to(device)

    all_preds: list[torch.Tensor] = []
    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size * 4, shuffle=False
        )
        for coords, _ in test_loader:
            coords = coords.to(device)
            all_preds.append(model(coords).cpu())

    nf_preds = torch.cat(all_preds, dim=0).numpy()    # (N_test, G)
    true_arr = test_dataset.exprs.numpy()              # (N_test, G)

    # ---- Baseline predictions ----
    log.info("Computing baselines …")

    knn_k: int = cfg.get("baselines", {}).get("knn_k", 10)

    ns_preds = nearest_section_baseline(df_train, df_test, coord_cols, gene_cols)
    log.info("  nearest_section done")

    lz_preds = linear_z_interpolation_baseline(df_train, df_test, coord_cols, gene_cols)
    log.info("  linear_z_interpolation done")

    knn_preds = knn_xyz_baseline(df_train, df_test, coord_cols, gene_cols, k=knn_k)
    log.info("  knn (k=%d) done", knn_k)

    # ---- Save per-method prediction CSVs ----
    pred_dir = Path(out_cfg["predictions_dir"])

    methods: list[tuple[str, np.ndarray]] = [
        ("neural_field", nf_preds),
        ("nearest_section", ns_preds),
        ("linear_z", lz_preds),
        ("knn", knn_preds),
    ]

    for method_name, preds in methods:
        save_predictions(
            df_test=df_test,
            coord_cols=coord_cols,
            section_col=section_col,
            gene_cols=gene_cols,
            pred_array=preds,
            path=pred_dir / f"{run_name}_{method_name}_predictions.csv",
        )

    # ---- Compute and save combined metrics ----
    all_metrics: list[pd.DataFrame] = []
    for method_name, preds in methods:
        m = compute_metrics(true_arr, preds, gene_names=gene_cols)
        summary = summarize_metrics(m)
        log.info(
            "%-22s  MSE=%.4f  MAE=%.4f  Pearson_r=%.4f",
            method_name, summary["mse"], summary["mae"], summary["pearson_r"],
        )
        m = m.reset_index().rename(columns={"index": "gene"})
        m.insert(0, "method", method_name)
        all_metrics.append(m)

    combined = pd.concat(all_metrics, ignore_index=True)
    combined_path = pred_dir / f"{run_name}_all_metrics.csv"
    save_csv(combined, combined_path)
    log.info("Combined metrics saved → %s", combined_path)

    # ---- Figures ----
    import matplotlib.pyplot as plt

    fig_dir = Path(out_cfg["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    nf_metrics = combined[combined["method"] == "neural_field"].set_index("gene")

    plot_genes: list[str] = out_cfg.get("plot_genes") or gene_cols[:4]
    plot_genes = [g for g in plot_genes if g in gene_cols]

    if plot_genes:
        g_indices = [gene_cols.index(g) for g in plot_genes]

        # Simple true-vs-neural-field map (2 columns).
        pred_cols_plot = [f"{g}_pred" for g in plot_genes]
        plot_df = df_test[coord_cols + [section_col] + plot_genes].copy()
        plot_df[pred_cols_plot] = nf_preds[:, g_indices]

        fig = plot_gene_maps(
            df=plot_df,
            gene_names=plot_genes,
            true_cols=plot_genes,
            pred_cols=pred_cols_plot,
            x_col=coord_cols[0],
            y_col=coord_cols[1],
            save_path=fig_dir / f"{run_name}_gene_maps.png",
        )
        plt.close(fig)
        log.info("Gene maps saved → %s/%s_gene_maps.png", fig_dir, run_name)

        # Multi-method comparison figure (True | baselines | NF | NF error).
        fig_comp = plot_method_comparison(
            x=df_test[coord_cols[0]].values,
            y=df_test[coord_cols[1]].values,
            true_expr=true_arr[:, g_indices],
            method_preds=[
                ("Nearest section",   ns_preds[:, g_indices]),
                ("Linear-z interp.",  lz_preds[:, g_indices]),
                ("KNN",               knn_preds[:, g_indices]),
                ("Neural field",      nf_preds[:, g_indices]),
            ],
            gene_names=plot_genes,
            x_label=coord_cols[0],
            y_label=coord_cols[1],
            save_path=fig_dir / f"{run_name}_method_comparison_gene_maps.png",
        )
        plt.close(fig_comp)
        log.info(
            "Method comparison figure saved → %s/%s_method_comparison_gene_maps.png",
            fig_dir, run_name,
        )

    fig_pearson = plot_pearson_barplot(
        nf_metrics["pearson_r"],
        title=f"Neural field – per-gene Pearson r (held-out: {held_out})",
        save_path=fig_dir / f"{run_name}_pearson_r.png",
    )
    plt.close(fig_pearson)
    log.info("Pearson barplot saved → %s/%s_pearson_r.png", fig_dir, run_name)

    log.info("Done.")


if __name__ == "__main__":
    main()
