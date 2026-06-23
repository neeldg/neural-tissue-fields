# Neural Tissue Fields

This project explores continuous neural representations for spatial omics.

Goal: learn a function

f(x, y, z) → molecular state / gene expression

from measured spatial transcriptomics sections, then test whether the model can predict a held-out tissue section.

First milestone:
- Load public spatial transcriptomics data
- Convert it into x, y, z → expression format
- Train a simple coordinate MLP
- Predict a held-out section
- Plot real vs predicted gene expression

---

## MALACHY v1

MALACHY v1 tests whether coordinate-based neural fields can learn real spatial-omics molecular fields from Visium HD data.

Current v1 scope:
- Input: x, y coordinates from one Visium HD section
- Output: selected gene expression values
- Model: MLP or GridField coordinate neural field
- Validation: held-out spatial regions within a real tissue section
- Baselines: KNN interpolation
- Goal: establish a real-data coordinate-field baseline before moving to serial-section 3D reconstruction.

### Quick start

Preprocess a Visium HD section (produces `data/processed/{section_id}_{bin_level}_visium_hd.parquet`):

```bash
python scripts/preprocess_visium_hd.py \
    --binned-dir /path/to/spaceranger_hd/outs/binned_outputs \
    --bin-level square_016um \
    --section-id breast_tma_hd_0 \
    --z 0 \
    --n-top-genes 50
```

Select spatially variable genes first, then train on them:

```bash
python scripts/select_spatial_genes.py \
    --input data/processed/breast_tma_hd_0_square016.parquet \
    --top-n 30

python scripts/train_2d_holdout.py \
    --input data/processed/breast_tma_hd_0_square016.parquet \
    --holdout-mode stripe \
    --model mlp \
    --epochs 50 \
    --gene-list outputs/spatial_genes/breast_tma_hd_0_square016_spatial_genes.txt \
    --output-prefix breast_tma_stripe_mlp_spatial30
```

Or train on all genes without a gene list:

```bash
python scripts/train_2d_holdout.py \
    --input data/processed/breast_tma_hd_0_square016.parquet \
    --holdout-mode stripe \
    --model mlp \
    --epochs 50 \
    --output-prefix breast_tma_stripe_mlp
```

Train a GridField with quadrant holdout:

```bash
python scripts/train_2d_holdout.py \
    --input data/processed/breast_tma_hd_0_square016.parquet \
    --holdout-mode quadrant \
    --model gridfield \
    --epochs 50 \
    --output-prefix breast_tma_quadrant_grid
```

Select spatially variable genes (run before training for best results):

```bash
python scripts/select_spatial_genes.py \
    --input data/processed/breast_tma_hd_0_square016.parquet \
    --top-n 30 --plot
```

This ranks genes by a spatial autocorrelation score and writes:
- `outputs/spatial_genes/{stem}_spatial_genes.csv` — full ranked table
- `outputs/spatial_genes/{stem}_spatial_genes.txt` — top-N gene names, comma-separated
- `outputs/figures/spatial_gene_ranking.png` — bar chart (if `--plot`)

Summarize results:

```bash
python scripts/summarize_run.py \
    --metrics outputs/predictions/breast_tma_stripe_mlp_metrics.csv
```

### Outputs

| File | Description |
|------|-------------|
| `outputs/predictions/{prefix}_predictions.csv` | Long-format: x, y, section_id, split, method, gene, true, pred |
| `outputs/predictions/{prefix}_metrics.csv` | Per-gene: method, gene, mse, mae, pearson_r |
| `outputs/figures/{prefix}_gene_maps.png` | Spatial maps: true / KNN / neural field / absolute error |
