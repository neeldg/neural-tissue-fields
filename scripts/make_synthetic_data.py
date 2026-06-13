"""Generate a small synthetic spatial transcriptomics dataset for development and testing.

Output: data/processed/synthetic_sections.parquet
Columns: section_id, x, y, z, gene_0, ..., gene_19

Gene expression is constructed from:
  - Genes 0–11: smooth spatial signals (sinusoidal waves, gradients, z-modulated)
  - Genes 12–15: localized Gaussian hotspots
  - Genes 16–19: cross-section z-gradient combined with spatial texture
All values are non-negative (ReLU-clipped) with small multiplicative noise added.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Grid and section geometry
# ---------------------------------------------------------------------------

N_SECTIONS = 5
GRID_SIZE = 30          # spots per side → 30×30 = 900 spots per section
X_RANGE = (0.0, 1.0)
Y_RANGE = (0.0, 1.0)
N_GENES = 20
NOISE_LEVEL = 0.05      # std of multiplicative noise relative to signal amplitude
SEED = 42


def make_grid(grid_size: int = GRID_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened x, y arrays for a regular grid in [0,1]^2."""
    xs = np.linspace(X_RANGE[0], X_RANGE[1], grid_size)
    ys = np.linspace(Y_RANGE[0], Y_RANGE[1], grid_size)
    xx, yy = np.meshgrid(xs, ys)
    return xx.ravel(), yy.ravel()


# ---------------------------------------------------------------------------
# Signal generators  (all return arrays of shape (N_spots,))
# ---------------------------------------------------------------------------

def smooth_wave(x: np.ndarray, y: np.ndarray, z: float,
                freq_x: float, freq_y: float, freq_z: float,
                phase_x: float = 0.0, phase_y: float = 0.0) -> np.ndarray:
    """Sinusoidal pattern that varies across x, y, and z."""
    return (
        np.sin(freq_x * np.pi * x + phase_x)
        * np.cos(freq_y * np.pi * y + phase_y)
        * (1.0 + 0.4 * np.sin(freq_z * np.pi * z))
    )


def linear_gradient(x: np.ndarray, y: np.ndarray, z: float,
                     wx: float, wy: float, wz: float) -> np.ndarray:
    """Smooth linear combination of coordinates."""
    return wx * x + wy * y + wz * z


def radial_basis(x: np.ndarray, y: np.ndarray,
                  cx: float, cy: float, sigma: float,
                  amplitude: float = 1.0) -> np.ndarray:
    """2-D Gaussian bump centred at (cx, cy)."""
    return amplitude * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))


def hotspot_gene(x: np.ndarray, y: np.ndarray, z: float,
                  centres: list[tuple[float, float]],
                  sigma: float = 0.12,
                  z_weight: float = 0.0) -> np.ndarray:
    """Sum of Gaussian hotspots at the given (cx, cy) centres.

    Optional z_weight makes hotspot intensity vary linearly with section depth.
    """
    signal = np.zeros_like(x)
    for cx, cy in centres:
        signal += radial_basis(x, y, cx, cy, sigma)
    if z_weight != 0.0:
        signal *= 1.0 + z_weight * z
    return signal


# ---------------------------------------------------------------------------
# Per-gene construction  (deterministic, parameterised by gene index)
# ---------------------------------------------------------------------------

def make_gene_expression(
    x: np.ndarray, y: np.ndarray, z: float, gene_idx: int, rng: np.random.Generator
) -> np.ndarray:
    """Return expression values for one gene across all spots in one section."""

    if gene_idx <= 3:
        # Genes 0-3: single-frequency sinusoidal waves
        freq = 1.0 + gene_idx * 0.75
        signal = smooth_wave(x, y, z, freq_x=freq, freq_y=freq * 0.8,
                              freq_z=1.0, phase_x=gene_idx * 0.5)

    elif gene_idx <= 7:
        # Genes 4-7: higher-frequency waves with z-modulation
        freq = 2.0 + (gene_idx - 4) * 1.0
        signal = smooth_wave(x, y, z, freq_x=freq, freq_y=freq * 1.2,
                              freq_z=2.0, phase_x=gene_idx, phase_y=gene_idx * 0.3)

    elif gene_idx <= 11:
        # Genes 8-11: smooth linear gradients (mimic expression axes / cell-type zones)
        angle = (gene_idx - 8) * np.pi / 4
        wx = np.cos(angle)
        wy = np.sin(angle)
        wz = 0.3 * (1 if gene_idx % 2 == 0 else -1)
        signal = linear_gradient(x, y, z, wx=wx, wy=wy, wz=wz)

    elif gene_idx <= 15:
        # Genes 12-15: localized hotspots (mimic spatially restricted markers)
        # Each gene has 1-2 hotspots at fixed locations that shift slightly with z.
        base_centres = [
            [(0.25, 0.25)],
            [(0.75, 0.25), (0.75, 0.75)],
            [(0.5, 0.5)],
            [(0.2, 0.8), (0.8, 0.2)],
        ][gene_idx - 12]
        # Hotspot position drifts slightly across sections.
        centres = [(cx + 0.02 * z, cy - 0.02 * z) for cx, cy in base_centres]
        sigma = 0.10 + 0.02 * (gene_idx - 12)
        signal = hotspot_gene(x, y, z, centres, sigma=sigma, z_weight=0.3)

    else:
        # Genes 16-19: z-gradient × spatial texture (mimic layer-specific expression)
        freq = 1.5 + (gene_idx - 16) * 0.5
        texture = np.sin(freq * np.pi * x) * np.sin(freq * np.pi * y)
        z_envelope = (gene_idx - 16 + 1) / 4 * z   # each gene peaks at a different z
        signal = texture * (0.5 + z_envelope)

    # Shift so the bulk of the signal is positive, then ReLU to mimic count-like data.
    signal = signal - signal.min() + 0.1
    signal = np.maximum(signal, 0.0)

    # Multiplicative noise: expression = signal * (1 + ε),  ε ~ N(0, noise_level)
    noise = rng.normal(loc=0.0, scale=NOISE_LEVEL, size=signal.shape)
    signal = signal * (1.0 + noise)
    signal = np.maximum(signal, 0.0)   # keep non-negative after noise

    return signal.astype(np.float32)


# ---------------------------------------------------------------------------
# Main dataset assembly
# ---------------------------------------------------------------------------

def make_dataset(grid_size: int = GRID_SIZE, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x_grid, y_grid = make_grid(grid_size)
    n_spots = len(x_grid)

    records = []
    for section_idx in range(N_SECTIONS):
        section_id = f"section_{section_idx}"
        z = float(section_idx)   # integer z per section; normalisation handles scale

        gene_exprs = {
            f"gene_{g}": make_gene_expression(x_grid, y_grid, z, g, rng)
            for g in range(N_GENES)
        }

        df_sec = pd.DataFrame({
            "section_id": section_id,
            "x": x_grid.astype(np.float32),
            "y": y_grid.astype(np.float32),
            "z": np.full(n_spots, z, dtype=np.float32),
            **gene_exprs,
        })
        records.append(df_sec)

    df = pd.concat(records, ignore_index=True)

    # Enforce column order: section_id, x, y, z, gene_0, ..., gene_19
    gene_cols = [f"gene_{g}" for g in range(N_GENES)]
    df = df[["section_id", "x", "y", "z"] + gene_cols]
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic spatial transcriptomics data.")
    parser.add_argument(
        "--output", default="data/processed/synthetic_sections.parquet",
        help="Output path for the Parquet file.",
    )
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE,
                        help=f"Spots per grid side (default {GRID_SIZE}, giving grid²  spots/section).")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic dataset  grid={args.grid_size}×{args.grid_size}, "
          f"sections={N_SECTIONS}, genes={N_GENES} ...")
    df = make_dataset(grid_size=args.grid_size, seed=args.seed)

    try:
        df.to_parquet(out_path, index=False)
    except ImportError:
        # pyarrow / fastparquet not installed — fall back to CSV.
        csv_path = out_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"Note: parquet engine not found; saved as CSV instead → {csv_path}")
        out_path = csv_path

    print(f"Saved {len(df)} spots ({N_SECTIONS} sections × {args.grid_size**2} spots) "
          f"→ {out_path}")
    print(f"Columns: {list(df.columns)}")
    print(df.describe().loc[["min", "mean", "max"]].to_string())


if __name__ == "__main__":
    main()
