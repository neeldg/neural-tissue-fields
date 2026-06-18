"""MALACHY-v1 coordinate encodings.

Two encoding strategies, both usable as drop-in front-ends for an MLP:

FourierEncoding
    Thin wrapper around SinusoidalEncoding that exposes the same ``out_dim``
    property and ``forward`` interface as the grid encoding, so the two can be
    swapped without changing downstream code.

MultiResolutionGridEncoding
    Learns a set of 2-D feature grids at increasing spatial resolutions.
    At each level a bilinear interpolation samples the grid at the query
    coordinates; all levels are concatenated into a single feature vector.
    The grids are trainable parameters, so the encoding adapts to data.

GridField
    Convenience module: ``MultiResolutionGridEncoding`` followed by a
    configurable MLP decoder.  Produces the same ``(coords) → expression``
    interface as ``CoordinateMLP``, so the training loop is unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.positional_encoding import SinusoidalEncoding


# ---------------------------------------------------------------------------
# FourierEncoding
# ---------------------------------------------------------------------------

class FourierEncoding(nn.Module):
    """Sinusoidal / NeRF positional encoding with a unified coordinate-encoding API.

    Delegates entirely to ``SinusoidalEncoding``; exists so both encoding
    strategies share the same ``out_dim`` property and ``forward`` signature.

    Args:
        in_dim:        Number of input coordinate dimensions.
        n_freqs:       Number of frequency bands (2^0 … 2^(n_freqs-1)).
        include_input: If True, prepend raw coordinates to the encoded output.
    """

    def __init__(
        self,
        in_dim: int = 2,
        n_freqs: int = 6,
        include_input: bool = True,
    ):
        super().__init__()
        self._enc = SinusoidalEncoding(
            in_dim=in_dim, n_freqs=n_freqs, include_input=include_input
        )

    @property
    def out_dim(self) -> int:
        return self._enc.out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., in_dim) coordinates.
        Returns:
            (..., out_dim) Fourier features.
        """
        return self._enc(x)


# ---------------------------------------------------------------------------
# MultiResolutionGridEncoding
# ---------------------------------------------------------------------------

class MultiResolutionGridEncoding(nn.Module):
    """Learnable multi-resolution 2-D feature grids with bilinear interpolation.

    For each resolution level R, a parameter grid of shape
    ``(1, n_features, R, R)`` is initialised small and learned during training.
    Given a batch of 2-D query coordinates in ``[-1, 1]``, bilinear
    interpolation samples each grid.  The resulting per-level features are
    concatenated to produce the final descriptor.

    This follows the spirit of instant-NGP / hash-grid encodings but uses a
    dense grid per level (no hashing), which is simpler and avoids hash
    collisions at the resolutions relevant for Visium-scale data.

    Args:
        resolutions: Spatial resolution of each grid level (number of cells
                     per side).  More levels → longer feature vector.
        n_features:  Number of feature channels per grid level.
        in_dim:      Number of spatial input dimensions (must be 2 for 2-D
                     grids; 3-D support can be added by switching to
                     ``F.grid_sample`` with 5-D tensors).
    """

    def __init__(
        self,
        resolutions: list[int] | tuple[int, ...] = (16, 32, 64, 128),
        n_features: int = 8,
        in_dim: int = 2,
    ):
        super().__init__()
        if in_dim != 2:
            raise ValueError(
                f"MultiResolutionGridEncoding only supports in_dim=2; got {in_dim}."
            )
        self.resolutions = list(resolutions)
        self.n_features = n_features
        self.in_dim = in_dim

        # One learnable grid per resolution level.
        # Small initialisation keeps early-training outputs near zero.
        self.grids = nn.ParameterList([
            nn.Parameter(torch.zeros(1, n_features, R, R).normal_(std=0.01))
            for R in self.resolutions
        ])

    @property
    def out_dim(self) -> int:
        return self.n_features * len(self.resolutions)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Bilinearly sample all grid levels and concatenate.

        Args:
            coords: (N, 2) tensor of (x, y) coordinates in ``[-1, 1]``.
                    x maps to the W (column) axis of the grid;
                    y maps to the H (row) axis — matching PyTorch's
                    ``grid_sample`` convention.
        Returns:
            (N, out_dim) feature tensor.
        """
        N = coords.shape[0]

        # grid_sample expects grid shape (batch, H_out, W_out, 2).
        # We treat all N query points as a (N, 1) spatial output — i.e.
        # (1, N, 1, 2).  Each grid has batch size 1.
        #
        # Clamp to [-1, 1] before sampling.  Inputs are already normalised to
        # this range; the clamp guards against floating-point overshoot and
        # lets us use padding_mode='zeros', which is supported on all backends
        # including MPS (unlike 'border', which MPS does not implement).
        sample_grid = coords.clamp(-1.0, 1.0).view(1, N, 1, 2)   # (1, N, 1, 2)

        level_feats: list[torch.Tensor] = []
        for grid in self.grids:
            # grid:    (1, n_features, R, R)
            # sampled: (1, n_features, N, 1)
            sampled = F.grid_sample(
                grid,
                sample_grid,
                mode="bilinear",
                padding_mode="zeros",    # safe: coords are clamped above
                align_corners=True,      # corners of the grid map to ±1
            )
            # (1, n_features, N, 1) → (N, n_features)
            level_feats.append(sampled[0, :, :, 0].T)

        return torch.cat(level_feats, dim=-1)   # (N, out_dim)


# ---------------------------------------------------------------------------
# GridField
# ---------------------------------------------------------------------------

class GridField(nn.Module):
    """Multi-resolution grid encoding followed by a ReLU MLP decoder.

    Provides the same ``forward(coords) → expression`` interface as
    ``CoordinateMLP``, so the training loop in ``train_2d_holdout.py`` (and
    ``train_v0.py``) does not need to change.

    Args:
        n_genes:     Number of output gene expression values.
        resolutions: Grid resolution levels passed to
                     ``MultiResolutionGridEncoding``.
        n_features:  Feature channels per resolution level.
        hidden_dims: MLP decoder hidden layer widths.
    """

    def __init__(
        self,
        n_genes: int,
        resolutions: list[int] | tuple[int, ...] = (16, 32, 64, 128),
        n_features: int = 8,
        hidden_dims: list[int] | tuple[int, ...] = (256, 256, 256),
    ):
        super().__init__()
        self.encoding = MultiResolutionGridEncoding(
            resolutions=resolutions, n_features=n_features
        )

        layers: list[nn.Module] = []
        in_dim = self.encoding.out_dim
        for width in hidden_dims:
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.ReLU(inplace=True))
            in_dim = width
        layers.append(nn.Linear(in_dim, n_genes))

        self.decoder = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (N, 2) normalized coordinates in ``[-1, 1]``.
        Returns:
            (N, n_genes) predicted expression.
        """
        return self.decoder(self.encoding(coords))
