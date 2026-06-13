"""Sinusoidal positional encoding for 3D (or arbitrary-dim) coordinates."""

import math
import torch
import torch.nn as nn


class SinusoidalEncoding(nn.Module):
    """Fourier feature / sinusoidal encoding for continuous coordinates.

    For each input dimension d and each frequency band k, we compute
        sin(2^k * pi * x_d),  cos(2^k * pi * x_d)
    and concatenate them, giving an output of size
        in_dim * 2 * n_freqs   (optionally + in_dim for the raw coords).

    This is the standard NeRF positional encoding; it helps the MLP learn
    high-frequency spatial variation.
    """

    def __init__(
        self,
        in_dim: int = 3,
        n_freqs: int = 6,
        include_input: bool = True,
    ):
        """
        Args:
            in_dim:        Number of input coordinate dimensions (typically 3).
            n_freqs:       Number of frequency bands.  Output size is
                           in_dim * 2 * n_freqs (+ in_dim if include_input).
            include_input: Whether to concatenate the raw coordinates.
        """
        super().__init__()
        self.in_dim = in_dim
        self.n_freqs = n_freqs
        self.include_input = include_input

        # Frequencies: 2^0, 2^1, ..., 2^(n_freqs-1)  (shape: n_freqs)
        freqs = 2.0 ** torch.arange(n_freqs, dtype=torch.float32) * math.pi
        self.register_buffer("freqs", freqs)

    @property
    def out_dim(self) -> int:
        base = self.in_dim * 2 * self.n_freqs
        return base + self.in_dim if self.include_input else base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., in_dim) coordinate tensor.
        Returns:
            (..., out_dim) encoded tensor.
        """
        # x: (..., D)  freqs: (F,)
        # Outer product: (..., D, F) -> flatten to (..., D*F)
        x_freq = x.unsqueeze(-1) * self.freqs          # (..., D, F)
        encoded = torch.cat([x_freq.sin(), x_freq.cos()], dim=-1)  # (..., D, 2F)
        encoded = encoded.flatten(-2)                  # (..., D*2F)

        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)  # (..., D + D*2F)

        return encoded
