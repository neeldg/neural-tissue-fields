"""CoordinateMLP: maps (x, y, z) coordinates to a gene-expression vector."""

import torch
import torch.nn as nn

from src.models.positional_encoding import SinusoidalEncoding


class CoordinateMLP(nn.Module):
    """Simple MLP neural field.

    Architecture:
        [optional positional encoding] → Linear → ReLU → ... → Linear → output
    The output is raw (no activation); use MSE loss with unnormalized targets or
    add a downstream activation if expression values are bounded.
    """

    def __init__(
        self,
        n_genes: int,
        coord_dim: int = 3,
        hidden_dims: list[int] | None = None,
        use_positional_encoding: bool = True,
        n_freqs: int = 6,
        dropout: float = 0.0,
    ):
        """
        Args:
            n_genes:                  Number of output gene expression values.
            coord_dim:                Dimensionality of input coordinates (default 3).
            hidden_dims:              List of hidden layer widths.
            use_positional_encoding:  If True, apply sinusoidal encoding first.
            n_freqs:                  Frequency bands for positional encoding.
            dropout:                  Dropout probability applied after each hidden layer.
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256, 256]

        self.use_pe = use_positional_encoding
        self.pos_enc: SinusoidalEncoding | None = None

        if use_positional_encoding:
            self.pos_enc = SinusoidalEncoding(
                in_dim=coord_dim, n_freqs=n_freqs, include_input=True
            )
            in_features = self.pos_enc.out_dim
        else:
            in_features = coord_dim

        layers: list[nn.Module] = []
        prev = in_features
        for width in hidden_dims:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = width

        layers.append(nn.Linear(prev, n_genes))

        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (batch, coord_dim) tensor of normalized coordinates in [-1, 1].
        Returns:
            (batch, n_genes) predicted expression tensor.
        """
        x = self.pos_enc(coords) if self.use_pe else coords
        return self.net(x)
