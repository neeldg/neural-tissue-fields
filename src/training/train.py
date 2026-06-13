"""Training loop for the coordinate MLP neural field."""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.mlp_field import CoordinateMLP

log = logging.getLogger(__name__)


def train(
    model: CoordinateMLP,
    train_loader: DataLoader,
    *,
    n_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    checkpoint_path: Path | str | None = None,
    device: str | torch.device = "cpu",
    log_every: int = 10,
) -> dict[str, list[float]]:
    """Train `model` with MSE loss and the Adam optimizer.

    Args:
        model:            CoordinateMLP (or any nn.Module with compatible I/O).
        train_loader:     DataLoader yielding (coords, expression) batches.
        n_epochs:         Number of full passes over the training data.
        lr:               Adam learning rate.
        weight_decay:     L2 regularisation coefficient.
        checkpoint_path:  If provided, save the final model weights here.
        device:           Torch device string or object.
        log_every:        Print a loss line every this many epochs.

    Returns:
        history dict with key 'train_loss' containing per-epoch mean MSE.
    """
    device = torch.device(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()

    history: dict[str, list[float]] = {"train_loss": []}

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for coords, targets in train_loader:
            coords = coords.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(coords)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        mean_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(mean_loss)

        if epoch % log_every == 0 or epoch == 1:
            log.info("Epoch %4d/%d  train_loss=%.6f", epoch, n_epochs, mean_loss)

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": n_epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": history["train_loss"][-1],
            },
            checkpoint_path,
        )
        log.info("Checkpoint saved → %s", checkpoint_path)

    return history


def load_checkpoint(
    model: CoordinateMLP,
    checkpoint_path: Path | str,
    device: str | torch.device = "cpu",
) -> CoordinateMLP:
    """Load model weights from a checkpoint produced by `train()`."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    log.info("Loaded checkpoint from %s (epoch %d)", checkpoint_path, ckpt["epoch"])
    return model
