from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from torch import Tensor

from energy_price_mlops.models.mlp import PriceMLP


class PriceForecastModule(L.LightningModule):
    """LightningModule that trains :class:`PriceMLP` to forecast prices.

    The module is intentionally thin: it owns the optimization concerns
    (loss, logged metrics, optimizer) and delegates the architecture to
    :class:`~energy_price_mlops.models.mlp.PriceMLP`. Each batch is a
    ``(inputs, targets)`` pair of float tensors shaped ``(batch, context)``
    and ``(batch, horizon)``.
    """

    def __init__(
        self,
        model: PriceMLP,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=["model"])

    def forward(self, inputs: Tensor) -> Tensor:
        predictions: Tensor = self.model(inputs)
        return predictions

    def _shared_step(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        inputs, targets = batch
        predictions = self(inputs)
        loss = F.mse_loss(predictions, targets)
        mae = F.l1_loss(predictions, targets)
        return loss, mae

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss, mae = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_mae", mae, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss, mae = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mae", mae, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
