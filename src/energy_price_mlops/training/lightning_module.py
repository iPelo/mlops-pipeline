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
        target_mean: float = 0.0,
        target_std: float = 1.0,
        feature_means: list[float] | None = None,
        feature_stds: list[float] | None = None,
    ) -> None:
        super().__init__()
        if target_std <= 0:
            raise ValueError("target_std must be positive.")
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.target_mean = target_mean
        self.target_std = target_std
        self.feature_means = feature_means or [target_mean]
        self.feature_stds = feature_stds or [target_std]
        self.save_hyperparameters(ignore=["model"])

    def forward(self, inputs: Tensor) -> Tensor:
        predictions: Tensor = self.model(inputs)
        return predictions

    def _shared_step(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        inputs, targets = batch
        predictions = self(inputs)
        loss = F.mse_loss(predictions, targets)
        original_predictions = self._to_original_scale(predictions)
        original_targets = self._to_original_scale(targets)
        mae = F.l1_loss(original_predictions, original_targets)
        rmse = torch.sqrt(F.mse_loss(original_predictions, original_targets))
        return loss, mae, rmse

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss, mae, rmse = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_mae", mae, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_rmse", rmse, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss, mae, rmse = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mae", mae, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_rmse", rmse, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss, mae, rmse = self._shared_step(batch)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_mae", mae, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_rmse", rmse, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def _to_original_scale(self, values: Tensor) -> Tensor:
        return values * self.target_std + self.target_mean
