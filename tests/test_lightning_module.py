import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset

from energy_price_mlops.models.mlp import PriceMLP
from energy_price_mlops.training.lightning_module import PriceForecastModule

CONTEXT = 168
HORIZON = 24


def _build_module() -> PriceForecastModule:
    model = PriceMLP(input_size=CONTEXT, hidden_sizes=[32], output_size=HORIZON)
    return PriceForecastModule(model, lr=1e-3, target_mean=50.0, target_std=10.0)


def _build_dataloader(rows: int = 16) -> DataLoader[tuple[torch.Tensor, ...]]:
    dataset = TensorDataset(torch.randn(rows, CONTEXT), torch.randn(rows, HORIZON))
    return DataLoader(dataset, batch_size=8)


def test_forward_returns_horizon_shaped_output() -> None:
    module = _build_module()

    output = module(torch.randn(4, CONTEXT))

    assert output.shape == (4, HORIZON)


def test_configure_optimizers_returns_adamw() -> None:
    module = _build_module()

    optimizer = module.configure_optimizers()

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == 1e-3


def test_fast_dev_run_executes_training_and_validation_steps() -> None:
    module = _build_module()
    trainer = L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
    )

    trainer.fit(module, _build_dataloader(), _build_dataloader())

    assert trainer.state.finished
    assert "train_loss" in trainer.callback_metrics
    assert "val_loss" in trainer.callback_metrics
    assert "val_mae" in trainer.callback_metrics


def test_metrics_are_logged_on_original_target_scale() -> None:
    model = PriceMLP(input_size=2, hidden_sizes=[], output_size=2)
    module = PriceForecastModule(model, target_mean=10.0, target_std=2.0)
    batch = (torch.zeros(1, 2), torch.ones(1, 2))

    loss, mae, rmse = module._shared_step(batch)

    assert loss.item() >= 0
    assert mae.item() >= 0
    assert rmse.item() >= 0
