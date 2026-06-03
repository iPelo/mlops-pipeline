from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, Logger, WandbLogger
from omegaconf import DictConfig, OmegaConf

from energy_price_mlops.data import SmardPriceDataModule
from energy_price_mlops.models.mlp import PriceMLP
from energy_price_mlops.training.lightning_module import PriceForecastModule


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(config: DictConfig) -> None:
    L.seed_everything(int(config.seed), workers=True)

    artifacts_dir = Path(config.paths.artifacts_dir)
    training_dir = artifacts_dir / "training"
    run_dir = training_dir / str(config.training.run_name)
    checkpoint_dir = run_dir / "checkpoints"
    metrics_path = run_dir / "metrics.json"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    data_module = SmardPriceDataModule(
        train_path=config.data.files.processed_train,
        valid_path=config.data.files.processed_valid,
        test_path=config.data.files.processed_test,
        target_col=config.target,
        feature_cols=[str(column) for column in config.data.feature_columns],
        context_size=int(config.data.forecasting.context_hours),
        horizon_size=int(config.data.forecasting.horizon_hours),
        batch_size=int(config.training.batch_size),
        num_workers=int(config.training.num_workers),
    )
    data_module.setup("fit")
    if data_module.normalizer is None:
        raise RuntimeError("DataModule did not initialize target normalization.")
    if data_module.feature_normalizer is None:
        raise RuntimeError("DataModule did not initialize feature normalization.")

    input_size = int(config.data.forecasting.context_hours) * len(config.data.feature_columns)
    model = PriceMLP(
        input_size=input_size,
        hidden_sizes=[int(size) for size in config.model.hidden_sizes],
        dropout=float(config.model.dropout),
        output_size=int(config.model.output_size),
    )
    module = PriceForecastModule(
        model,
        lr=float(config.optim.lr),
        weight_decay=float(config.optim.weight_decay),
        target_mean=data_module.normalizer.mean,
        target_std=data_module.normalizer.std,
        feature_means=data_module.feature_normalizer.means.astype(float).tolist(),
        feature_stds=data_module.feature_normalizer.stds.astype(float).tolist(),
    )

    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch:03d}-{val_mae:.3f}",
        monitor=str(config.training.monitor_metric),
        mode=str(config.training.monitor_mode),
        save_top_k=3,
        save_last=True,
    )
    callbacks = [
        checkpoint,
        EarlyStopping(
            monitor=str(config.training.monitor_metric),
            mode=str(config.training.monitor_mode),
            patience=int(config.training.early_stopping_patience),
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = L.Trainer(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        max_epochs=int(config.trainer.max_epochs),
        precision=config.trainer.precision,
        log_every_n_steps=int(config.trainer.log_every_n_steps),
        gradient_clip_val=float(config.trainer.gradient_clip_val),
        deterministic=bool(config.trainer.deterministic),
        enable_progress_bar=bool(config.trainer.enable_progress_bar),
        callbacks=callbacks,
        logger=_build_logger(config, run_dir),
    )

    trainer.fit(module, datamodule=data_module)
    test_results = trainer.test(module, datamodule=data_module, ckpt_path="best")

    metrics = {
        "config": OmegaConf.to_container(config, resolve=True),
        "model_input_size": input_size,
        "feature_columns": list(config.data.feature_columns),
        "feature_means": data_module.feature_normalizer.means.astype(float).tolist(),
        "feature_stds": data_module.feature_normalizer.stds.astype(float).tolist(),
        "best_checkpoint_path": checkpoint.best_model_path,
        "best_val_mae": _as_float(checkpoint.best_model_score),
        "test": test_results[0] if test_results else {},
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(f"Wrote metrics to {metrics_path}")


def _build_logger(config: DictConfig, run_dir: Path) -> Logger:
    provider = str(config.tracking.provider).lower()
    if provider == "wandb":
        return WandbLogger(
            project=str(config.tracking.project),
            entity=None if config.tracking.entity is None else str(config.tracking.entity),
            save_dir=str(run_dir),
            mode=str(config.tracking.mode),
            name=str(config.training.run_name),
            log_model=False,
        )
    if provider == "csv":
        return CSVLogger(save_dir=str(run_dir), name="csv")
    raise ValueError(f"Unsupported tracking provider: {config.tracking.provider}")


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


if __name__ == "__main__":
    main()
