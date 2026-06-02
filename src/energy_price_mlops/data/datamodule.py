from __future__ import annotations

from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from energy_price_mlops.data.windowing import (
    PriceWindowDataset,
    TargetNormalizer,
    build_forecast_start_indices,
)


class SmardPriceDataModule(L.LightningDataModule):
    """LightningDataModule for univariate SMARD price forecasting."""

    def __init__(
        self,
        *,
        train_path: str | Path,
        valid_path: str | Path,
        test_path: str | Path,
        target_col: str,
        context_size: int,
        horizon_size: int,
        batch_size: int = 128,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.train_path = Path(train_path)
        self.valid_path = Path(valid_path)
        self.test_path = Path(test_path)
        self.target_col = target_col
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.normalizer: TargetNormalizer | None = None
        self.train_dataset: PriceWindowDataset | None = None
        self.valid_dataset: PriceWindowDataset | None = None
        self.test_dataset: PriceWindowDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        train = self._read_split(self.train_path)
        valid = self._read_split(self.valid_path)
        test = self._read_split(self.test_path)

        train_values = self._target_values(train)
        valid_values = self._target_values(valid)
        test_values = self._target_values(test)

        mean = float(train_values.mean())
        std = float(train_values.std())
        if std == 0:
            raise ValueError("Training target standard deviation is zero.")
        self.normalizer = TargetNormalizer(mean=mean, std=std)

        train_series = train_values
        train_valid_series = np.concatenate([train_values, valid_values]).astype(np.float32)
        full_series = np.concatenate([train_values, valid_values, test_values]).astype(np.float32)

        self.train_dataset = PriceWindowDataset(
            train_series,
            build_forecast_start_indices(
                first_forecast_start=self.context_size,
                last_forecast_start_exclusive=len(train_series),
                context_size=self.context_size,
                horizon_size=self.horizon_size,
                series_length=len(train_series),
            ),
            context_size=self.context_size,
            horizon_size=self.horizon_size,
            normalizer=self.normalizer,
        )
        self.valid_dataset = PriceWindowDataset(
            train_valid_series,
            build_forecast_start_indices(
                first_forecast_start=len(train_values),
                last_forecast_start_exclusive=len(train_valid_series),
                context_size=self.context_size,
                horizon_size=self.horizon_size,
                series_length=len(train_valid_series),
            ),
            context_size=self.context_size,
            horizon_size=self.horizon_size,
            normalizer=self.normalizer,
        )
        self.test_dataset = PriceWindowDataset(
            full_series,
            build_forecast_start_indices(
                first_forecast_start=len(train_values) + len(valid_values),
                last_forecast_start_exclusive=len(full_series),
                context_size=self.context_size,
                horizon_size=self.horizon_size,
                series_length=len(full_series),
            ),
            context_size=self.context_size,
            horizon_size=self.horizon_size,
            normalizer=self.normalizer,
        )

    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        return self._dataloader(self._require_dataset(self.train_dataset, "train"))

    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        return self._dataloader(self._require_dataset(self.valid_dataset, "valid"), shuffle=False)

    def test_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        return self._dataloader(self._require_dataset(self.test_dataset, "test"), shuffle=False)

    def _dataloader(
        self,
        dataset: PriceWindowDataset,
        *,
        shuffle: bool = True,
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def _read_split(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Processed split does not exist: {path}")
        frame = pd.read_parquet(path)
        if self.target_col not in frame.columns:
            raise ValueError(f"Missing target column '{self.target_col}' in {path}")
        return frame.sort_values("interval_id").reset_index(drop=True)

    def _target_values(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[self.target_col].to_numpy(dtype=np.float32)

    @staticmethod
    def _require_dataset(
        dataset: PriceWindowDataset | None,
        split_name: str,
    ) -> PriceWindowDataset:
        if dataset is None:
            raise RuntimeError(f"{split_name} dataset has not been set up.")
        return dataset

