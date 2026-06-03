from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class TargetNormalizer:
    mean: float
    std: float

    def transform_array(self, values: FloatArray) -> FloatArray:
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse_transform_tensor(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std + self.mean


@dataclass(frozen=True)
class FeatureNormalizer:
    means: FloatArray
    stds: FloatArray

    def transform_array(self, values: FloatArray) -> FloatArray:
        return ((values - self.means) / self.stds).astype(np.float32)


class PriceWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Fixed-window dataset for multi-step price forecasting."""

    def __init__(
        self,
        features: FloatArray,
        targets: FloatArray,
        forecast_start_indices: IntArray,
        *,
        context_size: int,
        horizon_size: int,
        feature_normalizer: FeatureNormalizer,
        target_normalizer: TargetNormalizer,
    ) -> None:
        if context_size <= 0:
            raise ValueError("context_size must be positive.")
        if horizon_size <= 0:
            raise ValueError("horizon_size must be positive.")
        if len(features) == 0 or len(targets) == 0:
            raise ValueError("features and targets must not be empty.")
        if len(features) != len(targets):
            raise ValueError("features and targets must have the same number of rows.")
        if features.ndim != 2:
            raise ValueError("features must be a 2D array.")
        if len(forecast_start_indices) == 0:
            raise ValueError("forecast_start_indices must not be empty.")

        self.features = feature_normalizer.transform_array(features)
        self.targets = target_normalizer.transform_array(targets)
        self.forecast_start_indices = forecast_start_indices
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.num_features = features.shape[1]

        first_index = int(forecast_start_indices.min())
        last_index = int(forecast_start_indices.max())
        if first_index < context_size:
            raise ValueError("forecast_start_indices must leave enough context.")
        if last_index + horizon_size > len(targets):
            raise ValueError("forecast_start_indices exceed the available horizon.")

    def __len__(self) -> int:
        return len(self.forecast_start_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        forecast_start = int(self.forecast_start_indices[index])
        context_start = forecast_start - self.context_size
        forecast_end = forecast_start + self.horizon_size

        inputs = torch.from_numpy(self.features[context_start:forecast_start].reshape(-1))
        targets = torch.from_numpy(self.targets[forecast_start:forecast_end])
        return inputs, targets


def build_forecast_start_indices(
    *,
    first_forecast_start: int,
    last_forecast_start_exclusive: int,
    context_size: int,
    horizon_size: int,
    series_length: int,
) -> IntArray:
    """Build valid forecast-start indices for rolling-origin windows."""
    start = max(first_forecast_start, context_size)
    stop = min(last_forecast_start_exclusive, series_length - horizon_size + 1)
    if stop <= start:
        raise ValueError("No valid rolling-origin windows for the requested split.")
    return np.arange(start, stop, dtype=np.int64)
