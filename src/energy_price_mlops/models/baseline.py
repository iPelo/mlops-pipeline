from __future__ import annotations

from collections.abc import Sequence


class LastValueForecaster:
    """Naive baseline that repeats the latest observed value."""

    def __init__(self, horizon: int) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive.")
        self.horizon = horizon

    def predict(self, history: Sequence[float]) -> list[float]:
        if not history:
            raise ValueError("history must contain at least one value.")
        return [float(history[-1])] * self.horizon


class MeanForecaster:
    """Naive baseline that repeats the mean of the observed history."""

    def __init__(self, horizon: int) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive.")
        self.horizon = horizon

    def predict(self, history: Sequence[float]) -> list[float]:
        if not history:
            raise ValueError("history must contain at least one value.")
        mean_value = sum(history) / len(history)
        return [float(mean_value)] * self.horizon


class SeasonalNaiveForecaster:
    """Naive baseline that repeats the most recent seasonal cycle.

    For hourly day-ahead prices the natural season is one day (24 hours):
    the forecast for the next ``horizon`` hours reuses the matching hours
    from one season ago. This is the honest baseline to beat for data with
    a strong daily price profile.
    """

    def __init__(self, horizon: int, season_length: int = 24) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive.")
        if season_length <= 0:
            raise ValueError("season_length must be positive.")
        self.horizon = horizon
        self.season_length = season_length

    def predict(self, history: Sequence[float]) -> list[float]:
        if len(history) < self.season_length:
            raise ValueError(
                f"history must contain at least {self.season_length} values."
            )
        season = [float(value) for value in history[-self.season_length :]]
        return [season[index % self.season_length] for index in range(self.horizon)]

