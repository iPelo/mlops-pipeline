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

