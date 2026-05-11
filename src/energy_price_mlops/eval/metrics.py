from __future__ import annotations

from collections.abc import Sequence
from math import sqrt


def _validate_inputs(y_true: Sequence[float], y_pred: Sequence[float]) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if not y_true:
        raise ValueError("Metric inputs must not be empty.")


def mean_absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    _validate_inputs(y_true, y_pred)
    absolute_errors = (
        abs(actual - predicted) for actual, predicted in zip(y_true, y_pred, strict=True)
    )
    return sum(absolute_errors) / len(y_true)


def root_mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    _validate_inputs(y_true, y_pred)
    squared_errors = (
        (actual - predicted) ** 2 for actual, predicted in zip(y_true, y_pred, strict=True)
    )
    mse = sum(squared_errors) / len(y_true)
    return sqrt(mse)


def mean_absolute_percentage_error(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    epsilon: float = 1e-8,
) -> float:
    _validate_inputs(y_true, y_pred)
    errors = [
        abs((actual - predicted) / max(abs(actual), epsilon))
        for actual, predicted in zip(y_true, y_pred, strict=True)
    ]
    return sum(errors) / len(errors)
