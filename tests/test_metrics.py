import pytest

from energy_price_mlops.eval.metrics import mean_absolute_error, root_mean_squared_error


def test_mean_absolute_error() -> None:
    assert mean_absolute_error([1.0, 2.0, 3.0], [1.0, 4.0, 2.0]) == pytest.approx(1.0)


def test_root_mean_squared_error() -> None:
    assert root_mean_squared_error([1.0, 2.0, 3.0], [1.0, 4.0, 2.0]) == pytest.approx(
        (5 / 3) ** 0.5
    )

