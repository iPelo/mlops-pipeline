import pytest

from energy_price_mlops.models.baseline import (
    LastValueForecaster,
    MeanForecaster,
    SeasonalNaiveForecaster,
)


def test_last_value_forecaster_repeats_latest_value() -> None:
    model = LastValueForecaster(horizon=3)

    assert model.predict([10.0, 20.0, 30.0]) == [30.0, 30.0, 30.0]


def test_mean_forecaster_repeats_history_mean() -> None:
    model = MeanForecaster(horizon=2)

    assert model.predict([10.0, 20.0, 30.0]) == [20.0, 20.0]


def test_seasonal_naive_forecaster_repeats_last_season() -> None:
    model = SeasonalNaiveForecaster(horizon=4, season_length=2)

    assert model.predict([1.0, 2.0, 3.0, 4.0]) == [3.0, 4.0, 3.0, 4.0]


def test_seasonal_naive_forecaster_rejects_short_history() -> None:
    model = SeasonalNaiveForecaster(horizon=2, season_length=24)

    with pytest.raises(ValueError, match="at least 24 values"):
        model.predict([1.0, 2.0, 3.0])
