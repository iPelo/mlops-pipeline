from energy_price_mlops.models.baseline import LastValueForecaster


def test_last_value_forecaster_repeats_latest_value() -> None:
    model = LastValueForecaster(horizon=3)

    assert model.predict([10.0, 20.0, 30.0]) == [30.0, 30.0, 30.0]

