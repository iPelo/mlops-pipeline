from pathlib import Path

import pandas as pd

from energy_price_mlops.data import SmardPriceDataModule


def _write_split(path: Path, start_interval: int, rows: int) -> None:
    frame = pd.DataFrame(
        {
            "interval_id": range(start_interval, start_interval + rows),
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="h"),
            "day_ahead_price_eur_mwh": [float(start_interval + index) for index in range(rows)],
        }
    )
    frame.to_parquet(path, index=False)


def test_smard_price_datamodule_builds_rolling_origin_splits(tmp_path: Path) -> None:
    train_path = tmp_path / "train.parquet"
    valid_path = tmp_path / "valid.parquet"
    test_path = tmp_path / "test.parquet"
    _write_split(train_path, start_interval=0, rows=20)
    _write_split(valid_path, start_interval=20, rows=8)
    _write_split(test_path, start_interval=28, rows=8)

    data_module = SmardPriceDataModule(
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path,
        target_col="day_ahead_price_eur_mwh",
        context_size=4,
        horizon_size=3,
        batch_size=2,
    )

    data_module.setup("fit")

    assert data_module.normalizer is not None
    assert data_module.train_dataset is not None
    assert data_module.valid_dataset is not None
    assert data_module.test_dataset is not None
    assert len(data_module.train_dataset) == 14
    assert len(data_module.valid_dataset) == 6
    assert len(data_module.test_dataset) == 6

    inputs, targets = next(iter(data_module.train_dataloader()))
    assert inputs.shape == (2, 4)
    assert targets.shape == (2, 3)
