from pathlib import Path

import pandas as pd

from energy_price_mlops.data.smard import (
    build_smard_dataset,
    normalize_smard_column,
    split_by_timestamp,
)


def test_normalize_smard_column_maps_target_price() -> None:
    assert (
        normalize_smard_column("Germany/Luxembourg [\u20ac/MWh] Calculated resolutions")
        == "day_ahead_price_eur_mwh"
    )


def test_split_by_timestamp_uses_ordered_time_windows() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01 00:00", "2024-11-01 00:00", "2024-12-01 00:00"]
            ),
            "day_ahead_price_eur_mwh": [1.0, 2.0, 3.0],
        }
    )

    splits = split_by_timestamp(
        frame,
        train_start="2024-01-01",
        valid_start="2024-11-01",
        test_start="2024-12-01",
        test_end="2025-01-01",
    )

    assert len(splits["train"]) == 1
    assert len(splits["valid"]) == 1
    assert len(splits["test"]) == 1


def test_build_smard_dataset_preserves_duplicate_dst_hour(tmp_path: Path) -> None:
    price_path = tmp_path / "prices.csv"
    consumption_path = tmp_path / "consumption.csv"
    generation_path = tmp_path / "generation.csv"

    rows = "\n".join(
        [
            "Oct 27, 2024 1:00 AM;Oct 27, 2024 2:00 AM;10.00",
            "Oct 27, 2024 2:00 AM;Oct 27, 2024 3:00 AM;20.00",
            "Oct 27, 2024 2:00 AM;Oct 27, 2024 3:00 AM;30.00",
        ]
    )
    price_path.write_text(
        "Start date;End date;Germany/Luxembourg [\u20ac/MWh] Calculated resolutions\n"
        f"{rows}\n",
        encoding="utf-8",
    )
    consumption_path.write_text(
        "Start date;End date;grid load [MWh] Calculated resolutions\n"
        f"{rows}\n",
        encoding="utf-8",
    )
    generation_path.write_text(
        "Start date;End date;Biomass [MWh] Calculated resolutions\n"
        f"{rows}\n",
        encoding="utf-8",
    )

    dataset = build_smard_dataset(
        actual_consumption_path=consumption_path,
        actual_generation_path=generation_path,
        day_ahead_prices_path=price_path,
    )

    assert len(dataset) == 3
    assert dataset["timestamp"].duplicated().sum() == 1
    assert dataset["day_ahead_price_eur_mwh"].tolist() == [10.0, 20.0, 30.0]
