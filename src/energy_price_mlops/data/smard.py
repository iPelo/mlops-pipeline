from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from energy_price_mlops.data.features import add_calendar_features

REQUIRED_COLUMNS = {
    "interval_id",
    "timestamp",
    "day_ahead_price_eur_mwh",
}
KEY_COLUMNS = {"interval_id", "timestamp"}


def list_raw_files(raw_data_dir: str | Path = "data/raw", pattern: str = "*.csv") -> list[Path]:
    """Return sorted raw SMARD files from the local data directory."""
    directory = Path(raw_data_dir)
    return sorted(directory.glob(pattern))


def normalize_smard_column(column: str) -> str:
    """Normalize SMARD export headers to stable snake_case names."""
    normalized = column.strip().replace("\ufeff", "")
    normalized = normalized.replace("Calculated resolutions", "")
    normalized = normalized.replace("\u20ac/MWh", "eur_mwh")
    normalized = normalized.replace("\u2205", "average")
    normalized = normalized.replace("[", " ")
    normalized = normalized.replace("]", " ")
    normalized = normalized.replace("/", " ")
    normalized = normalized.replace("-", " ")
    normalized = normalized.replace(".", " ")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = normalized.strip("_")

    aliases = {
        "start_date": "timestamp",
        "end_date": "timestamp_end",
        "germany_luxembourg_eur_mwh": "day_ahead_price_eur_mwh",
        "de_at_lu_eur_mwh": "price_de_at_lu_eur_mwh",
    }
    return aliases.get(normalized, normalized)


def read_smard_csv(
    path: str | Path,
    *,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Read one semicolon-delimited SMARD CSV export."""
    frame = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        na_values=["-", ""],
        dtype=str,
    )
    frame = frame.rename(columns=normalize_smard_column)

    if "timestamp" not in frame.columns:
        raise ValueError(f"SMARD file has no start timestamp column: {path}")

    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        format="%b %d, %Y %I:%M %p",
        errors="raise",
    )
    if "timestamp_end" in frame.columns:
        frame = frame.drop(columns=["timestamp_end"])

    numeric_columns = [column for column in frame.columns if column != "timestamp"]
    for column in numeric_columns:
        values = frame[column].str.replace(",", "", regex=False).str.strip()
        frame[column] = pd.to_numeric(values, errors="coerce")

    if prefix is not None:
        frame = frame.rename(
            columns={
                column: _prefixed_column_name(column, prefix)
                for column in frame.columns
                if column not in KEY_COLUMNS
            }
        )

    frame.insert(0, "interval_id", range(len(frame)))
    return frame


def build_smard_dataset(
    *,
    actual_consumption_path: str | Path,
    actual_generation_path: str | Path,
    day_ahead_prices_path: str | Path,
) -> pd.DataFrame:
    """Merge the raw SMARD exports into one modeling table."""
    prices = read_smard_csv(day_ahead_prices_path, prefix="price")
    consumption = read_smard_csv(actual_consumption_path, prefix="consumption")
    generation = read_smard_csv(actual_generation_path, prefix="generation")

    _validate_aligned_intervals(
        reference=prices,
        frames={
            "actual_consumption": consumption,
            "actual_generation": generation,
        },
    )

    consumption = consumption.drop(columns=["timestamp"])
    generation = generation.drop(columns=["timestamp"])

    dataset = prices.merge(consumption, on="interval_id", how="inner", validate="one_to_one")
    dataset = dataset.merge(generation, on="interval_id", how="inner", validate="one_to_one")
    dataset = dataset.sort_values("interval_id").reset_index(drop=True)
    dataset = add_calendar_features(dataset)

    validate_required_columns(set(dataset.columns))
    return dataset


def split_by_timestamp(
    frame: pd.DataFrame,
    *,
    train_start: str,
    valid_start: str,
    test_start: str,
    test_end: str,
) -> dict[str, pd.DataFrame]:
    """Split a time-series frame without shuffling."""
    timestamp = frame["timestamp"]
    train_start_ts = pd.Timestamp(train_start)
    valid_start_ts = pd.Timestamp(valid_start)
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)

    splits = {
        "train": frame[(timestamp >= train_start_ts) & (timestamp < valid_start_ts)],
        "valid": frame[(timestamp >= valid_start_ts) & (timestamp < test_start_ts)],
        "test": frame[(timestamp >= test_start_ts) & (timestamp < test_end_ts)],
    }
    empty_splits = [name for name, split in splits.items() if split.empty]
    if empty_splits:
        split_names = ", ".join(empty_splits)
        raise ValueError(f"Empty SMARD split(s): {split_names}")
    return {name: split.reset_index(drop=True) for name, split in splits.items()}


def write_splits(splits: dict[str, pd.DataFrame], output_paths: dict[str, Path]) -> None:
    """Write train, validation, and test splits as parquet files."""
    for split_name, frame in splits.items():
        output_path = output_paths[split_name]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output_path, index=False)


def validate_required_columns(columns: set[str]) -> None:
    """Validate that a processed SMARD table has the minimum required columns."""
    missing = REQUIRED_COLUMNS - columns
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required SMARD columns: {missing_text}")


def _prefixed_column_name(column: str, prefix: str) -> str:
    if column == "day_ahead_price_eur_mwh":
        return column
    if column.startswith(f"{prefix}_"):
        return column
    return f"{prefix}_{column}"


def _validate_aligned_intervals(
    *,
    reference: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
) -> None:
    for name, frame in frames.items():
        if len(reference) != len(frame):
            raise ValueError(
                f"SMARD file row count mismatch for {name}: "
                f"expected {len(reference):,}, got {len(frame):,}"
            )

        mismatched_timestamps = reference["timestamp"].ne(frame["timestamp"])
        if mismatched_timestamps.any():
            first_mismatch = int(mismatched_timestamps.idxmax())
            raise ValueError(
                f"SMARD file timestamp mismatch for {name} at row {first_mismatch}: "
                f"expected {reference.at[first_mismatch, 'timestamp']}, "
                f"got {frame.at[first_mismatch, 'timestamp']}"
            )
