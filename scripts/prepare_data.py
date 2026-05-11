from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from energy_price_mlops.data.smard import build_smard_dataset, split_by_timestamp, write_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SMARD data for modeling.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the data config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.config.exists():
        raise FileNotFoundError(f"Config file does not exist: {args.config}")

    config = OmegaConf.load(args.config)
    dataset = build_smard_dataset(
        actual_consumption_path=config.files.actual_consumption,
        actual_generation_path=config.files.actual_generation,
        day_ahead_prices_path=config.files.day_ahead_prices,
    )
    splits = split_by_timestamp(
        dataset,
        train_start=config.split.train_start,
        valid_start=config.split.valid_start,
        test_start=config.split.test_start,
        test_end=config.split.test_end,
    )
    output_paths = {
        "train": Path(config.files.processed_train),
        "valid": Path(config.files.processed_valid),
        "test": Path(config.files.processed_test),
    }
    write_splits(splits, output_paths)

    print(f"Prepared SMARD dataset with {len(dataset):,} hourly rows.")
    for split_name, split_frame in splits.items():
        print(f"{split_name}: {len(split_frame):,} rows -> {output_paths[split_name]}")


if __name__ == "__main__":
    main()
