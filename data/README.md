# Data

This project uses SMARD electricity market data for German electricity price
forecasting.

## Local Layout

```text
data/
├── raw/          # Original downloaded/exported SMARD files
├── interim/      # Cleaned but not modeling-ready files
├── processed/    # Train/valid/test model tables
└── external/     # Optional weather, holidays, or other joined data
```

The data folders are ignored by Git. Track large files with DVC once the first
usable raw and processed datasets exist.

## Current Raw Files

The project currently contains hourly 2024 SMARD exports:

- `data/raw/Actual_consumption_202401010000_202501010000_Hour.csv`
- `data/raw/Actual_generation_202401010000_202501010000_Hour.csv`
- `data/raw/Day-ahead_prices_202401010000_202501010000_Hour.csv`

Each file covers `2024-01-01 00:00` through `2025-01-01 00:00` as hourly
intervals.

## Target Schema

The processed table uses one row per hourly interval. It keeps `interval_id` as
a monotonic key because the local SMARD timestamp repeats during the
Europe/Berlin DST fallback hour.

Required columns:

- `interval_id`
- `timestamp`
- `day_ahead_price_eur_mwh`

Recommended columns:

- `load_actual_mw`
- `load_forecast_mw`
- `wind_generation_mw`
- `solar_generation_mw`
- `residual_load_mw`
- `hour`
- `day_of_week`
- `month`
- `is_weekend`

## Source Notes

SMARD export settings for the raw files in this project:

- Provider: Bundesnetzagentur (German Federal Network Agency), via SMARD.de
- Market area: Germany/Luxembourg for day-ahead prices; Germany for load and
  generation
- Date range: 2024-01-01 00:00 through 2025-01-01 00:00 (Europe/Berlin)
- Resolution: Hour
- Download date: 2026-05-12
- Download center: <https://www.smard.de/en/downloadcenter/download-market-data>
- Datasets used: Day-ahead prices, Actual consumption, Actual generation

## License and Attribution

SMARD market data is published by the Bundesnetzagentur under the
**Creative Commons Attribution 4.0 International (CC BY 4.0)** license
(<https://creativecommons.org/licenses/by/4.0/>). SMARD documents this on its
data-use page: <https://www.smard.de/en/datennutzung>.

Required attribution when using or redistributing this data:

> Source: Bundesnetzagentur | SMARD.de — licensed under CC BY 4.0.

The raw CSV files themselves are not committed to this repository (see the
top-level `.gitignore`); only documentation and DVC pipeline metadata are
tracked. The project code (MIT licensed) and the SMARD data (CC BY 4.0) carry
separate licenses.
