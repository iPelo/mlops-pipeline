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

Record the exact SMARD export settings here:

- Market area: Germany/Luxembourg for day-ahead prices
- Date range: 2024-01-01 00:00 through 2025-01-01 00:00
- Resolution: Hour
- Download date: TODO
- Source URL: TODO
- License or usage notes: TODO
