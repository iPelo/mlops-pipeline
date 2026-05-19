# Model Card

## Model Details

Project: EnergyPriceMLOps

Task: German day-ahead electricity price forecasting

Dataset: SMARD electricity market data

## Intended Use

Forecast short-horizon electricity prices for MLOps demonstration and portfolio
purposes.

## Limitations

This model should not be used for real trading, financial decisions, or grid
operations without stronger validation, monitoring, and governance.

## Metrics

The trained model's metrics will be added here after the first real training
run. Until then, this section records the naive baselines it must beat.

### Baselines

Rolling-origin evaluation on the December 2024 test split: 168-hour context,
24-hour forecast horizon, 721 windows, predictions pooled across all windows.

| Model | MAE (EUR/MWh) | RMSE (EUR/MWh) |
|---|---|---|
| `mean` | 61.80 | 98.63 |
| `last_value` | 48.14 | 92.75 |
| `seasonal_naive_24h` | **46.21** | **81.96** |

`seasonal_naive_24h` (repeat the previous day's 24-hour price profile) is the
baseline to beat. MAPE is intentionally omitted: day-ahead prices cross zero
and go negative, so percentage error is not a meaningful metric for this
target. See `notebooks/02_baseline.ipynb` for the full evaluation.

