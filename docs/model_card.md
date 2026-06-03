# Model Card

## Model Details

- Project: EnergyPriceMLOps
- Task: German day-ahead electricity price forecasting
- Dataset: SMARD electricity market data

## Intended Use

This model forecasts short-horizon German day-ahead electricity prices for a
public engineering project. It is useful for testing the pipeline, comparing
baselines, and exercising serving infrastructure.

## Limitations

Do not use this model for trading, financial decisions, or grid operations. It
uses a small feature set, a single year of data, and a narrow validation setup.
Any production use would need broader data, stricter backtesting, monitoring,
and governance.

## Metrics

Evaluation uses rolling-origin windows on the December 2024 test split:
168-hour context, 24-hour forecast horizon, 721 windows, and pooled predictions
across all windows.

MAPE is omitted because day-ahead prices can be zero or negative.

### Baselines

| Model | MAE (EUR/MWh) | RMSE (EUR/MWh) |
|---|---|---|
| `mean` | 61.80 | 98.63 |
| `last_value` | 48.14 | 92.75 |
| `seasonal_naive_24h` | **46.21** | **81.96** |

`seasonal_naive_24h` repeats the previous day's 24-hour price profile and is
the main baseline for this repo.

### First MLP Experiments

The first completed training runs used lagged target prices only. They do not
use realized load/generation or neighbouring market prices from the forecast
period.

| Run | Hidden sizes | Dropout | LR | Best val MAE | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_default` | 256, 128 | 0.10 | 0.0003 | **28.73** | **45.22** | **65.99** |
| `mlp_no_dropout_lr1e3` | 256, 128 | 0.00 | 0.0010 | 28.84 | 46.25 | 66.17 |
| `mlp_small_lr1e3` | 128, 64 | 0.00 | 0.0010 | 29.76 | 46.46 | 67.46 |

`mlp_default` is the current best trained model. The improvement over the
seasonal naive baseline is small, so the next useful work is validation,
leakage-safe feature work, and error analysis.

### Multifeature Training Path

The current training code can build rolling windows from 11 historical features:
target price, grid load, residual load, offshore wind, onshore wind, solar,
fossil gas generation, hour, day of week, month, and weekend flag. A one-epoch
smoke run validated the path and produced December test MAE `45.17` EUR/MWh,
but this is not a full experiment.

## Export

The export path writes ONNX metadata with context length, flattened input size,
feature columns, and target scaling. The latest multifeature smoke export used
input size `1848` (`168 * 11`) and produced max absolute Torch-vs-ONNX error
`1.83e-04`.
