# EnergyPriceMLOps

EnergyPriceMLOps is a public MLOps project for forecasting German day-ahead
electricity prices from SMARD market data.

The repo is built around the full workflow, not only the model: data prep,
versioned pipeline steps, baseline checks, Lightning training, ONNX export,
FastAPI serving, and monitoring notes.

## Project Status

- Status: in development
- Dataset: SMARD electricity market data, hourly 2024 export
- Task: 24-hour electricity price forecasting

## Stack

- Python 3.11
- PyTorch and Lightning
- Hydra
- DVC
- Weights & Biases
- FastAPI and ONNX Runtime
- Docker
- GitHub Actions for CI
- Evidently or custom drift checks

## Repository Structure

```text
mlops-pipeline/
├── configs/                  # Hydra configuration
│   ├── config.yaml
│   ├── data/
│   ├── model/
│   ├── optim/
│   └── trainer/
├── data/                     # Local data, ignored by Git
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
├── docs/
├── notebooks/
├── scripts/
├── src/
│   └── energy_price_mlops/
│       ├── data/
│       ├── eval/
│       ├── models/
│       ├── serving/
│       └── training/
└── tests/
```

Raw data, processed tables, checkpoints, experiment outputs, and exported model
files are intentionally left out of Git. They belong in local storage or DVC
remote storage.

## PyCharm Professional Setup

1. Open this folder in PyCharm Professional.
2. Create a Python 3.11 virtual environment named `.venv`.
3. Install `uv` if needed:

   ```bash
   pip install uv
   ```

4. Install project dependencies:

   ```bash
   uv sync --all-extras
   ```

5. In PyCharm, set the interpreter to `.venv/bin/python`.
6. Mark `src/` as a Sources Root if PyCharm does not detect it automatically.
7. Use the built-in terminal for project commands:

   ```bash
   make test
   make lint
   make train
   make serve
   ```

## Data

Expected raw files:

- `data/raw/Actual_consumption_202401010000_202501010000_Hour.csv`
- `data/raw/Actual_generation_202401010000_202501010000_Hour.csv`
- `data/raw/Day-ahead_prices_202401010000_202501010000_Hour.csv`

The processed table has one row per hour and keeps both `interval_id` and the
local SMARD timestamp. `interval_id` is the safe modeling key because local time
repeats during the Europe/Berlin DST fallback hour.

Prepare the first processed splits with:

```bash
uv run python scripts/prepare_data.py --config configs/data/smard.yaml
```

This creates:

- `data/processed/train.parquet` for January through October 2024
- `data/processed/valid.parquet` for November 2024
- `data/processed/test.parquet` for December 2024

## Training

Run the default training job:

```bash
uv run python -m energy_price_mlops.training.train
```

The current best local run is `mlp_default`, a univariate MLP that uses 168
hours of price history to forecast the next 24 hours. On the December 2024 test
split it reaches MAE `45.22` EUR/MWh and RMSE `65.99` EUR/MWh. The seasonal
naive baseline is MAE `46.21` EUR/MWh.

## Serving

Export a local checkpoint to ONNX:

```bash
uv run python -m energy_price_mlops.export_onnx \
  --metrics artifacts/training/mlp_default/metrics.json \
  --output artifacts/models/price_mlp.onnx
```

The exported graph includes target normalization, so callers send price history
in EUR/MWh. Start the API with:

```bash
make serve
```

The API exposes `/health`, `/metrics`, and `/predict`. If the ONNX file is not
available locally, predictions fall back to the last-value baseline.
