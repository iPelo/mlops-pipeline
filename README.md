# EnergyPriceMLOps

End-to-end MLOps project for forecasting German electricity market prices from
SMARD electricity market data.

The goal is not just to train a model. The goal is to make the full workflow
reproducible: data versioning, feature generation, experiment tracking, model
training, ONNX export, FastAPI serving, deployment, and drift monitoring.

## Project Status

Status: In development

Dataset: SMARD electricity market data, hourly 2024 export

Task: Time-series forecasting for day-ahead electricity prices

## Stack

- Python 3.11
- PyTorch + Lightning for model training
- Hydra for configuration
- DVC for data and artifact versioning
- Weights & Biases for experiment tracking
- FastAPI + ONNX Runtime for serving
- Docker for train and serving images
- GitHub Actions for CI
- Evidently or custom checks for drift monitoring

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

## Data Layout

The current raw dataset contains hourly 2024 SMARD exports:

- `data/raw/Actual_consumption_202401010000_202501010000_Hour.csv`
- `data/raw/Actual_generation_202401010000_202501010000_Hour.csv`
- `data/raw/Day-ahead_prices_202401010000_202501010000_Hour.csv`

The prepared model table uses one row per hourly interval. It keeps both the
monotonic `interval_id` and the local SMARD `timestamp`; the local timestamp can
repeat during the Europe/Berlin DST fallback hour.

- `interval_id`
- `timestamp`
- `day_ahead_price_eur_mwh`

Recommended extra features:

- actual load
- forecast load
- wind generation
- solar generation
- residual load
- cross-border exchange features
- calendar features

Prepare the first processed splits with:

```bash
uv run python scripts/prepare_data.py --config configs/data/smard.yaml
```

This creates:

- `data/processed/train.parquet` for January-October 2024
- `data/processed/valid.parquet` for November 2024
- `data/processed/test.parquet` for December 2024

## First Milestones

1. Put the raw SMARD files in `data/raw/`.
2. Build a clean processed table in `data/processed/`.
3. Commit a dumb baseline, such as last-value or seasonal naive.
4. Add the first Lightning model.
5. Track experiments in W&B.
6. Export the best model to ONNX.
7. Serve the model through FastAPI.
