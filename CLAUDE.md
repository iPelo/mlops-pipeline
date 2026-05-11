# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**EnergyPriceMLOps** — end-to-end MLOps pipeline for German day-ahead electricity
price forecasting from hourly SMARD electricity market data. The goal is full
workflow reproducibility (data versioning, feature generation, experiment
tracking, ONNX export, FastAPI serving, drift monitoring), not just a model.

This repo is intended to be **public**. Do not commit secrets, local IDE files,
virtual environments, raw or processed data, model checkpoints, W&B output, or
deployment credentials. See `.gitignore` for the enforced list.

## Commands

Dependencies are managed with `uv`. The `Makefile` wraps the common flows.

```bash
make install        # uv sync --all-extras
make lint           # ruff check + mypy on src/ and tests/
make format         # ruff format
make test           # pytest
make train          # python -m energy_price_mlops.training.train
make serve          # uvicorn on :8000
```

Single-test invocation:

```bash
uv run pytest tests/test_smard.py::test_split_by_timestamp_uses_ordered_time_windows
```

Pre-handoff validation (these are what CI runs):

```bash
uv run --extra dev pytest
uv run --extra dev ruff check .
uv run --extra dev mypy src tests
```

Last known passing state: `6 passed`, ruff clean, `mypy: no issues found in 16 source files`.

Regenerate processed splits from raw CSVs:

```bash
uv run python scripts/prepare_data.py --config configs/data/smard.yaml
```

Expected output: `8,784` total rows → train `7,320` (Jan–Oct 2024) / valid `720`
(Nov 2024) / test `744` (Dec 2024), `39` columns. The same command is wired as
the `prepare` DVC stage in `dvc.yaml`.

## Architecture

The Python package lives under `src/energy_price_mlops/` (installed as a wheel
via `hatchling`; `pyproject.toml` pins it to that path). The four subpackages
mirror pipeline stages:

- **`data/`** — `smard.py` reads the three SMARD CSV exports (consumption,
  generation, prices), normalizes headers (`normalize_smard_column` handles the
  semicolon delimiter, € symbol, and column aliases like
  `germany_luxembourg_eur_mwh` → `day_ahead_price_eur_mwh`), assigns a synthetic
  `interval_id`, merges on it, and writes parquet splits. `features.py` adds
  deterministic calendar features and lag helpers.
- **`models/`** — `baseline.py` (`LastValueForecaster`) is the naive baseline
  used both in tests and currently by the serving endpoint. `mlp.py`
  (`PriceMLP`) is the first Lightning-targeted model.
- **`training/`** — `train.py` is a Hydra entrypoint that loads
  `configs/config.yaml`. It is currently a scaffold; the `LightningDataModule`
  and `LightningModule` are not implemented yet.
- **`serving/`** — `app.py` is the FastAPI app (`/health`, `/predict`).
  `/predict` currently wraps `LastValueForecaster` directly — there is no ONNX
  loading yet.
- **`eval/`** — `metrics.py` has MAE, RMSE, MAPE implementations.

Configuration is Hydra-composed from `configs/`:

```
configs/config.yaml          # defaults: data=smard, model=baseline, optim=adamw, trainer=default
configs/data/smard.yaml      # paths, split dates, forecast horizon, file locations
configs/model/{baseline,mlp}.yaml
configs/optim/adamw.yaml
configs/trainer/default.yaml
```

The training entrypoint resolves the config tree via `@hydra.main(config_path="../../../configs", config_name="config")`.

### Critical data detail: SMARD DST fallback

SMARD local timestamps **repeat** during the Europe/Berlin DST fallback hour
(e.g. `2024-10-27 02:00` appears twice). The pipeline therefore uses
`interval_id` as the stable monotonic key and keeps the local `timestamp` as a
feature/time label. **Do not merge raw files only on `timestamp`** — that
collapses the duplicate fallback hour and corrupts the dataset.
`build_smard_dataset` validates row alignment before merging on `interval_id`,
and `test_build_smard_dataset_preserves_duplicate_dst_hour` guards this
invariant.

## Public-repo safety

Tracked and meant to be public:
- `src/`, `tests/`, `configs/`, `scripts/`, `docs/`, READMEs
- `pyproject.toml`, `uv.lock`, `Makefile`, `Dockerfile.*`
- `dvc.yaml`, `.github/workflows/`, `.pre-commit-config.yaml`, `.python-version`

Must stay ignored (already covered by `.gitignore`):
- `.venv/`, `.idea/`, `.vscode/`, `.DS_Store`
- `.env`, `.env.*`, `*.pem`, `*.key`, `.netrc`, `secrets/`
- `data/raw/**`, `data/processed/**`, `data/interim/**`, `data/external/**`
  (only `.gitkeep` and `*.dvc` are kept under `data/`)
- `wandb/`, `lightning_logs/`, `mlruns/`, `outputs/`, `artifacts/`, `checkpoints/`
- Model export formats: `*.ckpt`, `*.onnx`, `*.pt`, `*.pth`, `*.pkl`, `*.joblib`, `*.safetensors`
- `.dvc/cache/`, `.dvc/tmp/`, `.dvc/config.local` (commit `.dvc` metadata files
  and `dvc.lock`, never remote credentials or cache contents)

Before committing anything new, double-check it does not leak local paths,
W&B API keys, or model artifacts.
