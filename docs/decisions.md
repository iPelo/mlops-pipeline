# Decisions

Use this file as an engineering log. Keep entries short and dated.

## Template

### YYYY-MM-DD - Decision title

Context:

Decision:

Consequence:

## Log

### 2026-05-19 - Naive baselines and the leakage constraint

Context: Phase 1 needs a metric floor before any model is trained. EDA
(`notebooks/01_eda.ipynb`) showed the neighbour-country price columns are
settled in the same day-ahead auction as the target, and the load/generation
columns are realized (post-auction) values.

Decision: Ship three naive baselines (last-value, mean, seasonal-naive-24h) and
score them with rolling-origin windows on the Dec 2024 test split. Treat only
calendar and lagged features as valid model inputs.

Consequence: `seasonal_naive_24h` (MAE 46.21 EUR/MWh) is the bar to beat,
recorded in `docs/model_card.md`. The Phase 2+ model must not consume
contemporaneous prices or realized load/generation.

### 2026-05-19 - DVC initialised

Context: Raw and processed data must stay out of Git on this public repo.

Decision: Ran `dvc init` and disabled anonymous analytics. DVC remote storage
and `dvc add` tracking of the datasets are deferred to a later phase.

Consequence: `.dvc/` metadata is committed; data stays ignored. The `dvc.yaml`
`prepare` stage is ready for `dvc repro`.

### 2026-05-19 - PriceMLP wrapped in a LightningModule

Context: Phase 2 calls for the model architecture plus a Lightning training
loop, kept separate from Hydra wiring (Phase 3).

Decision: `PriceForecastModule` (`training/lightning_module.py`) wraps
`PriceMLP`, owns the MSE loss / logged MAE / AdamW optimizer, and is verified by
a `fast_dev_run` integration test. The Hydra entrypoint and `LightningDataModule`
remain a Phase 3 task.

Consequence: The architecture and optimization logic are testable now; the
end-to-end `train.py` wiring is intentionally still a scaffold.

