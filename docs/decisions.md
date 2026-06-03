# Decisions

Short engineering notes for choices that affect the shape of the project.

## Log

### 2026-05-19 - Naive baselines and the leakage constraint

Before training anything, the project needed a metric floor. EDA
(`notebooks/01_eda.ipynb`) showed that neighbour-country price columns are
settled in the same day-ahead auction as the target. Load and generation
columns are realized after the auction.

The baseline set is `last_value`, `mean`, and `seasonal_naive_24h`, evaluated
with rolling-origin windows on the December 2024 test split. Models should use
calendar features and lagged signals only unless a feature is clearly available
before the forecast is made.

`seasonal_naive_24h` is the main baseline to beat, with MAE 46.21 EUR/MWh. The
first trained model does not consume contemporaneous prices or realized
load/generation.

### 2026-05-19 - DVC initialised

Raw and processed data must stay out of Git on this public repo.

DVC is initialized and anonymous analytics are disabled. Remote storage and
`dvc add` tracking are left for the next data-management pass.

`.dvc/` metadata is committed, data directories stay ignored, and `dvc.yaml`
can run the `prepare` stage through `dvc repro`.

### 2026-05-19 - PriceMLP wrapped in a LightningModule

The model architecture needed a testable Lightning wrapper before the full
Hydra training command was wired.

`PriceForecastModule` wraps `PriceMLP`, owns MSE loss, logged MAE, and the AdamW
optimizer, and is covered by a `fast_dev_run` integration test.

The model and optimizer path are testable on their own, which made the later
training entrypoint easier to finish.

### 2026-05-25 - First real MLP training run

The training entrypoint now runs end to end, so the project has its first model
comparison against the seasonal naive baseline.

The first training pass uses a univariate rolling-window `LightningDataModule`,
train-only target normalization, and original-scale MAE/RMSE logging. Artifacts
and checkpoints are written under ignored `artifacts/training/` directories.

`mlp_default` is the best run so far: validation MAE 28.73, December test MAE
45.22, and December test RMSE 65.99. It beats the seasonal naive MAE baseline,
but the margin is small enough that feature work and error analysis matter more
than tuning the current MLP.

### 2026-06-01 - ONNX export and serving foundation

The trained model needs a reproducible export path and an API that can use the
same preprocessing assumptions as training.

The ONNX export wraps target normalization inside the graph. The FastAPI app
loads `artifacts/models/price_mlp.onnx` when it exists, exposes `/metrics`, and
uses the last-value baseline when the model file is missing or the request asks
for an unsupported horizon.

The current local ONNX export has max absolute Torch-vs-ONNX error `1.14e-05`.
Model artifacts stay ignored by Git and can be recreated from the local
checkpoint plus `metrics.json`.

### 2026-06-02 - DVC data pointers and multifeature windows

The raw SMARD files are now tracked with DVC pointer files, and the processed
parquet outputs are committed to the DVC pipeline cache. The default DVC remote
is local-only, so public Git history contains metadata but not data bytes.

The training data module now accepts configured feature columns and normalizes
features from train-split statistics. The first default feature set uses price,
load, residual load, selected generation columns, and calendar features. The
serving export writes feature metadata so ONNX inference can validate the
flattened input window.
