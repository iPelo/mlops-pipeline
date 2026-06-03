from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI
from pydantic import BaseModel, Field

from energy_price_mlops.models.baseline import LastValueForecaster
from energy_price_mlops.serving.model import OnnxPriceForecaster, PredictionStats

app = FastAPI(title="EnergyPriceMLOps API", version="0.1.0")
stats = PredictionStats()
MODEL_PATH = Path(os.getenv("ENERGY_PRICE_ONNX_PATH", "artifacts/models/price_mlp.onnx"))
MODEL = OnnxPriceForecaster(MODEL_PATH) if MODEL_PATH.exists() else None


class PredictionRequest(BaseModel):
    history: list[float] = Field(..., min_length=1)
    feature_history: list[list[float]] | None = None
    horizon: int = Field(default=24, ge=1, le=168)


class PredictionResponse(BaseModel):
    model_name: str
    predictions: list[float]
    fallback: bool = False


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model_name": MODEL.model_name if MODEL is not None else "last_value",
    }


@app.get("/metrics")
def metrics() -> dict[str, float | int | str | bool | None]:
    return {
        "model_loaded": MODEL is not None,
        "model_name": MODEL.model_name if MODEL is not None else "last_value",
        **stats.as_dict(),
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> PredictionResponse:
    start_time = perf_counter()
    if MODEL is not None and request.horizon == MODEL.horizon_size:
        model_features = _model_features(request)
        if model_features is not None:
            predictions = MODEL.predict(model_features)
            stats.observe(start_time)
            return PredictionResponse(model_name=MODEL.model_name, predictions=predictions)

    model = LastValueForecaster(horizon=request.horizon)
    predictions = model.predict(request.history)
    stats.observe(start_time)
    return PredictionResponse(model_name="last_value", predictions=predictions, fallback=True)


def _model_features(request: PredictionRequest) -> list[float] | None:
    if MODEL is None:
        return None
    if MODEL.num_features == 1:
        return request.history
    if request.feature_history is None:
        return None
    if len(request.feature_history) < MODEL.context_size:
        return None
    recent_rows = request.feature_history[-MODEL.context_size :]
    if any(len(row) != MODEL.num_features for row in recent_rows):
        return None
    return [float(value) for row in recent_rows for value in row]
