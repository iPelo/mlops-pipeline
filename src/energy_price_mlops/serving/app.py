from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from energy_price_mlops.models.baseline import LastValueForecaster

app = FastAPI(title="EnergyPriceMLOps API", version="0.1.0")


class PredictionRequest(BaseModel):
    history: list[float] = Field(..., min_length=1)
    horizon: int = Field(default=24, ge=1, le=168)


class PredictionResponse(BaseModel):
    model_name: str
    predictions: list[float]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> PredictionResponse:
    model = LastValueForecaster(horizon=request.horizon)
    predictions = model.predict(request.history)
    return PredictionResponse(model_name="last_value", predictions=predictions)

