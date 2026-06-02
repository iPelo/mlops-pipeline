import json
from pathlib import Path

import torch
from fastapi.testclient import TestClient

from energy_price_mlops.export_onnx import ServingPriceModel
from energy_price_mlops.models.mlp import PriceMLP
from energy_price_mlops.serving.app import app
from energy_price_mlops.serving.model import OnnxPriceForecaster


def test_onnx_price_forecaster_predicts_expected_horizon(tmp_path: Path) -> None:
    model = ServingPriceModel(
        PriceMLP(input_size=4, hidden_sizes=[8], output_size=2),
        target_mean=10.0,
        target_std=2.0,
    ).eval()
    onnx_path = tmp_path / "price.onnx"
    metadata_path = tmp_path / "price.json"
    sample = torch.randn(1, 4)

    torch.onnx.export(
        model,
        (sample,),
        onnx_path,
        input_names=["history"],
        output_names=["forecast"],
        dynamic_axes={"history": {0: "batch"}, "forecast": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    metadata_path.write_text(
        json.dumps({"model_name": "test_mlp", "context_size": 4, "horizon_size": 2})
    )

    forecaster = OnnxPriceForecaster(onnx_path, metadata_path)
    predictions = forecaster.predict([1.0, 2.0, 3.0, 4.0])

    assert len(predictions) == 2


def test_api_predict_falls_back_when_requested_horizon_differs_from_model() -> None:
    client = TestClient(app)

    response = client.post("/predict", json={"history": [1.0, 2.0, 3.0], "horizon": 2})

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "last_value"
    assert payload["fallback"] is True
    assert payload["predictions"] == [3.0, 3.0]


def test_api_metrics_returns_prediction_counter() -> None:
    client = TestClient(app)

    response = client.get("/metrics")

    assert response.status_code == 200
    payload = response.json()
    assert "request_count" in payload
    assert "model_loaded" in payload
