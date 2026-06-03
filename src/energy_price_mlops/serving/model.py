from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import cast

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]


class OnnxPriceForecaster:
    """Small ONNX Runtime wrapper for serving the exported price model."""

    def __init__(self, model_path: str | Path, metadata_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model does not exist: {self.model_path}")

        self.metadata_path = (
            Path(metadata_path)
            if metadata_path is not None
            else self.model_path.with_suffix(".json")
        )
        self.metadata = self._load_metadata(self.metadata_path)
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    @property
    def model_name(self) -> str:
        return str(self.metadata.get("model_name", "onnx_price_mlp"))

    @property
    def context_size(self) -> int:
        value = self.metadata.get("context_size", 168)
        return int(cast(int | str, value))

    @property
    def input_size(self) -> int:
        value = self.metadata.get("input_size", self.context_size)
        return int(cast(int | str, value))

    @property
    def num_features(self) -> int:
        value = self.metadata.get("num_features", 1)
        return int(cast(int | str, value))

    @property
    def horizon_size(self) -> int:
        value = self.metadata.get("horizon_size", 24)
        return int(cast(int | str, value))

    def predict(self, features: list[float]) -> list[float]:
        if len(features) < self.input_size:
            raise ValueError(f"features must contain at least {self.input_size} values.")
        recent_features = np.asarray(features[-self.input_size :], dtype=np.float32).reshape(1, -1)
        output = self.session.run([self.output_name], {self.input_name: recent_features})[0]
        return [float(value) for value in output.reshape(-1)]

    @staticmethod
    def _load_metadata(path: Path) -> dict[str, object]:
        if not path.exists():
            return {}
        return cast(dict[str, object], json.loads(path.read_text()))


class PredictionStats:
    def __init__(self) -> None:
        self.request_count = 0
        self.last_latency_ms: float | None = None

    def observe(self, start_time: float) -> None:
        self.request_count += 1
        self.last_latency_ms = (perf_counter() - start_time) * 1000

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "request_count": self.request_count,
            "last_latency_ms": self.last_latency_ms,
        }
