from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
import torch
from torch import nn

from energy_price_mlops.models.mlp import PriceMLP
from energy_price_mlops.training.lightning_module import PriceForecastModule


class ServingPriceModel(nn.Module):
    """Wrap a normalized training model behind original-scale inputs and outputs."""

    def __init__(
        self,
        model: PriceMLP,
        *,
        target_mean: float,
        target_std: float,
        feature_means: list[float] | None = None,
        feature_stds: list[float] | None = None,
        context_size: int = 1,
    ) -> None:
        super().__init__()
        self.model = model
        self.target_mean = target_mean
        self.target_std = target_std
        feature_means = feature_means or [target_mean]
        feature_stds = feature_stds or [target_std]
        flattened_means = torch.tensor(feature_means * context_size, dtype=torch.float32)
        flattened_stds = torch.tensor(feature_stds * context_size, dtype=torch.float32)
        self._feature_means: torch.Tensor
        self._feature_stds: torch.Tensor
        self.register_buffer("_feature_means", flattened_means)
        self.register_buffer("_feature_stds", flattened_stds)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized_features = (features - self._feature_means) / self._feature_stds
        normalized_forecast = self.model(normalized_features)
        return cast(torch.Tensor, normalized_forecast * self.target_std + self.target_mean)


def export_from_metrics(
    *,
    metrics_path: str | Path,
    output_path: str | Path,
    sample_seed: int = 42,
) -> dict[str, Any]:
    metrics = json.loads(Path(metrics_path).read_text())
    config = metrics["config"]
    checkpoint_path = Path(metrics["best_checkpoint_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint from metrics file does not exist: {checkpoint_path}")

    module = _load_module_from_checkpoint(metrics)
    serving_model = ServingPriceModel(
        module.model,
        target_mean=module.target_mean,
        target_std=module.target_std,
        feature_means=metrics.get("feature_means", module.feature_means),
        feature_stds=metrics.get("feature_stds", module.feature_stds),
        context_size=int(config["data"]["forecasting"]["context_hours"]),
    ).eval()

    input_size = int(metrics.get("model_input_size", config["model"]["input_size"]))
    output_size = int(config["model"]["output_size"])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator().manual_seed(sample_seed)
    sample = (
        torch.randn(2, input_size, generator=generator) * module.target_std + module.target_mean
    )

    torch.onnx.export(
        serving_model,
        (sample,),
        output,
        input_names=["history"],
        output_names=["forecast"],
        dynamic_axes={"history": {0: "batch"}, "forecast": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    max_abs_error = compare_torch_and_onnx(serving_model, output, sample)
    feature_columns = metrics.get("feature_columns", config["data"].get("feature_columns", []))
    metadata = {
        "model_name": config["model"]["name"],
        "input_size": input_size,
        "context_size": int(config["data"]["forecasting"]["context_hours"]),
        "horizon_size": output_size,
        "feature_columns": feature_columns,
        "num_features": len(feature_columns),
        "target_col": config["target"],
        "checkpoint_path": str(checkpoint_path),
        "target_mean": module.target_mean,
        "target_std": module.target_std,
        "onnx_max_abs_error": max_abs_error,
        "test_metrics": metrics.get("test", {}),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def compare_torch_and_onnx(
    model: nn.Module,
    onnx_path: str | Path,
    sample: torch.Tensor,
) -> float:
    with torch.no_grad():
        torch_output = model(sample).detach().cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    model_inputs = {"history": sample.detach().cpu().numpy().astype(np.float32)}
    onnx_output = session.run(None, model_inputs)[0]
    return float(np.max(np.abs(torch_output - onnx_output)))


def _load_module_from_checkpoint(metrics: dict[str, Any]) -> PriceForecastModule:
    config = metrics["config"]
    model = PriceMLP(
        input_size=int(metrics.get("model_input_size", config["model"]["input_size"])),
        hidden_sizes=[int(size) for size in config["model"]["hidden_sizes"]],
        dropout=float(config["model"]["dropout"]),
        output_size=int(config["model"]["output_size"]),
    )
    return PriceForecastModule.load_from_checkpoint(
        metrics["best_checkpoint_path"],
        model=model,
        map_location="cpu",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained price forecaster to ONNX.")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("artifacts/training/mlp_default/metrics.json"),
        help="Path to the training metrics JSON containing the best checkpoint path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/models/price_mlp.onnx"),
        help="Output ONNX path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = export_from_metrics(metrics_path=args.metrics, output_path=args.output)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
