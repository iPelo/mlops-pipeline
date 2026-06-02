from pathlib import Path

import torch

from energy_price_mlops.export_onnx import ServingPriceModel, compare_torch_and_onnx
from energy_price_mlops.models.mlp import PriceMLP


def test_serving_onnx_export_matches_torch(tmp_path: Path) -> None:
    model = ServingPriceModel(
        PriceMLP(input_size=4, hidden_sizes=[8], output_size=2),
        target_mean=10.0,
        target_std=2.0,
    ).eval()
    sample = torch.randn(3, 4)
    onnx_path = tmp_path / "model.onnx"

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

    max_abs_error = compare_torch_and_onnx(model, onnx_path, sample)

    assert max_abs_error < 1e-4
