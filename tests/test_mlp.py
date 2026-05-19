import torch
from torch import nn

from energy_price_mlops.models.mlp import PriceMLP


def test_forward_pass_returns_expected_output_shape() -> None:
    model = PriceMLP(input_size=168, hidden_sizes=[64, 32], output_size=24)

    output = model(torch.randn(8, 168))

    assert output.shape == (8, 24)


def test_gradient_flows_to_every_parameter() -> None:
    model = PriceMLP(input_size=168, hidden_sizes=[64, 32], output_size=24)

    model(torch.randn(4, 168)).sum().backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"missing gradient for {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient for {name}"


def test_parameter_count_matches_layer_sizes() -> None:
    # Linear(10, 8): 10*8 + 8 = 88 ; Linear(8, 4): 8*4 + 4 = 36 ; total = 124.
    model = PriceMLP(input_size=10, hidden_sizes=[8], output_size=4)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    assert total_parameters == 124


def test_dropout_layer_added_only_when_requested() -> None:
    without_dropout = PriceMLP(input_size=10, hidden_sizes=[8], output_size=4)
    with_dropout = PriceMLP(input_size=10, hidden_sizes=[8], output_size=4, dropout=0.5)

    assert not any(isinstance(module, nn.Dropout) for module in without_dropout.modules())
    assert any(isinstance(module, nn.Dropout) for module in with_dropout.modules())
