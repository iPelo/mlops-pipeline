from __future__ import annotations

from typing import cast

import torch
from torch import nn


class PriceMLP(nn.Module):
    """Simple multi-layer perceptron for fixed-window price forecasting."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        current_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_size = hidden_size

        layers.append(nn.Linear(current_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(inputs))
