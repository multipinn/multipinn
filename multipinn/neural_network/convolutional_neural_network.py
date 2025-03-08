from __future__ import annotations

from typing import List, Literal

import torch
import torch.nn as nn

from .activation_function import GELU, Sine


class CNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: List[int]):
        super(CNN, self).__init__()
        network_layers = [nn.Sequential(nn.LazyConv1d(hidden_layers[0], 1, 1), Sine())]

        for output_features in hidden_layers[1:]:
            network_layers.append(
                nn.Sequential(nn.LazyConv1d(output_features, 1, 1), GELU())
            )

        network_layers.append(nn.LazyConv1d(output_dim, 1, 1))

        self.layers = nn.ModuleList(network_layers)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(0).permute(0, 2, 1)
        for layer in self.layers:
            x = layer(x)
        return x.permute(0, 2, 1).squeeze(0)
