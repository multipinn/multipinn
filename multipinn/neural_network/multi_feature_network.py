from typing import Callable, List

import torch
import torch.nn as nn

from .activation_function import GELU


class Shift(nn.Module):
    def __init__(self, shift):
        super().__init__()
        self.shift = shift

    def forward(self, x):
        return x - self.shift


class SquareShift(nn.Module):
    def __init__(self, shift):
        super().__init__()
        self.shift = shift

    def forward(self, x):
        return (x - self.shift) ** 2


class SigmoidShift(nn.Module):
    def __init__(self, shift):
        super().__init__()
        self.shift = shift

    def forward(self, x):
        return torch.sigmoid(x - self.shift)


class ScaledTrig(nn.Module):
    def __init__(self, trig_func: Callable, scale: float):
        super().__init__()
        self.trig_func = trig_func
        self.scale = scale

    def forward(self, x):
        return self.trig_func(torch.pi * x * self.scale)


class MultiFeatureEncoding(nn.Module):
    def __init__(self, functions: nn.ModuleList = None):
        super().__init__()

        self.functions = (
            functions
            if functions
            else nn.ModuleList(
                [
                    nn.Identity(),
                    SquareShift(0),
                    SquareShift(1.5),
                    SquareShift(4.5),
                    SigmoidShift(0),
                    SigmoidShift(1.5),
                    SigmoidShift(4.5),
                    Shift(1.5),
                    Shift(4.5),
                    ScaledTrig(torch.sin, 1 / torch.pi),
                    ScaledTrig(torch.cos, 1 / torch.pi),
                    ScaledTrig(torch.sin, 1),
                    ScaledTrig(torch.cos, 1),
                    ScaledTrig(torch.sin, 1 / 4),
                    ScaledTrig(torch.cos, 1 / 4),
                    ScaledTrig(torch.sin, 1 / 2),
                    ScaledTrig(torch.cos, 1 / 2),
                ]
            )
        )

    @property
    def num_features(self):
        return len(self.functions)

    def forward(self, x):
        results = [f(x) for f in self.functions]
        return torch.hstack(results)


class MultiFeatureNetwork(nn.Module):
    """
    Multi-layer neural network with multiple features for solving problems of various dimensions.

    This network uses MultiFeatureEncoding for preprocessing input data
    and applies a sequence of linear layers with different activation functions.

    Args:
        input_dim (int): Size of the input layer.
        output_dim (int): Size of the output layer.
        hidden_layers (List[int]): List of sizes for hidden layers.
        use_jit (bool): Whether to use JIT compilation for the encoding layer.

    Attributes:
        layers (nn.ModuleList): List of neural network layers, including encoding and linear transformations.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int],
        use_jit: bool = False,
    ):
        super().__init__()

        if use_jit:
            encoding = torch.jit.script(MultiFeatureEncoding())
            dummy_input = torch.zeros((1, input_dim))
            num_features = encoding(dummy_input).shape[1] // input_dim
        else:
            encoding = MultiFeatureEncoding()
            num_features = encoding.num_features

        layers = [
            encoding,
            nn.Sequential(
                nn.Linear(input_dim * num_features, hidden_layers[0]), GELU()
            ),
        ]

        for layer, next_layer in zip(hidden_layers[:-1], hidden_layers[1:]):
            layers.append(nn.Sequential(nn.Linear(layer, next_layer), GELU()))

        layers.append(nn.Linear(hidden_layers[-1], output_dim))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor after passing through all network layers.
        """
        for layer in self.layers:
            x = layer(x)
        return x
