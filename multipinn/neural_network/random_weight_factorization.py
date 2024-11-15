from typing import List

import torch
import torch.nn as nn

from .activation_function import GELU


class FactorizedDense(nn.Module):
    """Implementation of the Random Weight Factorization layer from
    "Random Weight Factorization Improves the Training of Continuous Neural Representations" paper.
    https://arxiv.org/abs/2210.01274.
    """

    __constants__ = ["in_features", "out_features"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mean: float = 1.0,
        stddev: float = 0.1,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        w = torch.empty((out_features, in_features), **factory_kwargs)
        s = torch.empty(in_features, **factory_kwargs)
        torch.nn.init.xavier_normal_(w)
        torch.nn.init.normal_(s, mean=mean, std=stddev)
        self.s = nn.Parameter(torch.exp(s))
        self.v = nn.Parameter(w / self.s)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            torch.nn.init.uniform_(self.bias, -stddev, stddev)
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        kernel = self.s * self.v
        return torch.nn.functional.linear(input, kernel, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class FactorizedFNN(nn.Module):
    """
    Factorized Feedforward Neural Network
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_layers: List[int]):
        """
        Args:
            input_dim (int): Input dimension.
            output_dim (int): Output dimension.
            hidden_layers (List[int]): List containing the number of neurons for each hidden layer.
        """
        super(FactorizedFNN, self).__init__()

        layers = [nn.Sequential(FactorizedDense(input_dim, hidden_layers[0]), GELU())]

        for in_features, out_features in zip(hidden_layers[:-1], hidden_layers[1:]):
            layers.append(
                nn.Sequential(FactorizedDense(in_features, out_features), GELU())
            )

        layers.append(FactorizedDense(hidden_layers[-1], output_dim))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x (torch.Tensor): Input tensor (batch size, input dimension).

        Returns:
            torch.Tensor: Output tensor (batch size, output dimension).
        """
        for layer in self.layers:
            x = layer(x)
        return x
