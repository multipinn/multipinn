from typing import List, Literal

import torch
import torch.nn as nn

from .activation_function import GELU, Sine


class FNN(nn.Module):
    """
    Feedforward Neural Network implementation.

    Args:
        input_dim (int): Dimension of input data.
        output_dim (int): Dimension of output data.
        hidden_layers (List[int]): List containing the number of neurons for each hidden layer.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_layers: List[int]):
        """
        Args:
            input_dim (int): Dimension of input data.
            output_dim (int): Dimension of output data.
            hidden_layers (List[int]): List containing the number of neurons for each hidden layer.
        """
        super(FNN, self).__init__()

        network_layers = [nn.Sequential(nn.Linear(input_dim, hidden_layers[0]), Sine())]

        for in_features, out_features in zip(hidden_layers[:-1], hidden_layers[1:]):
            network_layers.append(
                nn.Sequential(nn.Linear(in_features, out_features), GELU())
            )

        network_layers.append(nn.Linear(hidden_layers[-1], output_dim))

        self.network_layers = nn.ModuleList(network_layers)

    def forward(self, input_tensor: torch.Tensor):
        """
        Forward pass through the network.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """

        for layer in self.network_layers:
            input_tensor = layer(input_tensor)
        return input_tensor


class XavierFNN(FNN):
    """
    Feedforward Neural Network with Xavier initialization.

    Args:
        input_dim (int): Dimension of input data.
        output_dim (int): Dimension of output data.
        hidden_layers (List[int]): List containing the number of neurons for each hidden layer.
        init_mode (Literal["norm", "uniform"]): Initialization mode for Xavier initialization.
            'norm' uses normal distribution, 'uniform' uses uniform distribution.
            Defaults to "norm".
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int],
        init_mode: Literal["norm", "uniform"] = "norm",
    ):
        super().__init__(input_dim, output_dim, hidden_layers)

        if init_mode == "norm":
            weight_init_func = nn.init.xavier_normal_
        elif init_mode == "uniform":
            weight_init_func = nn.init.xavier_uniform_

        for layer in self.network_layers[:-1]:
            weight_init_func(layer[0].weight)
            nn.init.zeros_(layer[0].bias)

        weight_init_func(self.network_layers[-1].weight)
        nn.init.zeros_(self.network_layers[-1].bias)
