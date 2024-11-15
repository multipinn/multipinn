from typing import List

import torch
import torch.nn as nn

from .activation_function import GELU, Sine


class DenseNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        neurons_per_block: List[int],
        layers_per_block: List[int] = None,
    ):
        """Dense Neural Network with skip connections within blocks.

        Args:
            input_dim (int): Dimension of input features
            output_dim (int): Dimension of output features
            neurons_per_block (List[int]): Number of neurons for each dense block.
                The length of this list determines the number of dense blocks.
            layers_per_block (List[int], optional): Number of layers in each dense block.
                If None, defaults to 3 layers per block. Must match length of neurons_per_block.
        """
        super().__init__()

        if layers_per_block is None:
            layers_per_block = [3 for _ in neurons_per_block]

        layers_list = [
            nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=neurons_per_block[0]),
                Sine(),
            )
        ]

        for i in range(len(neurons_per_block)):
            if i != len(neurons_per_block) - 1:
                layers_list.append(
                    nn.Sequential(
                        DenseBlock(layers_per_block[i], neurons_per_block[i]),
                        nn.Linear(neurons_per_block[i], neurons_per_block[i + 1]),
                        GELU(),
                    )
                )
            else:
                layers_list.append(
                    DenseBlock(layers_per_block[i], neurons_per_block[i])
                )

        layers_list.append(nn.Linear(neurons_per_block[-1], output_dim))
        self.layers = nn.ModuleList(layers_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        for layer in self.layers:
            x = layer(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, layers_amount: int, features: int):
        """Dense block with skip connections.

        Each layer in the block has the same number of features and includes
        skip connections (residual connections) from input to output.
        The first activation is Sine, followed by GELU for subsequent layers.

        Args:
            layers_amount (int): Number of layers in this dense block
            features (int): Number of features (neurons) for each layer
        """
        super().__init__()
        layers = [nn.Linear(features, features) for _ in range(layers_amount - 1)]
        layers.insert(0, nn.Linear(features, features))
        activations = [nn.GELU() for _ in range(layers_amount - 1)]
        activations.insert(0, Sine())
        self.layers = nn.ModuleList(layers)
        self.activations = nn.ModuleList(activations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the dense block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, features)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, features)
                         with added skip connections
        """
        for layer, activation in zip(self.layers, self.activations):
            out = layer(x)
            out = activation(out)
            x = x + out
        return x
