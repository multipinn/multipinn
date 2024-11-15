from __future__ import annotations

import torch
import torch.nn as nn

from .activation_function import GELU, Sine
from .light_residual_block import LightResidualBlock


class ResNet(nn.Module):
    """
    Residual Neural Network (ResNet) class.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: list,
        blocks: list = None,
        res_block: nn.Module = LightResidualBlock,
    ):
        """
        Args:
            input_dim (int): Input dimension.
            output_dim (int): Output dimension.
            hidden_layers (List[int]): List containing the number of neurons for each hidden layer.
            blocks (list): List containing the number of ResidualBlocks for each neuron count.
            res_block (nn.Module): ResidualBlock module to be used in the model.
        """

        super(ResNet, self).__init__()

        if blocks is None:
            blocks = []

        layers_list = [
            nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=hidden_layers[0]), Sine()
            )
        ]

        if len(blocks) == 0:
            for layer, next_layer in zip(hidden_layers[:-1], hidden_layers[1:]):
                layers_list.append(
                    self.__make_layers(res_block, 1, layer, next_layer, GELU())
                )
            layers_list.append(
                self.__make_layers(
                    res_block, 1, hidden_layers[-1], hidden_layers[-1], GELU(), False
                )
            )
        else:
            for layer, block, next_layer in zip(
                hidden_layers[:-1], blocks[:-1], hidden_layers[1:]
            ):
                layers_list.append(
                    self.__make_layers(res_block, block, layer, next_layer, GELU())
                )
            layers_list.append(
                self.__make_layers(
                    res_block,
                    blocks[-1],
                    hidden_layers[-1],
                    hidden_layers[-1],
                    GELU(),
                    False,
                )
            )

        layers_list.append(
            nn.Linear(in_features=hidden_layers[-1], out_features=output_dim)
        )

        self.layers = nn.ModuleList(layers_list)

    def __make_layers(
        self,
        res_block: nn.Module,
        count_blocks: int,
        in_features: int,
        out_features: int,
        activation: nn.Module,
        is_not_last: bool = True,
    ):
        """
        Function:
            Creates ResidualBlocks of the same size and a subsequent dimension-changing layer.
        Args:
            count_blocks (int): Number of ResidualBlocks for a specific element in the layers list.
            in_features (int): Input dimension.
            out_features (int): Output dimension.
            activation (nn.Module): Activation function. For custom activation functions,
                it is sufficient to implement a wrapper class inheriting from nn.Module
                and implementing the forward function.
            res_block (nn.Module): ResidualBlock module to be used in the model.
            is_not_last (bool): Flag indicating whether there is no next ResidualBlock.
        """

        layers = []

        for i in range(count_blocks - 1):
            layers.append(res_block(activation, in_features))

        layers.append(res_block(activation, in_features))

        if is_not_last:
            layers.append(
                nn.Sequential(nn.Linear(in_features, out_features), activation)
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """
        Forward pass

        Args:
            x (torch.Tensor): Input tensor (batch size, x dimension).

        Returns:
            torch.Tensor: Output tensor (batch size, 1).
        """
        for layer in self.layers:
            x = layer(x)
        return x
