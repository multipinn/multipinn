import torch
import torch.nn as nn


class LightResidualBlock(nn.Module):
    """
    Residual Block class.
    """

    def __init__(self, activation, features: int):
        """
        Args:
            features (int): Dimension of input and output data.
            activation (nn.Module): Activation function. For custom activation functions,
                it is sufficient to implement a wrapper class inheriting from nn.Module
                and implementing the forward function.
        """
        super(LightResidualBlock, self).__init__()
        self.linear_first = nn.Sequential(nn.Linear(features, features), activation)
        self.linear_second = nn.Linear(features, features)
        self.activation = activation

    def forward(self, x: torch.Tensor):
        """
        Forward pass

        Args:
            x (torch.Tensor): Input tensor (batch size, in_features dimension).

        Returns:
            torch.Tensor: Output tensor (batch size, out_features dimension).
        """
        residual = x
        out = self.linear_first(x)
        out = self.linear_second(out)
        out += residual
        out = self.activation(out)
        return out
