import torch
from torch import nn


@torch.jit.script
def fused_gelu(x):
    """
    Custom GELU activation function
    Args:
        x (torch.Tensor): Input tensor
    Returns:
        torch.Tensor: Output after applying GELU activation
    """
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))


@torch.jit.script
def fused_sin(x):
    """
    Custom sine activation function
    Args:
        x (torch.Tensor): Input tensor
    Returns:
        torch.Tensor: Output after applying sine activation
    """
    return torch.sin(x)


class GELU(nn.Module):
    """
    GELU activation function module

    Applies the Gaussian Error Linear Unit (GELU) function element-wise.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        return fused_gelu(input)


class Sine(nn.Module):
    """
    Sine activation function module

    Applies the sine function element-wise.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        return fused_sin(input)
