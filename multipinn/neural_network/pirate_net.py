import torch
import torch.nn as nn

from .activation_function import GELU
from .fourier_features import FourierEncoding


class PIModifiedBottleneck(nn.Module):
    """
    Modified bottleneck with parametric interpolation (PI).

    Arguments:
        hidden_size: the size of the hidden layers.
        output_dim: the size of the output vector.
        nonlinearity: the parameter for mixing identity with new features.
    """

    def __init__(self, hidden_size, output_dim, nonlinearity) -> None:
        super().__init__()

        self.layer_1 = nn.Linear(hidden_size, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        self.layer_3 = nn.Linear(hidden_size, output_dim)

        self.act = GELU()

        # Alpha controls the degree of interpolation between original and new features
        self.alpha = nn.Parameter(torch.tensor([nonlinearity], dtype=torch.float32))

    def forward(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
        identity = x

        x = self.layer_1(x)
        x = self.act(x)

        # Interpolation between two data streams u and v
        x = x * u + (1 - x) * v

        x = self.layer_2(x)
        x = self.act(x)

        x = x * u + (1 - x) * v

        x = self.layer_3(x)
        x = self.act(x)

        # Mixing the original and new features with the alpha parameter
        x = self.alpha * x + (1 - self.alpha) * identity

        return x


class PirateNet(nn.Module):
    """
    Main PirateNet model using modified PIModifiedBottleneck blocks.

    Implementation of PirateNets: Physics-informed Deep Learning with Residual Adaptive Networks
    Sifan Wang, Bowen Li, Yuhan Chen, Paris Perdikaris

    https://arxiv.org/abs/2402.00326

    Arguments:
        input_dim: number of input features.
        output_dim: number of output features.
        emb_size: embedding size (default is 256).
        hidden_size: size of the hidden layers.
        num_blocks: number of modified bottleneck layers.
        nonlinearity: coefficient for interpolation nonlinearity.
        gamma: coefficient for Fourier encoding.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 256,
        num_blocks: int = 3,
        nonlinearity: float = 0.0,
        gamma: float = 1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.gamma = gamma

        self.encoding = FourierEncoding(input_dim, hidden_size, gamma)
        self.u = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)

        self.blocks = nn.ModuleList(
            [
                PIModifiedBottleneck(hidden_size, hidden_size, nonlinearity)
                for _ in range(num_blocks)
            ]
        )

        self.out = nn.Linear(hidden_size, output_dim)

        self.act = GELU()

    def forward(self, x: torch.Tensor):
        x = self.encoding(x)

        u = self.act(self.u(x))
        v = self.act(self.v(x))

        for b in self.blocks:
            x = b(x, u, v)

        x = self.out(x)

        return x
