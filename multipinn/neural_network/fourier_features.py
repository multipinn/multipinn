from typing import List, Literal

import torch
import torch.nn as nn

from .feedforward_neural_network import FNN, XavierFNN


class FourierEncoding(nn.Module):
    """
    Fourier encoding layer from "On the eigenvector bias of Fourier
    feature networks: From regression to solving multi-scale PDEs
    with physics-informed neural networks" paper.
    https://arxiv.org/pdf/2012.10047.pdf
    """

    def __init__(
        self,
        input_dim: int,
        encoding_dim: int,
        sigma: float,
        is_trainable: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.sigma = sigma
        self.is_trainable = is_trainable

        w = torch.zeros((input_dim, encoding_dim // 2))
        torch.nn.init.normal_(w)
        self.B = nn.Parameter(w * self.sigma, requires_grad=is_trainable)

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, encoding_dim={self.encoding_dim}, sigma={self.sigma}"

    def forward(self, x: torch.Tensor):
        s = torch.sin(x @ self.B)
        c = torch.cos(x @ self.B)
        return torch.cat([s, c], dim=1)


class FourierFeatureNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int],
        encoding_dim: int,
        sigma: float = 1.0,
        is_trainable: bool = False,
        xavier_init: bool = False,
        xavier_init_mode: Literal["norm", "uniform"] = "norm",
        use_jit: bool = False,
    ) -> None:
        super().__init__()

        self.fourier_encoding = FourierEncoding(
            input_dim=input_dim,
            encoding_dim=encoding_dim,
            sigma=sigma,
            is_trainable=is_trainable,
        )
        if use_jit:
            self.fourier_encoding = torch.jit.script(self.fourier_encoding)

        if xavier_init:
            self.fnn = XavierFNN(
                encoding_dim, output_dim, hidden_layers, init_mode=xavier_init_mode
            )
        else:
            self.fnn = FNN(encoding_dim, output_dim, hidden_layers)

    def forward(self, x: torch.Tensor):
        x = self.fourier_encoding(x)
        return self.fnn(x)


class MultiScaleFFNN(nn.Module):
    """
    MultiScaleFFNN from "On the eigenvector bias of Fourier
    feature networks: From regression to solving multi-scale PDEs
    with physics-informed neural networks" paper.
    Usage:
    model = MultiScaleFFNN(input_dim, output_dim, hidden_layers, encoding_dim, sigmas)
    https://arxiv.org/pdf/2012.10047.pdf
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int],
        encoding_dim: int,
        sigmas: List[float],
        xavier_init: bool = False,
        xavier_init_mode: Literal["norm", "uniform"] = "norm",
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleList(
            [FourierEncoding(input_dim, encoding_dim, s) for s in sigmas]
        )

        if xavier_init:
            self.fnn = XavierFNN(
                encoding_dim,
                hidden_layers[-1],
                hidden_layers[:-1],
                init_mode=xavier_init_mode,
            )
        else:
            self.fnn = FNN(encoding_dim, hidden_layers[-1], hidden_layers[:-1])

        self.linear = nn.Linear(len(sigmas) * hidden_layers[-1], output_dim)

    def forward(self, x: torch.Tensor):
        x_encoded = [e(x) for e in self.encoders]
        x_decoded = [self.fnn(x_e) for x_e in x_encoded]
        x_cat = torch.cat(x_decoded, dim=1)
        return self.linear(x_cat)
