import torch

from multipinn.condition.condition import Condition
from multipinn.condition.diff import grad, unpack
from multipinn.geometry import *


def poisson_2D_1C():
    input_dim = 2
    output_dim = 1

    def exact_solution(args):
        return (
            torch.cos(2 * torch.pi * args[:, 0]) * torch.sin(4 * torch.pi * args[:, 1])
            + 0.5 * args[:, 0]
        )

    def basic_symbols(model, arg):
        f = model(arg)
        (u,) = unpack(f)
        x, y = unpack(arg)
        return f, u, x, y

    def inner(model, arg):
        f, u, x, y = basic_symbols(model, arg)
        u_x, u_y = unpack(grad(u, arg))
        u_xx, u_xy = unpack(grad(u_x, arg))
        u_yx, u_yy = unpack(grad(u_y, arg))
        eq1 = (
            u_xx
            + u_yy
            + 20
            * torch.pi**2
            * torch.cos(2 * torch.pi * x)
            * torch.sin(4 * torch.pi * y)
        )
        return [eq1]

    def bc1(model, arg):
        f, u, x, y = basic_symbols(model, arg)
        # assert torch.all(torch.isclose(x, torch.zeros_like(x)))
        return [u - torch.sin(4 * torch.pi * y)]

    def bc2(model, arg):
        f, u, x, y = basic_symbols(model, arg)
        # assert torch.all(torch.isclose(x, torch.ones_like(x) * 0.5))
        return [u + torch.sin(4 * torch.pi * y) - 0.25]

    def bc3(model, arg):
        f, u, x, y = basic_symbols(model, arg)
        # assert torch.all(torch.isclose(y, torch.zeros_like(y)))
        return [u - 0.5 * x]

    def bc4(model, arg):
        f, u, x, y = basic_symbols(model, arg)
        # assert torch.all(torch.isclose(y, torch.ones_like(y) * 0.5))
        return [u - 0.5 * x]

    domain = Hypercube(low=[0, 0], high=[0.5, 0.5])
    x_min = Hypercube(low=[0, 0], high=[0, 0.5])
    x_max = Hypercube(low=[0.5, 0], high=[0.5, 0.5])
    y_min = Hypercube(low=[0, 0], high=[0.5, 0])
    y_max = Hypercube(low=[0, 0.5], high=[0.5, 0.5])
    pde = [
        Condition(inner, domain),
        Condition(bc1, x_min),
        Condition(bc2, x_max),
        Condition(bc3, y_min),
        Condition(bc4, y_max),
    ]
    return pde, input_dim, output_dim
