import torch

from multipinn.condition.condition import Condition
from multipinn.condition.diff import grad, unpack
from multipinn.geometry import *


def conv_diff_2D1C():
    input_dim = 2
    output_dim = 1

    def exact_solution(args):
        return torch.exp(-args[:, 0] + 0.5 * args[:, 1])

    def basic_symbols(model, arg):
        f = model(arg)
        (u,) = unpack(f)
        x, t = unpack(arg)

        return f, u, x, t

    def inner(model, arg):
        f, u, x, t = basic_symbols(model, arg)
        u_x, u_t = unpack(grad(u, arg))
        eq1 = u_t + 0.5 * u_x
        return [eq1]

    def bc1(model, arg):
        f, u, x, t = basic_symbols(model, arg)
        # assert torch.all(torch.isclose(x, torch.zeros_like(x)))
        return [u]

    def bc2(model, arg):
        f, u, x, t = basic_symbols(model, arg)
        # assert torch.all(torch.isclose(x, torch.ones_like(x) * 5))
        return [u]

    def ic(model, arg):
        f, u, x, t = basic_symbols(model, arg)
        # assert torch.all(torch.isclose(t, torch.zeros_like(t)))
        return [u - torch.exp(-x)]

    domain = Hypercube(low=[0, 0], high=[5, 5])
    x_min = Hypercube(low=[0, 0], high=[0, 5])
    x_max = Hypercube(low=[5, 0], high=[5, 5])
    t_0 = Hypercube(low=[0, 0], high=[5, 0])
    pde = [
        Condition(inner, domain),
        Condition(bc1, x_min),
        Condition(bc2, x_max),
        Condition(ic, t_0),
    ]
    return pde, input_dim, output_dim
