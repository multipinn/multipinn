import torch

from multipinn.condition import Symbols
from multipinn.condition.condition import Condition
from multipinn.condition.diff import grad, unpack
from multipinn.geometry import *


def problem_burgers1D():
    symbols = Symbols(variables="x, t", functions="u")
    v = 0.01 / torch.pi

    inner_symbols = symbols("u, u_x, u_t, u_xx")

    def inner(model, arg):
        u, u_x, u_t, u_xx = inner_symbols(model, arg)
        eq1 = u_t + u * u_x - v * u_xx
        return [eq1]

    def bc(model, arg):
        arg_x = arg.clone()
        u_x_min = model(arg)[:, 0]
        arg_x[:, 0] += 2.0
        u_x_max = model(arg_x)[:, 0]
        return [u_x_min, u_x_max]

    initial_symbols = symbols("u, x")

    def ic(model, arg):
        u, x = initial_symbols(model, arg)
        return [u + torch.sin(torch.pi * x)]

    domain = Hypercube(low=[-1, 0], high=[1, 1])
    x_min = Hypercube(low=[-1, 0], high=[-1, 1])
    t_min = Hypercube(low=[-1, 0], high=[1, 0])
    pde = [
        Condition(inner, domain),
        Condition(bc, x_min),
        Condition(ic, t_min),
    ]
    return pde, symbols.input_dim, symbols.output_dim
