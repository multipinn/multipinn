import torch

from multipinn.condition import Condition, Symbols
from multipinn.geometry import *


def heat_2D_simple():
    symbols = Symbols(variables="x, t", functions="u")

    def exact_solution(args):
        return torch.exp(0.5 * args[:, 0]) + 0.5 * args[:, 1]

    inner_symbols = symbols("x, u_t, u_xx")

    def inner(model, arg):
        x, u_t, u_xx = inner_symbols(model, arg)
        eq1 = u_t - u_xx - (0.5 - 0.25 * torch.exp(0.5 * x))
        return [eq1]

    basic_symbols = symbols("x, t, u")

    def ic(model, arg):
        x, t, u = basic_symbols(model, arg)
        return [torch.exp(0.5 * x) - u]

    def bc1(model, arg):
        x, t, u = basic_symbols(model, arg)
        return [1 + 0.5 * t - u]

    def bc2(model, arg):
        x, t, u = basic_symbols(model, arg)
        return [torch.exp(torch.tensor(1)) + 0.5 * t - u]

    domain = Hypercube(low=[0, 0], high=[2, 1])
    x_min = Hypercube(low=[0, 0], high=[0, 1])
    x_max = Hypercube(low=[2, 0], high=[2, 1])
    t_0 = Hypercube(low=[0, 0], high=[2, 0])
    pde = [
        Condition(inner, domain),
        Condition(bc1, x_min),
        Condition(bc2, x_max),
        Condition(ic, t_0),
    ]
    return pde, symbols.input_dim, symbols.output_dim
