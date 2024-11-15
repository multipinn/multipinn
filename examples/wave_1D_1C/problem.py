import torch

from multipinn.condition import Symbols, unpack
from multipinn.condition.condition import Condition
from multipinn.geometry import *


def exact_solution(arg):
    x, t = unpack(arg)
    solution = (
        torch.sin(torch.pi * x) * torch.cos(2 * torch.pi * t)
        + torch.sin(4 * torch.pi * x) * torch.cos(8 * torch.pi * t) / 2
    )
    return solution.reshape(-1, 1)


def problem_wave1D1C(c):
    symbols = Symbols(variables="x, t", functions="u")

    inner_symbols = symbols("u_xx, u_tt")

    def inner(model, arg):
        u_xx, u_tt = inner_symbols(model, arg)
        eq1 = u_tt - c * c * u_xx
        return [eq1]

    def bc(model, arg):
        arg_x = arg.clone()
        u_x_min = model(arg)[:, 0]
        arg_x[:, 0] += 1.0
        u_x_max = model(arg_x)[:, 0]
        return [u_x_min, u_x_max]

    initial_symbols = symbols("u, x, u_t")

    def ic(model, arg):
        u, x, u_t = initial_symbols(model, arg)
        return [u - torch.sin(torch.pi * x) - torch.sin(4.0 * torch.pi * x) / 2.0, u_t]

    domain = Hypercube(low=[0, 0], high=[1, 1])
    x_min = Hypercube(low=[0, 0], high=[0, 1])
    t_min = Hypercube(low=[0, 0], high=[1, 0])
    pde = [
        Condition(inner, domain),
        Condition(bc, x_min),
        Condition(ic, t_min),
    ]
    return pde, symbols.input_dim, symbols.output_dim
