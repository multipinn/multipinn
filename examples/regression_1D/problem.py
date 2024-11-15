import torch

from multipinn.condition.condition import Condition
from multipinn.condition.diff import unpack
from multipinn.geometry import *


def regression_1D_problem():
    input_dim = 1
    output_dim = 1

    def exact_solution(args):  # not exact
        x = args[:, 0]
        return -(x**0.5) * (1 - 0.26 / (x + 0.307))
        # return x ** 0.5 * (1 - 0.26 / (x + 0.307))

    def basic_symbols(model, arg):
        f = model(arg)
        (u,) = unpack(f)
        (x,) = unpack(arg)
        return f, u, x

    def inner(model, arg):
        f, u, x = basic_symbols(model, arg)
        arg_2 = arg + torch.ones_like(arg)
        (u_2,) = unpack(model(arg_2))
        eq1 = u * u_2 - x
        return [eq1]

    domain = Hypercube(low=[0], high=[4])
    pde = [
        Condition(inner, domain),
    ]
    return pde, input_dim, output_dim
