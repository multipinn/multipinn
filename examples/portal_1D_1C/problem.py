import torch

from multipinn.condition.condition import Condition
from multipinn.condition.diff import grad, unpack
from multipinn.geometry import *


def portal_1D_1C_problem():
    input_dim = 1
    output_dim = 1

    def solution(x):
        return torch.sin(x)

    def basic_symbols(model, arg):
        f = model(arg)
        (u,) = unpack(f)
        (x,) = unpack(arg)
        return u, x

    def inner(model, arg):
        u, x = basic_symbols(model, arg)
        (u_x,) = unpack(grad(u, arg))
        (u_xx,) = unpack(grad(u_x, arg))
        eq1 = u + u_xx
        return [eq1]

    def middle(model, arg):
        u, x = basic_symbols(model, arg)
        # assert torch.all(torch.isclose(x, torch.pi * torch.ones_like(x)))
        return [u - 1]

    def portal_01(model, arg):
        ul, xl = basic_symbols(model, arg)
        # assert torch.all(torch.isclose(xl, torch.zeros_like(xl)))
        delta = 2 * torch.pi
        (ur,) = unpack(model(arg + delta))  # can do this in 1D
        (ul_x,) = unpack(grad(ul, arg))
        (ur_x,) = unpack(grad(ur, arg))
        return [ur - ul, ur_x - ul_x]

    domain = Hypercube(low=[0], high=[2 * torch.pi])
    x_middle = Hypercube(low=[torch.pi], high=[torch.pi])
    x_min = Hypercube(low=[0], high=[0])

    pde = [
        Condition(inner, domain),
        Condition(middle, x_middle),
        Condition(portal_01, x_min),
    ]
    return pde, input_dim, output_dim
