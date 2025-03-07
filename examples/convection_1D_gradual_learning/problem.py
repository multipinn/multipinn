import torch

from multipinn.condition import Condition, Symbols
from multipinn.geometry import *


def convection_problem(betta=1.0, t_max=1.0, x_max=torch.pi * 2):
    symbols = Symbols(variables="t, x", functions="p")

    inner_symbols = symbols("p_t, p_x")

    def inner(model, arg):
        p_t, p_x = inner_symbols(model, arg)
        eq = p_t + betta * p_x
        return [eq]

    def betta_setter(new_betta):
        nonlocal betta
        betta = new_betta
        return betta

    portal_symbols = symbols("p")

    def portal(model, arg):
        (p,) = portal_symbols(model, arg)
        arg_x = arg.clone()
        arg_x[:, 1] += x_max
        (p_x_max,) = portal_symbols(model, arg_x)
        return [p - p_x_max]

    ic_symbols = symbols("p, x")

    def ic(model, arg):
        p, x = ic_symbols(model, arg)
        return [p - torch.sin(x)]

    domain = Hypercube(low=[0, 0], high=[t_max, x_max])
    x_portal = Hypercube(low=[0, 0], high=[t_max, 0])
    t_min = Hypercube(low=[0, 0], high=[0, x_max])
    pde = [
        Condition(inner, domain),
        Condition(portal, x_portal),
        Condition(ic, t_min),
    ]
    return pde, symbols.input_dim, symbols.output_dim, betta_setter
