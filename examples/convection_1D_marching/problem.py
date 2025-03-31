import torch

from multipinn.condition import Condition, Symbols
from multipinn.condition.diff import unpack
from multipinn.geometry import *


def convection_problem(betta=1.0, t_max=1.0, x_max=torch.pi * 2):
    symbols = Symbols(variables="t, x", functions="p")

    inner_symbols = symbols("p_t, p_x")

    def inner(model, arg):
        p_t, p_x = inner_symbols(model, arg)
        eq = p_t + betta * p_x
        return [eq]

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

    def basic_symbols(model, arg):
        f = model(arg)
        (u,) = unpack(f)
        x, t = unpack(arg)
        return f, u, x, t

    def divide(conditions, step, next_step, first_iter, previous_model):
        def ic_new(model, arg):
            _, u, _, _ = basic_symbols(model, arg)
            _, u_prev, _, _ = basic_symbols(previous_model, arg)
            return [u - u_prev]

        new_domain = Hypercube(low=[step, 0], high=[next_step, torch.pi * 2])
        x_min = Hypercube(low=[step, 0], high=[next_step, 0])
        t_min = Hypercube(low=[step, 0], high=[step, torch.pi * 2])

        conditions[0].geometry = new_domain
        conditions[1].geometry = x_min
        conditions[2].geometry = t_min

        if not first_iter:
            conditions[2].function = ic_new

        conditions[0].points = None

    domain = Hypercube(low=[0, 0], high=[t_max, x_max])
    x_portal = Hypercube(low=[0, 0], high=[t_max, 0])
    t_min = Hypercube(low=[0, 0], high=[0, x_max])
    pde = [
        Condition(inner, domain),
        Condition(portal, x_portal),
        Condition(ic, t_min),
    ]
    return pde, symbols.input_dim, symbols.output_dim, divide
