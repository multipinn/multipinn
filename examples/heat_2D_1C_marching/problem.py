import torch

from multipinn.condition.condition import Condition
from multipinn.condition.diff import grad, unpack
from multipinn.geometry import *


def problem_2D1C_heat_equation(a=1, b=0.5, alpha=0.5, beta=10, gamma=0.7):
    input_dim = 2
    output_dim = 1

    def solution(args):
        return torch.exp(-(gamma**2) * args[:, 1]) * (
            a * torch.cos(alpha * args[:, 0]) + b * torch.sin(beta * args[:, 0])
        )

    def divide(conditions, step, next_step, first_iter, previous_model):
        def ic_new(model, arg):
            _, u, _, _ = basic_symbols(model, arg)
            _, u_prev, _, _ = basic_symbols(previous_model, arg)
            return [u - u_prev]

        new_domain = Hypercube(low=[0, step], high=[2, next_step])
        x_min = Hypercube(low=[0, step], high=[0, next_step])
        x_max = Hypercube(low=[2, step], high=[2, next_step])
        t_min = Hypercube(low=[0, step], high=[2, step])

        conditions[0].geometry = new_domain
        conditions[1].geometry = x_min
        conditions[2].geometry = x_max
        conditions[3].geometry = t_min

        if not first_iter:
            conditions[3].function = ic_new

        conditions[0].points = None

    def basic_symbols(model, arg):
        f = model(arg)
        (u,) = unpack(f)
        x, t = unpack(arg)
        return f, u, x, t

    def inner(model, arg):
        f, u, x, t = basic_symbols(model, arg)
        u_x, u_t = unpack(grad(u, arg))
        u_xx, u_xt = unpack(grad(u_x, arg))
        eq1 = (
            u_t
            - u_xx
            - torch.exp(-(gamma**2) * t)
            * (
                -(gamma**2) * (a * torch.cos(alpha * x) + b * torch.sin(beta * x))
                + b * beta**2 * torch.sin(beta * x)
                + a * alpha**2 * torch.cos(alpha * x)
            )
        )
        return [eq1]

    def ic(model, arg):
        f, u, x, t = basic_symbols(model, arg)
        # assert torch.all(torch.isclose(t, torch.zeros_like(t)))
        return [u - a * torch.cos(alpha * x) - b * torch.sin(beta * x)]

    def bc1(model, arg):
        f, u, x, t = basic_symbols(model, arg)
        # assert torch.all(torch.isclose(x, torch.zeros_like(x)))
        return [u - torch.exp(-(gamma**2) * t) * a]

    def bc2(model, arg):
        f, u, x, t = basic_symbols(model, arg)
        # assert torch.all(torch.isclose(x, torch.ones_like(x) * 2))
        return [
            u
            - torch.exp(-(gamma**2) * t)
            * (
                a * torch.cos(torch.tensor(alpha * 2))
                + b * torch.sin(torch.tensor(beta * 2))
            )
        ]

    domain = Hypercube(low=[0, 0], high=[2, 2])
    x_min = Hypercube(low=[0, 0], high=[0, 2])
    x_max = Hypercube(low=[2, 0], high=[2, 2])
    t_0 = Hypercube(low=[0, 0], high=[2, 0])

    pde = [
        Condition(inner, domain),
        Condition(bc1, x_min),
        Condition(bc2, x_max),
        Condition(ic, t_0),
    ]
    return pde, input_dim, output_dim, divide
