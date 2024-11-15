import torch

from multipinn.condition import *
from multipinn.geometry import *


def gray_scott_2D(t_max=200):
    input_dim = 3
    output_dim = 2

    B = 0.04
    D = 0.1
    EPS_1 = 1e-5
    EPS_2 = 5e-6

    def basic_symbols(model, arg):
        f = model(arg)
        (u, v) = unpack(f)
        x, y, t = unpack(arg)
        return f, u, v, x, y, t

    def inner(model, arg):
        f, u, v, x, y, t = basic_symbols(model, arg)
        u_x, u_y, u_t = unpack(grad(u, arg))
        v_x, v_y, v_t = unpack(grad(v, arg))

        u_xx, _, _ = unpack(grad(u_x, arg))
        v_xx, _, _ = unpack(grad(v_x, arg))
        _, u_yy, _ = unpack(grad(u_y, arg))
        _, v_yy, _ = unpack(grad(v_y, arg))
        laplace_u = u_xx + u_yy
        laplace_v = v_xx + v_yy

        eq1 = EPS_1 * laplace_u + B * (1 - u) - u * v**2
        eq2 = EPS_2 * laplace_v - D * v + u * v**2
        return [u_t - eq1, v_t - eq2]

    def ic(model, arg):
        f, u, v, x, y, t = basic_symbols(model, arg)
        u_cond = 1 - torch.exp(-80 * ((x + 0.05) ** 2 + (y + 0.02) ** 2))
        v_cond = torch.exp(-80 * ((x - 0.05) ** 2 + (y - 0.02) ** 2))
        return [u - u_cond, v - v_cond]

    domain = Hypercube(low=[-1, -1, 0], high=[1, 1, t_max])
    t_0 = Hypercube(low=[-1, -1, 0], high=[1, 1, 0])

    pde = [
        Condition(inner, domain),
        Condition(ic, t_0),
    ]
    return pde, input_dim, output_dim
