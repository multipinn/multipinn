from multipinn.condition.condition import Condition
from multipinn.condition.diff import grad, unpack
from multipinn.geometry import *


def problem_wave2D1C(c):
    input_dim = 2
    output_dim = 1

    def basic_symbols(model, arg):
        f = model(arg)
        (u,) = unpack(f)
        x, t = unpack(arg)
        return f, u, x, t

    def inner(model, arg):
        f, u, x, t = basic_symbols(model, arg)
        u_x, u_t = unpack(grad(u, arg))
        u_xx, u_xt = unpack(grad(u_x, arg))
        u_tx, u_tt = unpack(grad(u_t, arg))
        eq1 = u_tt - c * c * u_xx
        return [eq1]

    def bc1(model, arg):
        f, u, x, t = basic_symbols(model, arg)
        # assert torch.all(torch.isclose(x, torch.zeros_like(x)))
        return [u]

    def bc2(model, arg):
        f, u, x, t = basic_symbols(model, arg)
        # assert torch.all(torch.isclose(x, torch.ones_like(x) * 1))
        return [u]

    def ic(model, arg):
        f, u, x, t = basic_symbols(model, arg)
        # assert torch.all(torch.isclose(t, torch.zeros_like(t)))
        u_x, u_t = unpack(grad(u, arg))
        return [(u - x * (1 - x)) * 10, u_t]

    domain = Hypercube(low=[0, 0], high=[1, 1])
    x_min = Hypercube(low=[0, 0], high=[0, 1])
    x_max = Hypercube(low=[1, 0], high=[1, 1])
    t_min = Hypercube(low=[0, 0], high=[1, 0])
    pde = [
        Condition(inner, domain),
        Condition(bc1, x_min),
        Condition(bc2, x_max),
        Condition(ic, t_min),
    ]
    return pde, input_dim, output_dim
