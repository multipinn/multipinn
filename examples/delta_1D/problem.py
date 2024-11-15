import torch

from multipinn.condition.condition import Condition
from multipinn.condition.diff import grad, unpack
from multipinn.geometry import *


def delta_problem(sigma=0.01, t_max=1.0):
    input_dim = 2
    output_dim = 1

    x0 = 0.5
    inv_gauss_norm = 1 / (sigma * (2 * torch.pi) ** (1 / 2))
    q = 100 / inv_gauss_norm
    exp_coeff = -1 / (2 * sigma**2)

    def basic_symbols(model, arg):
        f = model(arg)
        (p,) = unpack(f)
        x, t = unpack(arg)
        return f, p, x, t

    def delta_func(x, x0):
        return torch.exp(exp_coeff * (x - x0) ** 2) * inv_gauss_norm

    def inner(model, arg):
        f, p, x, t = basic_symbols(model, arg)

        p_x, p_t = unpack(grad(p, arg))
        p_xx, _ = unpack(grad(p_x, arg))

        eq = p_t - p_xx - q * delta_func(x, x0=x0)

        return [eq]

    def bc1(model, arg):
        f, p, x, t = basic_symbols(model, arg)
        # assert torch.all(torch.isclose(x, torch.zeros_like(x)))
        return [p]

    def bc2(model, arg):
        f, p, x, t = basic_symbols(model, arg)
        # assert torch.all(torch.isclose(x, torch.ones_like(x)))
        return [p]

    def ic(model, arg):
        f, p, x, t = basic_symbols(model, arg)
        # assert torch.all(torch.isclose(t, torch.zeros_like(t)))
        return [p]

    domain = Hypercube(low=[0, 0], high=[1, t_max])
    x_min = Hypercube(low=[0, 0], high=[0, t_max])
    x_max = Hypercube(low=[1, 0], high=[1, t_max])
    t_min = Hypercube(low=[0, 0], high=[1, 0])
    pde = [
        Condition(inner, domain),
        Condition(bc1, x_min),
        Condition(bc2, x_max),
        Condition(ic, t_min),
    ]
    return pde, input_dim, output_dim
