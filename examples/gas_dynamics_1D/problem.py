import torch

from multipinn.condition.condition import Condition
from multipinn.condition.diff import grad, unpack
from multipinn.geometry import *


def gas_dynamics_1d(domain_size=5.0, rho0=1.0, D=1.0, p0=0.0):
    input_dim = 1
    output_dim = 3

    def pde(model, arg):
        f = model(arg)
        rho, v, p = unpack(f)
        (x,) = unpack(arg)

        rho_x = unpack(grad(rho, arg))[0]
        v_x = unpack(grad(v, arg))[0]
        p_x = unpack(grad(p, arg))[0]

        mass = rho_x * v + rho * v_x

        momentum = rho_x * v**2 + 2 * rho * v * v_x + p_x

        return [mass, momentum]

    def inlet_condition(model, arg):
        f = model(arg)
        rho, v, p = unpack(f)

        return [rho - rho0, v - D, p - p0]

    def rho_func(x, p2=0.5, p3=1):
        exp_min = p2 * torch.exp(torch.tensor([0])).item()
        p_true = p2 * torch.exp(-x / p3)
        p_true[p_true > exp_min] = 0
        return 1 / (1 - p_true)

    def data_rho(model, arg):
        f = model(arg)
        rho, _, _ = unpack(f)
        (x,) = unpack(arg)

        return [rho_func(x) - rho]

    domain = Hypercube(low=[-1.0], high=[domain_size])
    inlet = Hypercube(low=[-1.0], high=[-1.0])

    conditions = [
        Condition(pde, domain),
        Condition(inlet_condition, inlet),
        Condition(data_rho, domain),
    ]

    return conditions, input_dim, output_dim
