from typing import List

import torch

from multipinn.condition import Condition, Symbols, ConditionExtra
from multipinn.geometry import *


def helmholtz_3D():
    lambd = 0.5
    k = 2*torch.pi/lambd

    symbols = Symbols(variables="x, y, z", functions="u", mode="diff")

    inner_symbols = symbols("u, u_xx, u_yy, u_zz, u_x, u_y, u_z")
    basic_symbols = symbols("x, y, z")

    def inner(model, arg):
        u, u_xx, u_yy, u_zz, u_x, u_y, u_z = inner_symbols(model, arg)
        x, y, z = basic_symbols(model, arg)
        eq1 = u_xx + u_yy + u_zz + u * k**2 + 2 * \
            torch.cos(k*x) * torch.cos(k*y) * torch.cos(k*z) * k**2
        return [eq1]

    def bc_n(model, arg, data):
        normal = data[0]
        u, u_xx, u_yy, u_zz, u_x, u_y, u_z = inner_symbols(model, arg)
        return [normal[:, 0] * u_x + normal[:, 1] * u_y + normal[:, 2] * u_z]

    def bc_d(model, arg):
        u, u_xx, u_yy, u_zz, u_x, u_y, u_z = inner_symbols(model, arg)
        return [u]

    domain = Hypercube(low=[0.0, 0.0, 0.0], high=[1.0, 1.0, 1.0])

    pde = [
        Condition(inner, domain),
        ConditionExtra(bc_n, Shell(domain), ["normals"]),
        Condition(bc_d, Shell(domain))
    ]
    return pde, symbols.input_dim, symbols.output_dim
