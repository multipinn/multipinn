import torch
import numpy as np
from multipinn.condition import Condition, Symbols
from multipinn.geometry import *


def sphere():
    symbols = Symbols(variables="x, y, z, t", functions="u", mode="diff")

    def solution(points):
        x, y, z, t = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
        return torch.sin(x) * torch.sin(y) * torch.sin(z) * torch.sin(t)

    inner_symbols = symbols(
        "u, u_x, u_y, u_z, u_t, u_xx, u_yy, u_zz, u_xy, u_xz, u_yz, u_tt")
    basic_symbols = symbols("x, y, z, t")

    def surface(model, arg):
        u, u_x, u_y, u_z, u_t, u_xx, u_yy, u_zz, u_xy, u_xz, u_yz, u_tt = inner_symbols(
            model, arg)
        x, y, z, t = basic_symbols(model, arg)
        eq1 = u_tt - ((1-x**2)*u_xx + (1-y**2)*u_yy + (1-z**2)*u_zz - 2*x*y*u_xy - 2*x*z*u_xz - 2*y*z*u_yz - 2*x*u_x - 2*y*u_y - 2*z*u_z) \
            - ((1-x**2)*torch.sin(x)*torch.sin(y)*torch.sin(z)*(-1) * torch.sin(t) + (1-y**2)*torch.sin(x)*torch.sin(y)*torch.sin(z)*(-1) * torch.sin(t) + (1-z**2)*torch.sin(x)*torch.sin(y)*torch.sin(z)*(-1) * torch.sin(t) - 2*x*y*torch.cos(x)*torch.cos(y)*torch.sin(z) * torch.sin(t) - 2*x*z*torch.cos(x) * torch.sin(y)*torch.cos(z)
               * torch.sin(t) - 2*y*z*torch.sin(x)*torch.cos(y)*torch.cos(z) * torch.sin(t) - 2*x*torch.cos(x)*torch.sin(y) * torch.sin(z) * torch.sin(t) - 2*y*torch.sin(x)*torch.cos(y)*torch.sin(z) * torch.sin(t) - 2*z*torch.sin(x)*torch.sin(y)*torch.cos(z) * torch.sin(t) - torch.sin(x)*torch.sin(y)*torch.cos(z) * torch.sin(t))
        return [eq1]

    def bound(model, arg):
        u = model(arg)
        return [u]

    domain_xyz = Hypersphere(center=[0.0, 0.0, 0.0], radius=1)
    domain_time = Hypercube(low=[0], high=[2 * torch.pi])
    domain = Shell(domain_xyz).product_back(domain_time)
    bound_xyz = Hypersphere(center=[0.0, 0.0, 0.0], radius=1)
    bound_time_0 = Hypercube(low=[0], high=[0])
    bound_0 = Shell(bound_xyz).product_back(bound_time_0)
    bound_xyz = Hypersphere(center=[0.0, 0.0, 0.0], radius=1)
    bound_time_2 = Hypercube(low=[2 * torch.pi], high=[2 * torch.pi])
    bound_2 = Shell(bound_xyz).product_back(bound_time_2)

    pde = [
        Condition(surface, domain),
        Condition(bound, bound_0),
        Condition(bound, bound_2)
    ]
    return pde, symbols.input_dim, symbols.output_dim
