import torch

from multipinn.condition import *
from multipinn.geometry import *


def poisson_3D():
    input_dim = 3
    output_dim = 1

    def solution(x, y, z):
        cos_pix = torch.cos(torch.pi * x)
        numerator = (cos_pix - 1) * torch.exp(2 * y)
        denominator = 11 * (2 + z)
        return numerator / denominator

    def unpack_symbols(model, arg):
        f = model(arg)
        x = arg[:, 0]
        y = arg[:, 1]
        z = arg[:, 2]
        u = f[:, 0]
        g = grad(u, arg)
        u_x = g[:, 0]
        u_y = g[:, 1]
        u_z = g[:, 2]
        u_xx = grad(u_x, arg)[:, 0]
        u_yy = grad(u_y, arg)[:, 1]
        u_zz = grad(u_z, arg)[:, 2]
        return x, y, z, u_xx, u_yy, u_zz, u

    def inner(model, arg):
        x, y, z, u_xx, u_yy, u_zz, u = unpack_symbols(model, arg)
        cos_pix = torch.cos(torch.pi * x)
        two_z_2 = torch.square(2 + z)
        numerator = torch.exp(2 * y) * (
                (4 * (cos_pix - 1) - torch.square(torch.tensor(torch.pi)) * cos_pix) * two_z_2 + 2 * cos_pix - 2)
        denominator = 11 * (2 + z) * two_z_2
        eq1 = u_xx + u_yy + u_zz - (numerator / denominator)
        return [eq1]

    def bc_x0(model, arg):
        x, y, z, u_xx, u_yy, u_zz, u = unpack_symbols(model, arg)
        # assert torch.all(torch.isclose(x, torch.zeros_like(x)))
        return [u]

    def bc_y0(model, arg):
        x, y, z, u_xx, u_yy, u_zz, u = unpack_symbols(model, arg)
        # assert torch.all(torch.isclose(y, torch.zeros_like(y)))
        return [u - solution(x, torch.zeros_like(y), z)]

    def bc_z0(model, arg):
        x, y, z, u_xx, u_yy, u_zz, u = unpack_symbols(model, arg)
        # assert torch.all(torch.isclose(z, torch.zeros_like(z)))
        return [u - solution(x, y, torch.zeros_like(z))]

    def bc_x1(model, arg):
        x, y, z, u_xx, u_yy, u_zz, u = unpack_symbols(model, arg)
        # assert torch.all(torch.isclose(x, torch.ones_like(x)))
        return [u - solution(torch.ones_like(x), y, z)]

    def bc_y1(model, arg):
        x, y, z, u_xx, u_yy, u_zz, u = unpack_symbols(model, arg)
        # assert torch.all(torch.isclose(y, torch.ones_like(y)))
        return [u - solution(x, torch.ones_like(y), z)]

    def bc_z1(model, arg):
        x, y, z, u_xx, u_yy, u_zz, u = unpack_symbols(model, arg)
        # assert torch.all(torch.isclose(z, torch.ones_like(z)))
        return [u - solution(x, y, torch.ones_like(z))]

    domain = Hypercube(low=[-1, -1, -1], high=[1, 1, 1])
    x_0 = Hypercube(low=[-1, -1, -1], high=[-1, 1, 1])
    x_1 = Hypercube(low=[1, -1, -1], high=[1, 1, 1])
    y_0 = Hypercube(low=[-1, -1, -1], high=[1, -1, 1])
    y_1 = Hypercube(low=[-1, 1, -1], high=[1, 1, 1])
    z_0 = Hypercube(low=[-1, -1, -1], high=[1, 1, -1])
    z_1 = Hypercube(low=[-1, -1, 1], high=[1, 1, 1])

    pde = [Condition(inner, domain),
           Condition(bc_x0, x_0),
           Condition(bc_x1, x_1),
           Condition(bc_y0, y_0),
           Condition(bc_y1, y_1),
           Condition(bc_z0, z_0),
           Condition(bc_z1, z_1)
           ]
    return pde, input_dim, output_dim
