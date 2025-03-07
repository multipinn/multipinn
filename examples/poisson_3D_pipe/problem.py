import torch

from multipinn.condition import *
from multipinn.geometry import *


def poisson_3D_pipe():
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
            (4 * (cos_pix - 1) - torch.square(torch.tensor(torch.pi)) * cos_pix)
            * two_z_2
            + 2 * cos_pix
            - 2
        )
        denominator = 11 * (2 + z) * two_z_2
        eq1 = u_xx + u_yy + u_zz - (numerator / denominator)
        return [eq1]

    def walls(model, arg):
        x, y, z, u_xx, u_yy, u_zz, u = unpack_symbols(model, arg)
        # assert torch.all(torch.isclose(x, torch.zeros_like(x)))
        return [u - solution(x, y, z)]

    inlet = Hypercube([0.0, 0.0, 0.0], [4.0, 1.0, 1.0])
    middle = Hypercube([4.0, 0.0, 0.0], [5.0, 3.0, 1.0])
    outlet = Hypercube([5.0, 2.0, 0.0], [9.0, 3.0, 1.0])
    domain = inlet | middle | outlet

    my_walls = Shell(domain)

    pde = [
        Condition(inner, domain),
        Condition(walls, my_walls),
    ]
    return pde, input_dim, output_dim
