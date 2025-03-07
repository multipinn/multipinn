import torch

from multipinn.condition.condition import Condition
from multipinn.condition.diff import grad, unpack
from multipinn.geometry import *


def poisson_3D():
    input_dim = 3
    output_dim = 1

    def solution(x, y, z):
        cos_pix = torch.cos(torch.pi * x)
        numerator = (cos_pix - 1) * torch.exp(2 * y)
        denominator = 11 * (2 + z)
        return numerator / denominator

    def basic_symbols(model, arg):
        f = model(arg)
        (u,) = unpack(f)
        x, y, z = unpack(arg)
        return f, u, x, y, z

    # def unpack_grads(model, arg):
    #     f, u, x, y, z = basic_symbols(model, arg)
    #     u_x, u_y, u_z = unpack(grad(u, arg))

    #     u_xx, _, _ = unpack(grad(u_x, arg))
    #     _, u_yy, _ = unpack(grad(u_y, arg))
    #     _, _, u_zz = unpack(grad(u_z, arg))
    #     return u_xx, u_yy, u_zz

    def inner(model, arg):
        f, u, x, y, z = basic_symbols(model, arg)
        u_x, u_y, u_z = unpack(grad(u, arg))

        u_xx, _, _ = unpack(grad(u_x, arg))
        _, u_yy, _ = unpack(grad(u_y, arg))
        _, _, u_zz = unpack(grad(u_z, arg))

        eg1 = (
            u_xx
            + u_yy
            + u_zz
            - torch.exp(2 * y)
            / (22 + 11 * z)
            * (
                -torch.pi**2 * torch.cos(torch.pi * x)
                + 4 * (torch.cos(torch.pi * x) - 1)
                + (2 * torch.cos(torch.pi * x) - 1) / (2 + z) ** 2
            )
        )
        return [eg1]

    def bc_x0(model, arg):
        f, u, x, y, z = basic_symbols(model, arg)
        return [u - solution(torch.zeros_like(x), y, z)]

    def bc_y0(model, arg):
        f, u, x, y, z = basic_symbols(model, arg)
        return [u - solution(x, torch.zeros_like(y), z)]

    def bc_x2(model, arg):
        f, u, x, y, z = basic_symbols(model, arg)
        return [u - solution(2 * torch.ones_like(x), y, z)]

    def bc_y2(model, arg):
        f, u, x, y, z = basic_symbols(model, arg)
        return [u - solution(x, 2 * torch.ones_like(y), z)]

    def bc_x3(model, arg):
        f, u, x, y, z = basic_symbols(model, arg)
        return [u - solution(3 * torch.ones_like(x), y, z)]

    def bc_y3(model, arg):
        f, u, x, y, z = basic_symbols(model, arg)
        return [u - solution(x, 3 * torch.ones_like(y), z)]

    def bc_x1(model, arg):
        f, u, x, y, z = basic_symbols(model, arg)
        return [u - solution(1 * torch.ones_like(x), y, z)]

    def bc_y1(model, arg):
        f, u, x, y, z = basic_symbols(model, arg)
        return [u - solution(x, 1 * torch.ones_like(y), z)]

    def bc_z0(model, arg):
        f, u, x, y, z = basic_symbols(model, arg)
        return [u - solution(x, y, torch.zeros_like(z))]

    def bc_z1(model, arg):
        f, u, x, y, z = basic_symbols(model, arg)
        return [u - solution(x, y, torch.ones_like(z))]

    inlet = Hypercube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    middle = Hypercube([1.0, 0.0, 0.0], [2.0, 3.0, 1.0])
    outlet = Hypercube([2.0, 2.0, 0.0], [3.0, 3.0, 1.0])
    domain = inlet | middle | outlet

    x_0 = Hypercube([0.0, 0.0, 0.0], [0.0, 1.0, 1.0])
    y_0 = Hypercube([0.0, 0.0, 0.0], [2.0, 0.0, 1.0])
    x_2 = Hypercube([2.0, 0.0, 0.0], [2.0, 2.0, 1.0])
    y_2 = Hypercube([2.0, 2.0, 0.0], [2.0, 3.0, 1.0])
    x_3 = Hypercube([3.0, 2.0, 0.0], [3.0, 3.0, 1.0])
    y_3 = Hypercube([1.0, 3.0, 0.0], [3.0, 3.0, 1.0])
    x_1 = Hypercube([1.0, 1.0, 0.0], [1.0, 3.0, 1.0])
    y_1 = Hypercube([0.0, 1.0, 0.0], [1.0, 1.0, 1.0])
    z_0 = (
        Hypercube([0.0, 0.0, 0.0], [1.0, 1.0, 0.0])
        - Hypercube([0.0, 1.0, 0.0], [1.0, 3.0, 0.0])
        - Hypercube([2.0, 0.0, 0.0], [3.0, 2.0, 0.0])
    )
    z_1 = (
        Hypercube([0.0, 0.0, 1.0], [1.0, 1.0, 1.0])
        - Hypercube([0.0, 1.0, 1.0], [1.0, 3.0, 1.0])
        - Hypercube([2.0, 0.0, 1.0], [3.0, 2.0, 1.0])
    )

    pde = [
        Condition(inner, domain),
        Condition(bc_x0, x_0),
        Condition(bc_y0, y_0),
        Condition(bc_x2, x_2),
        Condition(bc_y2, y_2),
        Condition(bc_x3, x_3),
        Condition(bc_y3, y_3),
        Condition(bc_x1, x_1),
        Condition(bc_y1, y_1),
        Condition(bc_z0, z_0),
        Condition(bc_z1, z_1),
    ]
    return pde, input_dim, output_dim
