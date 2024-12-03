import torch

from multipinn.condition import *
from multipinn.geometry import *


def laplace2D_1C(focus=1):
    input_dim = 2
    output_dim = 1

    def basic_symbols(model, arg):
        f = model(arg)
        (u,) = unpack(f)
        x, y = unpack(arg)
        return f, u, x, y

    def inner(model, arg):
        f, u, x, y = basic_symbols(model, arg)
        u_x, u_y = unpack(grad(u, arg))
        u_xx, u_xy = unpack(grad(u_x, arg))
        u_yx, u_yy = unpack(grad(u_y, arg))
        # lap_u, = unpack(num_laplace(model, f, arg))
        eq0 = u_xx + u_yy
        return [eq0]

    def bc(model, arg, data):
        f, u, x, y = basic_symbols(model, arg)
        normal = data[0]
        # u_x, u_y = unpack(grad(u, arg))
        # normal_x, normal_y = unpack(normal)
        # u_n = u_x * normal_x + u_y * normal_y
        (u_n,) = unpack(num_diff_random(model, arg, f, normal))
        return [u_n]

    def cold(model, arg):
        f, u, x, y = basic_symbols(model, arg)
        return [-1 - u]

    def hot(model, arg):
        f, u, x, y = basic_symbols(model, arg)
        return [1 - u]

    domain = Hypersphere([0, 0], 5)
    source1 = Hypersphere([-focus, 0], 0.1)
    source2 = Hypersphere([focus, 0], 0.1)

    pde = [
        Condition(inner, domain - (source1 | source2)),
        ConditionExtra(bc, Shell(domain), ["normals"]),
        Condition(cold, source1),
        Condition(hot, source2),
    ]
    return pde, input_dim, output_dim
