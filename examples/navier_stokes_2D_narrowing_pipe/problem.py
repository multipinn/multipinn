import torch

from multipinn.condition import *
from multipinn.geometry import *


def Hypersphere_2points(horizontal, other):
    x1, y1 = horizontal
    x2, y2 = other
    r = ((x1 - x2) ** 2 + (y1 - y2) ** 2) * 0.5 / (y2 - y1)
    return Hypersphere((x1, y1 + r), abs(r))


def nozzle_pipe(
    x_left, y_left_bottom, y_left_top, x_right, y_right_bottom, y_right_top
):
    assert y_left_bottom <= y_left_top
    assert y_right_bottom <= y_right_top
    assert x_left < x_right
    assert abs(y_left_top - y_right_top) <= x_right - x_left
    assert abs(y_left_bottom - y_right_bottom) <= x_right - x_left
    x_mid = (x_left + x_right) * 0.5
    y_mid_top = (y_left_top + y_right_top) * 0.5
    y_mid_bottom = (y_left_bottom + y_right_bottom) * 0.5
    domain = Hypercube([x_left, y_mid_bottom], [x_right, y_mid_top])
    if y_left_top < y_right_top:
        domain -= Hypersphere_2points([x_left, y_left_top], [x_mid, y_mid_top])
        domain |= Hypercube(
            [x_mid, y_mid_top], [x_right, y_right_top]
        ) & Hypersphere_2points([x_right, y_right_top], [x_mid, y_mid_top])
    elif y_left_top > y_right_top:
        domain -= Hypersphere_2points([x_right, y_right_top], [x_mid, y_mid_top])
        domain |= Hypercube(
            [x_left, y_mid_top], [x_mid, y_left_top]
        ) & Hypersphere_2points([x_left, y_left_top], [x_mid, y_mid_top])
    if y_left_bottom < y_right_bottom:
        domain -= Hypersphere_2points([x_right, y_right_bottom], [x_mid, y_mid_bottom])
        domain |= Hypercube(
            [x_left, y_left_bottom], [x_mid, y_mid_bottom]
        ) & Hypersphere_2points([x_left, y_left_bottom], [x_mid, y_mid_bottom])
    elif y_left_bottom > y_right_bottom:
        domain -= Hypersphere_2points([x_left, y_left_bottom], [x_mid, y_mid_bottom])
        domain |= Hypercube(
            [x_mid, y_right_bottom], [x_right, y_mid_bottom]
        ) & Hypersphere_2points([x_right, y_right_bottom], [x_mid, y_mid_bottom])
    return domain


def navier_stokes_2D_narrowing_pipe(re=50):
    input_dim = 2
    output_dim = 3
    i_re = 1 / re

    def basic_symbols(model, arg):
        f = model(arg)
        u, v, p = unpack(f)
        x, y = unpack(arg)
        return f, u, v, p, x, y

    def inner(model, arg):
        f, u, v, p, x, y = basic_symbols(model, arg)

        u_x, u_y = unpack(grad(u, arg))
        v_x, v_y = unpack(grad(v, arg))
        p_x, p_y = unpack(grad(p, arg))
        # u_x, v_x, p_x, = unpack(num_diff_random(model, f, arg, torch.tensor([[1, 0]])))
        # u_y, v_y, p_y, = unpack(num_diff_random(model, f, arg, torch.tensor([[1, 0]])))

        u_xx, _ = unpack(grad(u_x, arg))
        v_xx, _ = unpack(grad(v_x, arg))
        _, u_yy = unpack(grad(u_y, arg))
        _, v_yy = unpack(grad(v_y, arg))
        laplace_u = u_xx + u_yy
        laplace_v = v_xx + v_yy
        # laplace_u, laplace_v, laplace_p, = unpack(num_laplace(model, f, arg))

        eq1 = u * u_x + v * u_y + p_x - i_re * laplace_u
        eq2 = u * v_x + v * v_y + p_y - i_re * laplace_v
        eq3 = u_x + v_y

        return [eq1, eq2, eq3]

    def bc_x_min(model, arg):
        f, u, v, p, x, y = basic_symbols(model, arg)
        return [u - (-0.16 * y**2 + 1), v, p]

    # def bc_x_max(model, arg):
    #     f, u, v, p, x, y = basic_symbols(model, arg)
    #     return [p]

    def block(model, arg):
        f, u, v, p, x, y = basic_symbols(model, arg)
        return [u, v]

    nozzle1_start_x = 2
    nozzle1_end_x = 5
    cross_y = 0.5
    nozzle2_start_x = 5
    nozzle2_end_x = 8

    nozzle1 = nozzle_pipe(nozzle1_start_x, -1, 1, nozzle1_end_x, -cross_y, cross_y)
    nozzle2 = nozzle_pipe(nozzle2_start_x, -cross_y, cross_y, nozzle2_end_x, -1, 1)
    if nozzle1_end_x != nozzle2_start_x:
        straight_middle = Hypercube(
            [nozzle1_end_x, -cross_y], [nozzle2_start_x, cross_y]
        )
        nozzle_combined = nozzle1 | straight_middle | nozzle2
    else:
        nozzle_combined = nozzle1 | nozzle2
    straight_in = Hypercube([0, -1], [nozzle1_start_x, 1])
    straight_out = Hypercube([nozzle2_end_x, -1], [10, 1])
    domain = straight_in | nozzle_combined | straight_out

    x_min = Hypercube([0.0, -1.0], [0.0, 1.0])
    x_max = Hypercube([10.0, -1.0], [10.0, 1.0])

    shell = Shell(domain)
    walls = shell - (x_min | x_max)

    pde = [
        Condition(inner, domain),
        Condition(bc_x_min, x_min),
        # Condition(bc_x_max, x_max),
        Condition(block, walls),
    ]

    return pde, input_dim, output_dim
