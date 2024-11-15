from multipinn.condition import *
from multipinn.geometry import *


def problem(re=100, a=4):
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
        # u_y, v_y, p_y, = unpack(num_diff_random(model, f, arg, torch.tensor([[0, 1]])))

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

    def bc_g_1(model, arg):
        f, u, v, p, x, y = basic_symbols(model, arg)
        return [u - a * x * (1 - x), v]

    def bc_g_2(model, arg):
        f, u, v, p, x, y = basic_symbols(model, arg)
        return [u, v]

    def bc_p(model, arg):
        f, u, v, p, x, y = basic_symbols(model, arg)
        return [p]

    domain = Hypercube([0, 0], [1, 1])

    g_1 = Hypercube(low=[0, 1], high=[1, 1])

    shell = Shell(domain)

    g_2 = shell - (g_1)

    p_point = Hypercube([0, 0], [0, 0])

    pde = [
        Condition(inner, domain),
        Condition(bc_g_1, g_1),
        Condition(bc_g_2, g_2),
        Condition(bc_p, p_point),
    ]

    return pde, input_dim, output_dim
