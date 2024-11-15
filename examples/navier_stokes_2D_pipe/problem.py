from multipinn.condition import *
from multipinn.geometry import *


def navier_stokes_2D_pipe(re=100):
    input_dim = 2
    output_dim = 3
    iRe = 1 / re

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

        u_xx, _ = unpack(grad(u_x, arg))
        v_xx, _ = unpack(grad(v_x, arg))

        _, u_yy = unpack(grad(u_y, arg))
        _, v_yy = unpack(grad(v_y, arg))

        laplace_u = u_xx + u_yy
        laplace_v = v_xx + v_yy

        eq1 = u * u_x + v * u_y + p_x - iRe * laplace_u
        eq2 = u * v_x + v * v_y + p_y - iRe * laplace_v
        eq3 = u_x + v_y

        return [eq1, eq2, eq3]

    def inlet_cond(model, arg):
        f, u, v, p, x, y = basic_symbols(model, arg)
        inlet_sp = 1
        u_f = inlet_sp * (1 - 4 * (y - 0.5) ** 2)
        return [u - u_f, v]

    def outlet_cond(model, arg):
        f, u, v, p, x, y = basic_symbols(model, arg)
        u_x, u_y = unpack(grad(u, arg))
        return [u_x, p]

    def wall_cond(model, arg):
        f, u, v, p, x, y = basic_symbols(model, arg)
        return [u, v]

    inlet = Hypercube(low=[0.0, 0.0], high=[5.0, 1.0])
    middle = Hypercube(low=[4.0, 1.0], high=[5.0, 2.0])
    outlet = Hypercube(low=[4.0, 2.0], high=[9.0, 3.0])
    domain = inlet | middle | outlet

    # Round pipe, no square parts
    # domain = Hypercube([0, 0], [4, 1]) | Hypercube([4, 1], [5, 2]) | Hypercube([5, 2], [9, 3])\
    #          | Hypercube([3.5, 0], [5, 1.5]) - Hypersphere([3.5, 1.5], 0.5) & Hypersphere([4, 1], 1)\
    #          | Hypercube([4, 1.5], [5.5, 3]) - Hypersphere([5.5, 1.5], 0.5) & Hypersphere([5, 2], 1)

    input = Hypercube(low=[0.0, 0.0], high=[0.0, 1.0])
    output = Hypercube(low=[9.0, 2.0], high=[9.0, 3.0])
    shell = Shell(domain)
    walls = shell - (input | output)
    input_wall = shell & input
    output_wall = shell & output

    pde = [
        Condition(inner, domain),
        Condition(inlet_cond, input_wall),
        Condition(outlet_cond, output_wall),
        Condition(wall_cond, walls),
    ]

    return pde, input_dim, output_dim
