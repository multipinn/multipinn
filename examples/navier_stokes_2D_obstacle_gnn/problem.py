from multipinn.condition import *
from multipinn.geometry import *


def navier_stokes_derivatives(arg, output):
    data = {}

    x = arg.nodes

    data["u"], data["v"], data["p"] = unpack(output)

    data["u_x"], data["u_y"] = unpack(grad(data["u"], x))
    data["v_x"], data["v_y"] = unpack(grad(data["v"], x))
    data["p_x"], data["p_y"] = unpack(grad(data["p"], x))

    data["u_xx"], _ = unpack(grad(data["u_x"], x))
    data["v_xx"], _ = unpack(grad(data["v_x"], x))

    _, data["u_yy"] = unpack(grad(data["u_y"], x))
    _, data["v_yy"] = unpack(grad(data["v_y"], x))

    return data


def navier_stokes_equation_with_obstacle(re=50):
    input_dim = 2
    output_dim = 3
    i_re = 1 / re

    def basic_symbols(arg, model):
        data = model(arg, buff=True)
        x, y = unpack(arg.nodes)
        return data, x, y

    def inner(arg, model):
        data, x, y = basic_symbols(arg, model)

        laplace_u = data["u_xx"] + data["u_yy"]
        laplace_v = data["v_xx"] + data["v_yy"]

        eq1 = (
            data["u"] * data["u_x"]
            + data["v"] * data["u_y"]
            + data["p_x"]
            - i_re * laplace_u
        )
        eq2 = (
            data["u"] * data["v_x"]
            + data["v"] * data["v_y"]
            + data["p_y"]
            - i_re * laplace_v
        )
        eq3 = data["u_x"] + data["v_y"]
        return [eq1, eq2, eq3]

    def inlet_cond(arg, model):
        data, x, y = basic_symbols(arg, model)
        return [data["u"] - (-0.16 * y**2 + 1), data["v"]]

    def outlet_cond(arg, model):
        data, x, y = basic_symbols(arg, model)
        return [data["u_x"], data["p"]]

    def wall_cond(arg, model):
        data, x, y = basic_symbols(arg, model)
        return [data["u"], data["v"]]

    rect_back = Hypercube([-6, -3], [9, 3])

    obstacle = Hypercube([-1, -0.5], [1, 0.5])
    domain = rect_back - obstacle

    x_min = Hypercube(low=[-6.0, -3.0], high=[-6.0, 3.0])
    x_max = Hypercube(low=[9.0, -3.0], high=[9.0, 3.0])

    shell = Shell(domain)
    walls = shell - (x_min | x_max)

    pde = [
        Condition(inner, domain),
        Condition(inlet_cond, x_min),
        Condition(outlet_cond, x_max),
        Condition(wall_cond, walls),
    ]

    return pde, input_dim, output_dim
