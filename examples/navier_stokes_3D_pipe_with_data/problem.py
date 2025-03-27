import torch

from multipinn.condition import *
from multipinn.geometry import *
import meshio
import numpy as np


def navier_stokes_3D_pipe_with_data(re=60):
    input_dim = 3
    output_dim = 4
    iRe = 1 / re

    data = meshio.read(f'./examples/navier_stokes_3D_pipe_with_data/data.vtk')

    def basic_symbols(model, arg):
        f = model(arg)
        u, v, w, p = unpack(f)
        x, y, z = unpack(arg)
        return f, u, v, w, p, x, y, z

    def inner(model, arg):
        f, u, v, w, p, x, y, z = basic_symbols(model, arg) 

        u_x, u_y, u_z = unpack(grad(u, arg))
        v_x, v_y, v_z = unpack(grad(v, arg))
        w_x, w_y, w_z = unpack(grad(w, arg))
        p_x, p_y, p_z = unpack(grad(p, arg))

        u_xx, _, _ = unpack(grad(u_x, arg))
        v_xx, _, _ = unpack(grad(v_x, arg))
        w_xx, _, _ = unpack(grad(w_x, arg))

        _, u_yy, _ = unpack(grad(u_y, arg))
        _, v_yy, _ = unpack(grad(v_y, arg))
        _, w_yy, _ = unpack(grad(w_y, arg))

        _, _, u_zz = unpack(grad(u_z, arg))
        _, _, v_zz = unpack(grad(v_z, arg))
        _, _, w_zz = unpack(grad(w_z, arg))

        laplace_u = u_xx + u_yy + u_zz
        laplace_v = v_xx + v_yy + v_zz
        laplace_w = w_xx + w_yy + w_zz

        eq1 = u * u_x + v * u_y + w * u_z + p_x - iRe * laplace_u
        eq2 = u * v_x + v * v_y + w * v_z + p_y - iRe * laplace_v
        eq3 = u * w_x + v * w_y + w * w_z + p_z - iRe * laplace_w
        eq4 = u_x + v_y + w_z

        return [eq1, eq2, eq3, eq4]

    def input_cond(model, arg):
        f, u, v, w, p, x, y, z = basic_symbols(model, arg)

        inlet_speed = 2.0

        u_f = inlet_speed / (0.25 * 0.25) * (0.25 - (z - 0.5) ** 2) * (0.25 - (y - 0.5) ** 2) / 0.89031168
        u_f = torch.nn.functional.relu(u_f)

        return [(u - u_f), v, w]

    def output_cond(model, arg):
        f, u, v, w, p, x, y, z = basic_symbols(model, arg)

        return [p]

    def walls(model, arg):
        f, u, v, w, p, x, y, z = basic_symbols(model, arg)
        return [u, v, w]
    
    def data_cond(model, arg):
        f = model(torch.tensor(np.array(data.points, dtype=float), dtype=torch.float32))
        u, v, w, p = unpack(f)
        return [u - torch.tensor(np.array(data.point_data['U'][:, 0], dtype=float), dtype=torch.float32), v - torch.tensor(np.array(data.point_data['U'][:, 1], dtype=float), dtype=torch.float32), w - torch.tensor(np.array(data.point_data['U'][:, 2], dtype=float), dtype=torch.float32), p - torch.tensor(np.array(data.point_data['p'], dtype=float), dtype=torch.float32)]

    inlet = Hypercube([0.0, 0.0, 0.0], [4.0, 1.0, 1.0])
    middle = Hypercube([4.0, 0.0, 0.0], [5.0, 3.0, 1.0])
    outlet = Hypercube([5.0, 2.0, 0.0], [9.0, 3.0, 1.0])
    domain = inlet | middle | outlet

    inp = Hypercube([0.0, 0.0, 0.0], [0.0, 1.0, 1.0])
    output = Hypercube([9.0, 2.0, 0.0], [9.0, 3.0, 1.0])

    shell = Shell(domain)
    wall = shell - (inp | output)
    input_wall = shell & inp
    output_wall = shell & output

    pde = [
        Condition(inner, domain),
        Condition(input_cond, input_wall),
        Condition(output_cond, output_wall),
        Condition(walls, wall),
        Condition(data_cond, wall)
    ]

    return pde, input_dim, output_dim
