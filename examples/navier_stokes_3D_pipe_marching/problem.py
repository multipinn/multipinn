import torch

from multipinn.condition import *
from multipinn.geometry import *
from multipinn.geometry import *

import numpy as np

def navier_stocks_3d_new_geom_problem_dist(re=60):
    input_dim = 3
    output_dim = 4
    iRe = 1 / re

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
        # (
        #     u_x,
        #     v_x,
        #     w_x,
        #     p_x,
        # ) = unpack(num_diff_random(model, arg, f, torch.tensor([[1, 0, 0]])))
        # (
        #     u_y,
        #     v_y,
        #     w_y,
        #     p_y,
        # ) = unpack(num_diff_random(model, arg, f, torch.tensor([[0, 1, 0]])))
        # (
        #     u_z,
        #     v_z,
        #     w_z,
        #     p_z,
        # ) = unpack(num_diff_random(model, arg, f, torch.tensor([[0, 0, 1]])))

        # (
        #     laplace_u,
        #     laplace_v,
        #     laplace_w,
        #     laplace_p,
        # ) = unpack(num_laplace(model, arg, f))

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

        temp_p =  torch.tensor(np.zeros_like(p.cpu().detach().numpy()), dtype=p.dtype)

        return [(u - u_f), v, w, temp_p]

    def output_cond(model, arg):
        f, u, v, w, p, x, y, z = basic_symbols(model, arg)
        tmp = torch.tensor(np.zeros_like(p.cpu().detach().numpy()), dtype=p.dtype)
        return [tmp, tmp, tmp, p]

    def walls(model, arg):
        f, u, v, w, p, x, y, z = basic_symbols(model, arg)
        return [u, v, w]
    
    def divide(conditions, step, next_step, first_iter, previous_model):
        def input_cond_new(model, arg):
            f, u, v, w, p, x, y, z = basic_symbols(model, arg)
            prev_f, prev_u, prev_v, prev_w, prev_p, prev_x, prev_y, prev_z = basic_symbols(previous_model, arg)
            return [(u - prev_u), (v - prev_v), (w - prev_w), (p - prev_p)]

        if step < 4.0:
            new_domain = Hypercube([step, 0.0, 0.0], [next_step, 1.0, 1.0])
            new_inp = Hypercube([step, 0.0, 0.0], [step, 1.0, 1.0])
            new_out = Hypercube([next_step, 0.0, 0.0], [next_step, 1.0, 1.0])
        else:
            if step < 5.0:
                new_domain = Hypercube([step, 0.0, 0.0], [next_step, 3.0, 1.0])
                new_inp = Hypercube([step, 0.0, 0.0], [step, 1.0, 1.0])
                new_out = Hypercube([next_step, 2.0, 0.0], [next_step, 3.0, 1.0])
            else:
                new_domain = Hypercube([step, 2.0, 0.0], [next_step, 3.0, 1.0])
                new_inp = Hypercube([step, 2.0, 0.0], [step, 3.0, 1.0])
                new_out = Hypercube([next_step, 2.0, 0.0], [next_step, 3.0, 1.0])
        
        new_shell = Shell(new_domain)
        new_wall = new_shell - (new_inp | new_out)
        new_inp_wall = new_shell & new_inp
        new_out_wall = new_shell & new_out

        conditions[0].geometry = new_domain
        conditions[1].geometry = new_inp_wall
        conditions[2].geometry = new_out_wall
        conditions[3].geometry = new_wall

        if not first_iter:
            conditions[1].function = input_cond_new
        
        if next_step == 9.0:
            conditions[2].function = output_cond
        
        conditions[0].points = None

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
        Condition(inner, output_wall),
        Condition(walls, wall),
    ]

    return pde, input_dim, output_dim, divide