import torch

from multipinn.condition import *
from multipinn.geometry import *


def beltrami_flow(re=60):
    input_dim = 3
    output_dim = 4
    iRe = 1 / re

    def solution(x, y, z):
        return [
            -(torch.exp(x) * torch.sin(y+z) + torch.exp(z) * torch.cos(x+y)),
            -(torch.exp(y) * torch.sin(z+x) + torch.exp(x) * torch.cos(y+z)),
            -(torch.exp(z) * torch.sin(x+y) + torch.exp(y) * torch.cos(x+z)),
            -0.5 * (torch.exp(2*x) + torch.exp(2*y) + torch.exp(2*z) + 2 * torch.sin(x+y) * torch.cos(x+z) * torch.exp(y+z) + 
                                                                        2 * torch.sin(y+z) * torch.cos(x+y) * torch.exp(x+z) +
                                                                        2 * torch.sin(x+z) * torch.cos(y+z) * torch.exp(x+y))
        ]

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

    def lamb(model, arg):
        f, u, v, w, p, x, y, z = basic_symbols(model, arg) 
        u_x, u_y, u_z = unpack(grad(u, arg))
        v_x, v_y, v_z = unpack(grad(v, arg))
        w_x, w_y, w_z = unpack(grad(w, arg))
        p_x, p_y, p_z = unpack(grad(p, arg))
        w1 = w_y - v_z
        w2 = u_z = w_x
        w3 = v_x - u_y
        return [v * w3 - w * w2, 
                w * w1 - u * w3,
                u * w2 - v * w1]


    def walls(model, arg):
        f, u, v, w, p, x, y, z = basic_symbols(model, arg)
        return [u-solution(x, y, z)[0], v-solution(x, y, z)[1], w-solution(x, y, z)[2], p-solution(x, y, z)[3]]

    domain = Hypercube([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])

    shell = Shell(domain)

    pde = [
        Condition(inner, domain),
        Condition(lamb, domain),
        Condition(walls, shell),
    ]

    return pde, input_dim, output_dim
