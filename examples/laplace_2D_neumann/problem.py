import numpy as np
import torch

from multipinn.condition.condition import Condition
from multipinn.condition.diff import grad, unpack
from multipinn.geometry import *


def laplace_2D_neumann_problem():
    input_dim = 2
    output_dim = 1

    def basic_symbols(model, arg):
        f = model(arg)
        (u,) = unpack(f)
        x, y = unpack(arg)
        return f, u, x, y

    domain = Hypercube(low=[0, 0], high=[1, 1])
    x_min = Hypercube(low=[0, 0], high=[0, 1])
    x_max = Hypercube(low=[1, 0], high=[1, 1])
    y_min = Hypercube(low=[0, 0], high=[1, 0])
    y_max = Hypercube(low=[0, 1], high=[1, 1])

    def inner(model, arg):
        f, u, x, y = basic_symbols(model, arg)
        u_x, u_y = unpack(grad(u, arg))
        u_xx, u_xy = unpack(grad(u_x, arg))
        u_yx, u_yy = unpack(grad(u_y, arg))
        eq1 = u_xx + u_yy
        return [eq1]

    def bc1(model, arg):
        f, u, x, y = basic_symbols(model, arg)
        normals = torch.tensor([-1, 0]).repeat(arg.shape[0], 1)
        u_grad = grad(u, arg)
        u_n = torch.sum(u_grad * normals, 1)
        pi_trch = torch.ones_like(y) * np.pi
        return [
            u_n
            + (np.pi / (torch.exp(pi_trch) - torch.exp(-pi_trch)))
            * (torch.exp(pi_trch * y) - torch.exp(-pi_trch * y))
        ]

    def bc2(model, arg):
        f, u, x, y = basic_symbols(model, arg)
        normals = torch.tensor([1, 0]).repeat(arg.shape[0], 1)
        u_grad = grad(u, arg)
        u_n = torch.sum(u_grad * normals, 1)
        pi_trch = torch.ones_like(y) * np.pi

        return [
            u_n
            + (np.pi / (torch.exp(pi_trch) - torch.exp(-pi_trch)))
            * (torch.exp(pi_trch * y) - torch.exp(-pi_trch * y))
        ]

    def bc3(model, arg):
        f, u, x, y = basic_symbols(model, arg)
        normals = torch.tensor([0, -1]).repeat(arg.shape[0], 1)
        u_grad = grad(u, arg)
        u_n = torch.sum(u_grad * normals, 1)
        pi_trch = torch.ones_like(y) * np.pi
        return [
            u_n
            + (2 * np.pi / (torch.exp(pi_trch) - torch.exp(-pi_trch)))
            * torch.sin(pi_trch * x)
        ]

    def bc4(model, arg):
        f, u, x, y = basic_symbols(model, arg)
        normals = torch.tensor([0, 1]).repeat(arg.shape[0], 1)
        u_grad = grad(u, arg)
        u_n = torch.sum(u_grad * normals, 1)
        pi_trch = torch.ones_like(y) * np.pi
        return [
            u_n
            - (np.pi / (torch.exp(pi_trch) - torch.exp(-pi_trch)))
            * torch.sin(pi_trch * x)
            * (torch.exp(pi_trch) + torch.exp(-pi_trch))
        ]

    pde = [
        Condition(inner, domain),
        Condition(bc1, x_min),
        Condition(bc2, x_max),
        Condition(bc3, y_min),
        Condition(bc4, y_max),
    ]

    return pde, input_dim, output_dim
