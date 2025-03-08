import torch

from multipinn.condition.condition import Condition
from multipinn.condition.diff import grad, unpack
from multipinn.mesh.grid_reader import GridReader
from multipinn.mesh.mesh import MeshArea


def problem_3D1C_laplace_mesh(mesh_file_path=None):
    input_dim = 3
    output_dim = 1

    def solution(x, y, z):
        cos_pix = torch.cos(torch.pi * x)
        numerator = (cos_pix - 1) * torch.exp(2 * y)
        denominator = 11 * (2 + z)
        return numerator / denominator

    mesh_grid = GridReader().read(mesh_file_path)

    domain = MeshArea(mesh_grid.get_face_by_id(3), mesh_grid.dim)
    all_bnd = MeshArea(mesh_grid.get_face_by_id(2), mesh_grid.dim)

    def basic_symbols(model, arg):
        f = model(arg)
        (u,) = unpack(f)
        x, y, z = unpack(arg)
        return f, u, x, y, z

    def inner(model, arg):
        f, u, x, y, z = basic_symbols(model, arg)
        u_x, u_y, u_z = unpack(grad(u, arg))
        u_xx, _, _ = unpack(grad(u_x, arg))
        _, u_yy, _ = unpack(grad(u_y, arg))
        _, _, u_zz = unpack(grad(u_z, arg))

        cos_pix = torch.cos(torch.pi * x)
        two_z_2 = torch.square(2 + z)
        numerator = torch.exp(2 * y) * (
            (4 * (cos_pix - 1) - torch.square(torch.tensor(torch.pi)) * cos_pix)
            * two_z_2
            + 2 * cos_pix
            - 2
        )
        denominator = 11 * (2 + z) * two_z_2
        eq1 = u_xx + u_yy + u_zz - (numerator / denominator)
        return [eq1]

    def bc_all(model, arg):
        f, u, x, y, z = basic_symbols(model, arg)
        return [u - solution(x, y, z)]

    pde = [
        Condition(inner, domain),
        Condition(bc_all, all_bnd),
    ]

    return pde, input_dim, output_dim
