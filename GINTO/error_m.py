from multipinn.geometry import *
from multipinn.condition import *
from multipinn import *
from multipinn.mesh import comsol_reader
from multipinn.generation.generator import Generator

import torch
import numpy as np
import matplotlib.pyplot as plt
from train_ginto import PINTO_2D


def load(model, model_path, device="cpu"):
    state_dict = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    return model

def plot_compare(obstacle, filepath, model_path, name):
    def mask_f(X, Y):
        points = np.stack([X, Y], axis=-1).reshape(-1, 2)
        mask_flat = obstacle.inside(points)
        return mask_flat.reshape(X.shape)

    device = torch.device("cpu")
    model = PINTO_2D().to(device)

    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
    model.eval()

    save_path = f"results/err/{name}.png"
    _, pts_exact, u_exact = comsol_reader.read_comsol_file(filepath)

    mask = mask_f
    rect_back = Hypercube([-5, -2.5], [5, 2.5])
    domain = rect_back - obstacle

    x_min = Hypercube(low=[-5.0, -2.5], high=[-5.0, 2.5])
    x_max = Hypercube(low=[5.0, -2.5], high=[5.0, 2.5])
    y_min = Hypercube(low=[-5.0, -2.5], high=[5.0, -2.5])
    y_max = Hypercube(low=[-5.0, 2.5], high=[5.0, 2.5])

    shell = Shell(domain)
    block = shell - (x_min | x_max | y_min | y_max)

    c_domain = Condition(lambda x: x, domain)
    c_block = Condition(lambda x: x, block)

    conds = [c_domain, c_block]

    generator_domain = Generator(
        n_points=40_000, sampler="Hammersley"
    )
    generator_block = Generator(
        n_points=500, sampler="Hammersley"
    )

    print("Генерим точки")
    generator_domain.use_for(c_domain)
    generator_block.use_for(c_block)

    for c in conds:
        c.update_points()

    domain_pts = c_domain.points.to(device)
    block_pts = c_block.points.to(device)

    with torch.no_grad():
        u_pred = model(domain_pts, block_pts)
    u = u_pred.cpu().numpy()
    pts = domain_pts.clone().cpu().detach().numpy()

    x = np.linspace(-5, 5, 200)
    y = np.linspace(-2.5, 2.5, 200)
    X, Y = np.meshgrid(x, y)

    from scipy.interpolate import griddata
    grid_u0_pred = griddata(pts, u[:,0], (X, Y), method='cubic')
    grid_u1_pred = griddata(pts, u[:,1], (X, Y), method='cubic')
    grid_u2_pred = griddata(pts, u[:,2], (X, Y), method='cubic')

    grid_u0_exact = griddata(pts_exact, u_exact[:,0], (X, Y), method='cubic')
    grid_u1_exact = griddata(pts_exact, u_exact[:,1], (X, Y), method='cubic')
    grid_u2_exact = griddata(pts_exact, u_exact[:,2], (X, Y), method='cubic')

    vmin_u0 = min(grid_u0_exact.min(), grid_u0_pred.min())
    vmax_u0 = max(grid_u0_exact.max(), grid_u0_pred.max())
    vmin_u1 = min(grid_u1_exact.min(), grid_u1_pred.min())
    vmax_u1 = max(grid_u1_exact.max(), grid_u1_pred.max())
    vmin_u2 = min(grid_u2_exact.min(), grid_u2_pred.min())
    vmax_u2 = max(grid_u2_exact.max(), grid_u2_pred.max())

    mask_plot = mask(X,Y)
    grid_u0_pred[mask_plot] = np.nan
    grid_u1_pred[mask_plot] = np.nan
    grid_u2_pred[mask_plot] = np.nan

    grid_u0_exact[mask_plot] = np.nan
    grid_u1_exact[mask_plot] = np.nan
    grid_u2_exact[mask_plot] = np.nan

    plt.figure(figsize=(40, 20))

    plt.subplot(3, 3, 1)
    plt.imshow(grid_u0_exact, extent=[-5, 5, -2.5, 2.5], origin='lower', cmap='jet', vmin=vmin_u0, vmax=vmax_u0)
    plt.colorbar()
    plt.title("Exact Solution for Velocity X Component")

    plt.subplot(3, 3, 2)
    plt.imshow(grid_u0_pred, extent=[-5, 5, -2.5, 2.5], origin='lower', cmap='jet', vmin=vmin_u0, vmax=vmax_u0)
    plt.colorbar()
    plt.title("GINTO Prediction for Velocity X Component")

    plt.subplot(3, 3, 3)
    plt.imshow(grid_u0_pred - grid_u0_exact, extent=[-5, 5, -2.5, 2.5], origin='lower', cmap='jet')
    plt.colorbar()
    plt.title("Absolute Error for Velocity X Component")


    plt.subplot(3, 3, 4)
    plt.imshow(grid_u1_exact, extent=[-5, 5, -2.5, 2.5], origin='lower', cmap='jet', vmin=vmin_u1, vmax=vmax_u1)
    plt.colorbar()
    plt.title("Exact Solution for Velocity Y Component")

    plt.subplot(3, 3, 5)
    plt.imshow(grid_u1_pred, extent=[-5, 5, -2.5, 2.5], origin='lower', cmap='jet', vmin=vmin_u1, vmax=vmax_u1)
    plt.colorbar()
    plt.title("GINTO Prediction for Velocity Y Component")

    plt.subplot(3, 3, 6)
    plt.imshow(grid_u1_pred - grid_u1_exact, extent=[-5, 5, -2.5, 2.5], origin='lower', cmap='jet')
    plt.colorbar()
    plt.title("Absolute Error for Velocity Y Component")


    plt.subplot(3, 3, 7)
    plt.imshow(grid_u2_exact, extent=[-5, 5, -2.5, 2.5], origin='lower', cmap='jet', vmin=vmin_u2, vmax=vmax_u2)
    plt.colorbar()
    plt.title("Exact Solution for Pressure")

    plt.subplot(3, 3, 8)
    plt.imshow(grid_u2_pred, extent=[-5, 5, -2.5, 2.5], origin='lower', cmap='jet', vmin=vmin_u2, vmax=vmax_u2)
    plt.colorbar()
    plt.title("GINTO Prediction for Pressure")

    plt.subplot(3, 3, 9)
    plt.imshow(grid_u2_pred - grid_u2_exact, extent=[-5, 5, -2.5, 2.5], origin='lower', cmap='jet')
    plt.colorbar()
    plt.title("Absolute Error for Pressure")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    grid_u0_pred = np.nan_to_num(grid_u0_pred, nan=0.0)
    grid_u0_exact = np.nan_to_num(grid_u0_exact, nan=0.0)    
    print(f'\nX {name}: ', np.linalg.norm(grid_u0_pred - grid_u0_exact, ord=2) / np.linalg.norm(grid_u0_exact, ord=2))
    print(f'X_inf {name}: ', np.linalg.norm(grid_u0_pred - grid_u0_exact, ord=np.inf) / np.linalg.norm(grid_u0_exact, ord=np.inf))

    grid_u1_pred = np.nan_to_num(grid_u1_pred, nan=0.0)
    grid_u1_exact = np.nan_to_num(grid_u1_exact, nan=0.0)
    print(f'\nY {name}: ', np.linalg.norm(grid_u1_pred - grid_u1_exact, ord=2) / np.linalg.norm(grid_u1_exact, ord=2))
    print(f'Y_inf {name}: ', np.linalg.norm(grid_u1_pred - grid_u1_exact, ord=np.inf) / np.linalg.norm(grid_u1_exact, ord=np.inf))

    grid_u2_pred = np.nan_to_num(grid_u2_pred, nan=0.0)
    grid_u2_exact = np.nan_to_num(grid_u2_exact, nan=0.0)
    print(f'\nP {name}: ', np.linalg.norm(grid_u2_pred - grid_u2_exact, ord=2) / np.linalg.norm(grid_u2_exact, ord=2))
    print(f'P_inf {name}: ', np.linalg.norm(grid_u2_pred - grid_u2_exact, ord=np.inf) / np.linalg.norm(grid_u2_exact, ord=np.inf))

def read_polygon_from_file(filepath):

    vertices = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            x = float(parts[0])
            y = float(parts[1])
            vertices.append((x, y))

    return vertices

blocks = []

for i in range(1, 201):
    
    polygon_file = f"obstacles/train_ginto/test_200/test_{i}.txt"

    vertices = read_polygon_from_file(polygon_file)
    obstacle = Polygon(vertices=vertices)

    solution_path = f"obstacles/numeric_sol/test/{i}.txt"
    name =f"test_{i}"

    blocks.append((obstacle, solution_path, name))

for obsc, path, name in blocks:    
    plot_compare(obsc, path, "result/mod/mod_99000.pth", name)
