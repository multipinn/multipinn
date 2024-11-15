import numpy as np
import torch
from torch.optim.lr_scheduler import CyclicLR

from examples.heat_2D_simple.problem import heat_2D_simple
from multipinn import *
from multipinn.geometry import *
from multipinn.utils import set_device_and_seed

# from examples.navier_stokes_3D.problem import navier_stokes_3D


def save(data, name):
    from multipinn.visualization.figures_2d import plot_2d_scatter

    fig = plot_2d_scatter(data[:, 0], data[:, 1], {})
    fig.write_html(name + ".html")


def save3D(data, name):
    from multipinn.visualization.figures_3d import plot_3d_scatter

    fig = plot_3d_scatter(data[:, 0], data[:, 1], data[:, 2], {})
    fig.write_html(name + ".html")


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


def tore(small_radius, big_radius, hole_axis=1):
    if hole_axis == 0:
        result = Hypersphere((0, big_radius), small_radius)
        result = DomainAxisymmetricExtension(result, 1)
        return result
    elif hole_axis == 1:
        result = Hypersphere((big_radius, 0), small_radius)
        result = DomainAxisymmetricExtension(result, 0)
        return result
    else:
        raise ValueError


def pipe_segment_xz(start_x, start_z, end_x, end_z, radius):
    if start_z == end_z:
        domain = Hypercube((start_x, -radius), (end_x, radius))
        domain = DomainAxisymmetricExtension(domain, 1)
        domain = DomainShift(domain, (0, 0, start_z))
    elif start_x == end_x:
        domain = Hypercube((start_z, -radius), (end_z, radius))
        domain = DomainAxisymmetricExtension(domain, 1)
        domain = DomainShift(domain, (0, 0, start_x))
        return DomainPermute(domain, (2, 1, 0))
    else:
        raise ValueError
    return domain


def pipe_curve_xz(out_dir, cross_x, cross_z, big_radius, small_radius):
    assert small_radius <= big_radius
    domain = tore(small_radius, big_radius, 1)

    total_radius = big_radius + small_radius
    if out_dir == "lu" or out_dir == "ul":
        domain = domain & Hypercube(
            (0, -small_radius, -total_radius), (total_radius, small_radius, 0)
        )
        domain = DomainShift(domain, (cross_x - big_radius, 0, cross_z + big_radius))
    elif out_dir == "ld" or out_dir == "dl":
        domain = domain & Hypercube(
            (0, -small_radius, 0), (total_radius, small_radius, total_radius)
        )
        domain = DomainShift(domain, (cross_x - big_radius, 0, cross_z - big_radius))
    elif out_dir == "ru" or out_dir == "ur":
        domain = domain & Hypercube(
            (-total_radius, -small_radius, -total_radius), (0, small_radius, 0)
        )
        domain = DomainShift(domain, (cross_x + big_radius, 0, cross_z + big_radius))
    elif out_dir == "rd" or out_dir == "dr":
        domain = domain & Hypercube(
            (-total_radius, -small_radius, 0), (0, small_radius, total_radius)
        )
        domain = DomainShift(domain, (cross_x + big_radius, 0, cross_z - big_radius))
    else:
        raise ValueError
    return domain


def Z_shaped_pipe():
    radius = 0.5
    curve_radius = 1
    total_length = 9
    step_z = 2
    x_mid = total_length / 2
    input_segment = pipe_segment_xz(0, 0, x_mid - curve_radius, 0, radius)
    output_segment = pipe_segment_xz(
        x_mid + curve_radius, step_z, total_length, step_z, radius
    )
    curve1 = pipe_curve_xz("lu", x_mid, 0, curve_radius, radius)
    curve2 = pipe_curve_xz("dr", x_mid, step_z, curve_radius, radius)
    domain = input_segment | curve1 | curve2 | output_segment
    if step_z > 2 * curve_radius:
        domain |= pipe_segment_xz(
            x_mid, curve_radius, x_mid, step_z - curve_radius, radius
        )

    input = DomainAxisymmetricExtension(
        Hypercube(low=[0.0, -radius], high=[0.0, radius]), 1
    )
    output = DomainAxisymmetricExtension(
        Hypercube(
            low=[total_length, step_z - radius], high=[total_length, step_z + radius]
        ),
        1,
    )

    shell = Shell(domain)
    walls = shell - (input | output)
    input_wall = shell & input
    output_wall = shell & output

    save3D(walls.random_points(5000), "border")
    # save3D(Shell(domain).random_points(5000), 'border')
    save3D(domain.random_points(5000), "inner")


def demo_geom():
    nozzle1_start_x = 2
    nozzle1_end_x = 5
    cross_y = 0.5
    nozzle2_start_x = 5
    nozzle2_end_x = 8

    straight1 = Hypercube([0, -1], [nozzle1_start_x, 1])
    nozzle1 = nozzle_pipe(nozzle1_start_x, -1, 1, nozzle1_end_x, -cross_y, cross_y)
    nozzle2 = nozzle_pipe(nozzle2_start_x, -cross_y, cross_y, nozzle2_end_x, -1, 1)
    if nozzle1_end_x != nozzle2_start_x:
        straight2 = Hypercube([nozzle1_end_x, -cross_y], [nozzle2_start_x, cross_y])
        nozzle_combined = nozzle1 | straight2 | nozzle2
    else:
        nozzle_combined = nozzle1 | nozzle2
    straight3 = Hypercube([nozzle2_end_x, -1], [10, 1])
    domain = straight1 | nozzle_combined | straight3

    # input = Hypercube([0.0, -1.0], [0.0, 1.0])
    # output = Hypercube([10.0, -1.0], [10.0, 1.0])
    # shell = Shell(domain)
    # walls = shell - (input | output)
    #
    # save(Shell(domain).random_points(1000), 'border')
    # save(domain.random_points(1000), 'inner')

    domain_3D = DomainAxisymmetricExtension(domain, 1)
    save3D(Shell(domain_3D).random_points(5000), "border")
    save3D(domain_3D.random_points(5000), "inner")


def pipe_2d():
    # r = 0.5
    # R = 1
    # domain = Hypercube([0, 0], [4, 1]) | Hypercube([4, 1], [5, 2]) | Hypercube([5, 2], [9, 3])\
    #          | Hypercube([3+r, 0], [5, 2-r]) - Hypersphere([3+r, 2-r], r) & Hypersphere([3+R, 2-R], R)\
    #          | Hypercube([4, 2-r], [5+r, 3]) - Hypersphere([5+r, 2-r], r) & Hypersphere([6-R, 1+R], R)

    def create_domain(t):
        radius_smooth = 1

        radius_square = 0.1

        radius = (1 - t) * radius_smooth + t * radius_square

        domain = (
            Hypercube([0, 0], [5 - radius, 1])
            | Hypercube([4, 0 + radius], [5, 3 - radius])
            | Hypercube([4 + radius, 2], [9, 3])
            | Hypercube([4, 0], [5, 1]) & Hypersphere([5 - radius, 0 + radius], radius)
            | Hypercube([4, 2], [5, 3]) & Hypersphere([4 + radius, 3 - radius], radius)
        )
        return domain

    domain = create_domain(0)
    save(Shell(domain).random_points(1000), "border")
    # save(domain.random_points(5000), 'inner')


# demo_geom()
# Z_shaped_pipe()
# pipe_2d()
# exit(0)


set_device_and_seed(42)

conditions, input_dim, output_dim = heat_2D_simple()
# conditions, input_dim, output_dim = navier_stokes_3D()

# model = torch.load("Delta_start_0.01_FNN/checkpoints/40000.pth")
# model = FNN(layers_all=[input_dim, 16, output_dim])
model = FNN(input_dim, output_dim, [128, 128, 128])
# model = ResNet(layers_all=[input_dim, 5, 10, 15, 20, 25, output_dim], blocks=[1, 6, 5, 6, 2]).to(device)
# model = ResNet(layers_all=[input_dim, 128, output_dim], blocks=[2])

n_points = 1024
num_epochs = 1000
Generator(n_points // 16, "pseudo").use_for(conditions[1:])
GradBasedGenerator(n_points, "pseudo", 1).use_for(conditions[0])

pinn = PINN(model=model, conditions=conditions)
optimizer = torch.optim.Adam(model.parameters())
scheduler = CyclicLR(
    optimizer,
    base_lr=1e-6,
    max_lr=1e-3,
    step_size_up=500,
    step_size_down=None,
    mode="exp_range",
    cycle_momentum=False,
    gamma=0.999,
)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)


def full_solution(args):
    return (torch.exp(0.5 * args[:, 0]) + 0.5 * args[:, 1])[:, None]


save_dir = "my_output"
grid_big = Grid.from_pinn(pinn, 8001)
grid = Grid.from_pinn(pinn, 1001)
font_style = {"layout_font": dict(family="Times new roman", size=24)}
curve_style = {"layout_xaxis_range": [0, num_epochs]} | font_style
marker_range_style = {"data_0_marker_cmin": 1, "data_0_marker_cmax": 3.5}
heatmap_style = {
    "data_0_zmin": None,
    "data_0_zmax": None,
    "data_0_colorscale": "Jet",
} | font_style
points_style_fn = lambda bbox: {
    "layout_xaxis_range": (bbox[0][0], bbox[1][0]),
    "layout_yaxis_range": (bbox[0][1], bbox[1][1]),
}
callbacks = [
    TqdmBar("Epoch {epoch} lr={lr:.2e} Loss={loss_eq} Total={total_loss:.2e}"),
    LearningRateCurve(save_dir, 1000, style={"layout_yaxis_type": "linear"}),
    LossCurve(save_dir, 500, save_mode="html", style=curve_style),
    GridResidualCurve(save_dir, 500, grid=grid, save_mode="html", style=curve_style),
    HeatmapPrediction(
        save_dir, 500, grid=grid_big, save_mode="html", style=heatmap_style
    ),
    ErrorCurve(save_dir, 500, full_solution=full_solution, style=curve_style),
    # GridErrorCurve(save_dir, 100, grid=grid, full_solution=full_solution),
    # HeatmapError(save_dir, 500, grid=grid, solution=solution0),
    # SaveModel(save_dir, period=1000),
    # ScatterPoints(save_dir, 100, "html", condition_index=0, style=points_style_fn(pinn.conditions[0].geometry.bbox)),
    # GridResidualCurve(save_dir, 500, GridWithGrad.from_condition(pinn.conditions[1], 51), condition_index=1),
    # MeshErrorCurve.from_file(save_dir, 100, "nodes_corase.csv", "fields_coarse.csv"),
    LiveScatterPrediction(save_dir, 100, style=marker_range_style),
]
# PlotHeatmapSolution(save_dir, grid=grid, solution=solution0)

trainer = TrainerOneBatch(
    pinn=pinn,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=num_epochs,
    # num_batches=1,
    update_grid_every=100,
    callbacks_organizer=CallbacksOrganizer(callbacks),
    calc_loss="mean",
)
trainer.train()

# def errors(args): # final error heatmap
#     return exact_solution(args) - model(args)[:, 0].detach().cpu()
# src.callbacks.heatmap.PlotHeatmapSolution(save_dir, grid, solution=errors)
