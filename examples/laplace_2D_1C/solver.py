import numpy as np
import torch
from torch.optim.lr_scheduler import CyclicLR

from examples.laplace_2D_1C.problem import laplace2D_1C
from multipinn import *

set_device_and_seed(42)

conditions, input_dim, output_dim = laplace2D_1C()

# model = torch.load("Delta_start_0.01_FNN/checkpoints/40000.pth")
# model = FNN(layers_all=[input_dim, 16, output_dim])
model = FNN(input_dim, output_dim, [128, 128, 128, 128, 128])
# model = ResNet(layers_all=[input_dim, 128, output_dim], blocks=[2])
# model = ResNet(layers_all=[input_dim, 64, 64, 64, output_dim], blocks=[64])

n_points = 1024
num_epochs = 600
Generator(n_points // 8, "pseudo").use_for(conditions)  # default
GradBasedGenerator(n_points, "pseudo").use_for(conditions[0])
Generator(4, "pseudo").use_for(conditions[2:4])

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

save_dir = "my_output"
grid_big = Grid.from_pinn(pinn, 8001)
grid = Grid.from_pinn(pinn, 1001)
font_style = {"layout_font": dict(family="Times new roman", size=24)}
curve_style = {"layout_xaxis_range": [0, num_epochs]} | font_style
heatmap_style = {
    "data_0_zmin": None,
    "data_0_zmax": None,
    "data_0_colorscale": "Jet",
    "layout_xaxis": dict(scaleanchor="y", scaleratio=1),
    "layout_yaxis": dict(scaleanchor="x", scaleratio=1),
} | font_style
callbacks = [
    TqdmBar("Epoch {epoch} lr={lr:.2e} Loss={loss_eq} Total={total_loss:.2e}"),
    LearningRateCurve(save_dir, 1000, style={"layout_yaxis_type": "linear"}),
    LossCurve(save_dir, 500, save_mode="html", style=curve_style),
    GridResidualCurve(save_dir, 1000, grid=grid, save_mode="html", style=curve_style),
    HeatmapPrediction(
        save_dir, 1000, grid=grid_big, save_mode="html", style=heatmap_style
    ),
    # ErrorCurve(save_dir, 100, full_solution=full_solution),
    # GridErrorCurve(save_dir, 100, grid=grid, full_solution=full_solution),
    # HeatmapError(save_dir, 500, grid=grid, solution=solution0),
    # SaveModel(save_dir, period=1000),
    # GridResidualCurve(save_dir, 500, GridWithGrad.from_condition(pinn.conditions[1], 51), condition_index=1),
]
# PlotHeatmapSolution(save_dir, grid=grid, solution=solution0)

trainer = TrainerOneBatch(
    pinn=pinn,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=num_epochs,
    # num_batches=1,
    update_grid_every=200,
    callbacks_organizer=CallbacksOrganizer(callbacks),
)
trainer.train()

# def errors(args): # final error heatmap
#     return exact_solution(args) - model(args)[:, 0].detach().cpu()
# src.callbacks.heatmap.PlotHeatmapSolution(save_dir, grid, solution=errors)
