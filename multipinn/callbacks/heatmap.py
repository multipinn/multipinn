from __future__ import annotations

import gc
from typing import Callable, Literal, Union

import numpy as np
import plotly.graph_objects as go
import torch

from ..trainer import Trainer
from ..visualization.figures_2d import plot_2d_heatmap, plot_2d_scatter
from ..visualization.figures_3d import slider_surface_3d
from .base_callback import BaseCallback, BaseCallbackWithGrad
from .grid import Grid, GridWithGrad
from .save import BaseImageSave


class BaseHeatmap(BaseImageSave):
    """Base class for creating and saving heatmap visualizations.

    Provides core functionality for generating heatmap plots in 1D, 2D, or 3D
    based on grid data. Supports multiple output formats and customizable styling.

    Attributes:
        grid (Grid): Grid containing points for evaluation
        output_index (int): Index of output component to visualize
        style (dict): Plotly figure styling options
    """

    def __init__(
        self,
        grid: Grid,
        save_dir: str,
        period: int = 500,
        save_mode: Literal["html", "png", "pt", "show"] = "html",
        output_index: int = 0,
        style: dict = None,
    ):
        """Initialize the heatmap visualization.

        Args:
            grid (Grid): Grid for evaluation points
            save_dir (str): Directory where heatmaps will be saved
            period (int, optional): Update frequency in epochs. Defaults to 500.
            save_mode (Literal["html", "png", "pt", "show"]): Output format
            output_index (int, optional): Which output component to plot. Defaults to 0.
            style (dict, optional): Plotly figure styling options
        """
        super().__init__(period, save_dir + f"_{output_index}", save_mode)
        self.grid = grid
        self.output_index = output_index
        self.style = style if style is not None else {}

    def draw(self, values: torch.Tensor, plot_name: str, file_name: str = None):
        n = len(self.grid.coord)
        if n == 3:
            self.draw_3D(values, plot_name, file_name)
        elif n == 2:
            self.draw_2D(values, plot_name, file_name)
        elif n == 1:
            self.draw_1D(values, plot_name, file_name)
        else:
            raise Exception(f"Can't plot {n}D grid")

    def dict_data(self, fig):
        n = len(self.grid.coord)
        if n == 3:
            values = fig.data[0].marker.color
        elif n == 2:
            values = fig.data[0].z
        else:
            values = fig.data[0].y
        data = {
            "points": np.array(self.grid.points.detach().cpu().numpy()),
            "values": values,
        }
        return data

    def draw_3D(self, values: torch.Tensor, plot_name: str, file_name: str = None):
        assert len(self.grid.coord) == 3
        volume = np.reshape(values, self.grid.coord[0].shape).T
        fig = slider_surface_3d(
            self.grid.coord[0][:, 0, 0],
            self.grid.coord[1][0, :, 0],
            self.grid.coord[2][0, 0, :],
            volume,
            values.min().item(),
            values.max().item(),
            plot_name,
        )
        fig.update(**self.style)
        self.save_fig(fig, file_name)
        del fig
        gc.collect()

    def draw_2D(self, values: torch.Tensor, plot_name: str, file_name: str):
        assert len(self.grid.coord) == 2
        fig = plot_2d_heatmap(
            self.grid.coord[0].flatten(),
            self.grid.coord[1].flatten(),
            values.flatten(),
            self.style,
        )

        # fig.update_layout(title=plot_name)
        self.save_fig(fig, file_name)
        del fig
        gc.collect()

    def draw_1D(self, values: torch.Tensor, plot_name: str, file_name: str = None):
        assert len(self.grid.coord) == 1
        fig = plot_2d_scatter(
            self.grid.coord[0].flatten(), values.flatten(), self.style
        )
        fig.update_layout(title=plot_name)
        self.save_fig(fig, file_name)
        del fig
        gc.collect()

    def reset(
        self,
        new_save_dir: str = None,
        new_grid: Grid = None,
        new_output_index: int = None,
    ) -> None:
        super().reset(new_save_dir)
        if new_grid is not None:
            self.grid = new_grid
        if new_output_index is not None:
            self.output_index = new_output_index


class HeatmapError(BaseCallback, BaseHeatmap):
    """Callback for visualizing error distributions during training.

    Creates heatmap plots showing the absolute difference between predicted
    values and known solutions across the domain.

    Example:
        >>> error_plot = HeatmapError("./plots", 100, grid, exact_solution)
        >>> trainer.add_callback(error_plot)
    """

    def __init__(
        self,
        save_dir: str,
        period: int,
        grid: Grid,
        solution: Callable[[torch.Tensor], torch.Tensor],
        save_mode: Literal["html", "png", "pt", "show"] = "html",
        output_index: int = 0,
        style: dict = None,
    ):
        """Initialize error heatmap visualization.

        Args:
            save_dir (str): Directory for saving heatmaps
            period (int): Update frequency in epochs
            grid (Grid): Grid for evaluation points
            solution (Callable[[torch.Tensor], torch.Tensor]): Exact solution function
            save_mode (Literal["html", "png", "pt", "show"]): Output format
            output_index (int, optional): Which output to visualize. Defaults to 0.
            style (dict, optional): Plotly figure styling options
        """
        BaseHeatmap.__init__(
            self,
            grid,
            save_dir + "/heatmap_error_plots",
            period=period,
            save_mode=save_mode,
            output_index=output_index,
            style=style,
        )
        self.solution = solution

    def __call__(self, trainer: Trainer):
        if trainer.current_epoch % self.period == 0 and trainer.current_epoch != 0:
            predicted_values = (
                trainer.pinn.model(self.grid.points)[:, self.output_index].cpu().numpy()
            )
            exact = self.solution(self.grid.points).cpu().numpy()
            error = np.abs(predicted_values - exact)
            super().draw(
                error,
                "Error",
                f"{trainer.current_epoch}_heatmap_error_{self.output_index}",
            )


class HeatmapPrediction(BaseCallback, BaseHeatmap):
    """Callback for visualizing model predictions during training.

    Creates heatmap plots showing the predicted values across the domain,
    helpful for monitoring how predictions evolve during training.
    """

    def __init__(
        self,
        save_dir: str,
        period: int,
        grid: Grid,
        save_mode: Literal["html", "png", "pt", "show"] = "png",
        output_index: int = 0,
        style: dict = None,
    ):
        BaseHeatmap.__init__(
            self,
            grid,
            save_dir + "/heatmap_prediction_plots",
            period=period,
            save_mode=save_mode,
            output_index=output_index,
            style=style,
        )

    def __call__(self, trainer: Trainer):
        if trainer.current_epoch % self.period == 0 and trainer.current_epoch != 0:
            predicted_values = trainer.pinn.model(self.grid.points)[
                :, self.output_index
            ]
            super().draw(
                predicted_values.cpu().numpy(),
                "Prediction",
                f"{trainer.current_epoch}_heatmap_prediction_{self.output_index}",
            )


class PlotHeatmapSolution(BaseHeatmap):
    """One-time plotting of exact solution as a heatmap.

    Creates a reference heatmap visualization of the known solution.
    Note that this is not a callback - it creates a single plot when instantiated.
    """

    def __init__(
        self,
        save_dir: str,
        grid: Grid,
        solution: Callable[[torch.Tensor], torch.Tensor],
        save_mode: Literal["html", "png", "pt", "show"] = "html",
        style: dict = None,
    ):
        BaseHeatmap.__init__(
            self,
            grid,
            save_dir + "/heatmap_solution_plot",
            save_mode=save_mode,
            style=style,
        )
        self.solution = solution
        exact = self.solution(self.grid.points).detach().cpu().numpy()
        assert len(exact.shape) == 1  # solution.shape == (bs,)
        if save_mode != "show":
            super().mkdir()
        super().draw(exact, "Solution", "heatmap_solution_plot")

    def __call__(self, trainer: Trainer):
        print(
            "PlotHeatmapSolution is not a callback! You can remove it from callback list"
        )
        pass


class PlotHeatmapResidual(BaseCallbackWithGrad, BaseHeatmap):
    def __init__(
        self,
        save_dir: str,
        period: int,
        grid: Union[Grid, GridWithGrad],
        condition_index: int = 0,
        equation_index: int = 0,
        save_mode: Literal["html", "png", "pt", "show"] = "html",
        style: dict = None,
    ):
        if isinstance(grid, GridWithGrad):
            pass
        elif isinstance(grid, Grid):
            grid = GridWithGrad.from_Grid(grid)
        else:
            raise ValueError("Please provide an instance of Grid or GridWithGrad")
        BaseHeatmap.__init__(
            self,
            grid,
            save_dir + "/heatmap_residual_plots",
            period=period,
            save_mode=save_mode,
            style=style,
        )
        self.condition_index = condition_index
        self.equation_index = equation_index

    def __call__(self, trainer: Trainer):
        if trainer.current_epoch % self.period == 0 and trainer.current_epoch != 0:
            residual_fn = trainer.pinn.conditions[self.condition_index].get_residual_fn(
                trainer.pinn.model
            )
            residual_values = (
                residual_fn(self.grid.points)[self.equation_index]
                .detach()
                .cpu()
                .numpy()
            )
            super().draw(
                values=residual_values,
                plot_name="Residuals",
                file_name=f"{trainer.current_epoch}_heatmap_residual_{self.condition_index}.{self.equation_index}",
            )


class PlotHeatmapLoss(PlotHeatmapResidual):
    def __init__(
        self,
        save_dir: str,
        period: int,
        grid: Union[Grid, GridWithGrad],
        condition_index: int = 0,
        equation_index: int = 0,
        save_mode: Literal["html", "png", "pt", "show"] = "html",
        style: dict = None,
    ):
        PlotHeatmapResidual.__init__(
            self,
            save_dir,
            period,
            grid,
            condition_index,
            equation_index,
            save_mode,
            style,
        )

    def __call__(self, trainer: Trainer):
        if trainer.current_epoch % self.period == 0 and trainer.current_epoch != 0:
            residual_values = (
                trainer.pinn.calculate_loss_on_points(
                    trainer.pinn.conditions[self.condition_index], self.grid.points
                )[self.equation_index]
                .detach()
                .cpu()
                .numpy()
            )
            super().draw(
                values=residual_values,
                plot_name="Loss",
                file_name=f"{trainer.current_epoch}_heatmap_loss_{self.condition_index}.{self.equation_index}",
            )
