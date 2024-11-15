import gc
from typing import Literal

import plotly.graph_objects as go
import torch

from ..trainer import Trainer
from ..visualization.figures_2d import plot_2d_scatter, scatter_2d
from ..visualization.figures_3d import plot_3d_scatter, scatter_3d
from .base_callback import BaseCallback
from .save import BaseImageSave


class BaseScatter(BaseImageSave):
    """Base class for creating scatter plots with customizable saving options.

    This class provides fundamental functionality for generating and saving 2D and 3D scatter plots,
    with support for different file formats and styling options.

    Args:
        save_dir (str): Directory path where scatter plots will be saved.
        period (int, optional): Number of epochs between plot updates. Defaults to 500.
        save_mode (Literal["html", "png", "pt", "show"]): Format for saving plots. Defaults to "html".
        style (dict, optional): Custom styling options for the plots. Defaults to None.
    """

    def __init__(
        self,
        save_dir: str,
        period: int = 500,
        save_mode: Literal["html", "png", "pt", "show"] = "html",
        style: dict = None,
    ):
        super().__init__(period, save_dir, save_mode)
        self.style = style if style is not None else {}

    def draw(self, points: torch.Tensor, plot_name: str, file_name: str = None):
        """Creates and saves a scatter plot based on the dimensionality of input points.

        Args:
            points (torch.Tensor): Points to plot, shape (num_points, dim) where dim is 2 or 3.
            plot_name (str): Title of the plot.
            file_name (str, optional): Name for the saved file. Defaults to None.

        Raises:
            Exception: If points dimension is not 2 or 3.
        """
        n = points.shape[1]
        if n == 3:
            self.draw_3D(points, plot_name, file_name)
        elif n == 2:
            self.draw_2D(points, plot_name, file_name)
        else:
            raise Exception(f"Can't plot {n}D scatter")

    def dict_data(self, fig):
        data = {"x": fig.data[0].x}
        if hasattr(fig.data[0], "y"):
            data.update({"y": fig.data[0].y})
        if hasattr(fig.data[0], "z"):
            data.update({"z": fig.data[0].z})
        return data

    def draw_3D(self, points: torch.Tensor, plot_name: str, file_name: str = None):
        fig = plot_3d_scatter(points[:, 0], points[:, 1], points[:, 2], self.style)
        self.save_fig(fig, file_name)
        del fig
        gc.collect()

    def draw_2D(self, points: torch.Tensor, plot_name: str, file_name: str):
        fig = plot_2d_scatter(points[:, 0], points[:, 1], self.style)
        self.save_fig(fig, file_name)
        del fig
        gc.collect()


class ScatterPoints(BaseScatter, BaseCallback):
    """Callback for creating scatter plots of points from a specific condition during training.

    Visualizes the distribution of points used in training for a particular condition,
    helping to monitor the sampling strategy.

    Args:
        save_dir (str): Directory path where scatter plots will be saved.
        period (int, optional): Number of epochs between plot updates. Defaults to 500.
        save_mode (Literal["html", "png", "pt", "show"]): Format for saving plots. Defaults to "html".
        condition_index (int, optional): Index of the condition to plot. Defaults to 0.
        style (dict, optional): Custom styling options for the plots. Defaults to None.
    """

    def __init__(
        self,
        save_dir: str,
        period: int = 500,
        save_mode: Literal["html", "png", "pt", "show"] = "html",
        condition_index: int = 0,
        style: dict = None,
    ):
        BaseScatter.__init__(
            self,
            save_dir + f"/points_plots_{condition_index}",
            period,
            save_mode,
            style,
        )
        self.condition_index = condition_index

    def __call__(self, trainer: Trainer):
        if trainer.current_epoch % self.period == 0:
            self.draw(
                trainer.pinn.conditions[self.condition_index]
                .points.detach()
                .cpu()
                .numpy(),
                "Points",
                f"{trainer.current_epoch}_points",
            )


class LiveScatterPrediction(BaseCallback, BaseImageSave):
    """Callback for creating live scatter plots of model predictions during training.

    Creates color-coded scatter plots where point colors represent the model's predictions,
    providing visual feedback on how the model's predictions evolve during training.

    Args:
        save_dir (str): Directory path where scatter plots will be saved.
        period (int): Number of epochs between plot updates.
        save_mode (Literal["html", "png", "pt", "show"]): Format for saving plots. Defaults to "html".
        output_index (int, optional): Index of the output dimension to plot. Defaults to 0.
        style (dict, optional): Custom styling options for the plots. Defaults to None.

    Methods:
        draw: Creates a scatter plot with points colored by their predicted values.
        draw_2D, draw_3D: Specialized plotting functions for 2D and 3D data.
        calculate_and_draw_on_points: Computes model predictions and creates the plot.
    """

    def __init__(
        self,
        save_dir: str,
        period: int,
        save_mode: Literal["html", "png", "pt", "show"] = "html",
        output_index: int = 0,
        style: dict = None,
    ):
        self.style = style if style is not None else {}
        BaseImageSave.__init__(
            self,
            period,
            save_dir + "/live_scatter_pred_plots",
            save_mode=save_mode,
        )
        self.output_index = output_index

    def draw(
        self,
        points: torch.Tensor,
        values: torch.Tensor,
        plot_name: str,
        file_name: str = None,
    ):
        if points.shape[1] == 3:
            self.draw_3D(points, values, plot_name, file_name)
        elif points.shape[1] == 2:
            self.draw_2D(points, values, plot_name, file_name)
        else:
            raise ValueError

    def draw_3D(
        self,
        points: torch.Tensor,
        values: torch.Tensor,
        plot_name: str,
        file_name: str = None,
    ):
        fig = go.Figure(
            scatter_3d(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                values.flatten(),
                values.min().item(),
                values.max().item(),
                name=plot_name,
            )
        )
        default_style = {
            "layout_autosize": True,
            "layout_scene_aspectmode": "data",
        }
        fig.update(**(default_style | self.style))
        self.save_fig(fig, file_name)
        del fig
        gc.collect()

    def draw_2D(
        self,
        points: torch.Tensor,
        values: torch.Tensor,
        plot_name: str,
        file_name: str = None,
    ):
        fig = go.Figure(
            scatter_2d(
                points[:, 0],
                points[:, 1],
                values.flatten(),
                values.min().item(),
                values.max().item(),
                name=plot_name,
            )
        )
        default_style = {
            "layout_autosize": True,
            "layout_xaxis": dict(scaleanchor="y", scaleratio=1),
            "layout_yaxis": dict(scaleanchor="x", scaleratio=1),
        }
        fig.update(**(default_style | self.style))
        self.save_fig(fig, file_name)
        del fig
        gc.collect()

    def calculate_and_draw_on_points(self, trainer: Trainer, points: torch.Tensor):
        predicted_values = trainer.pinn.model(points)[:, self.output_index]
        self.draw(
            points.detach().cpu().numpy(),
            predicted_values.cpu().numpy(),
            "Prediction",
            f"{trainer.current_epoch}_scatter_prediction_{self.output_index}",
        )

    def __call__(self, trainer: Trainer):
        if trainer.current_epoch % self.period == 0:
            points = []
            for cond in trainer.pinn.conditions:
                points.append(cond.points)
            points = torch.concatenate(points)
            self.calculate_and_draw_on_points(trainer, points)


class MeshScatterPrediction(LiveScatterPrediction):
    """Callback for creating scatter plots of model predictions on a fixed mesh.

    Extends LiveScatterPrediction to work with a predefined set of mesh points,
    useful for monitoring model predictions on a consistent evaluation grid.

    Args:
        save_dir (str): Directory path where scatter plots will be saved.
        period (int): Number of epochs between plot updates.
        points (torch.Tensor): Fixed mesh points to evaluate, shape (num_points, dim).
        save_mode (Literal["html", "png", "pt", "show"]): Format for saving plots. Defaults to "html".
        output_index (int, optional): Index of the output dimension to plot. Defaults to 0.
        style (dict, optional): Custom styling options for the plots. Defaults to None.

    Note:
        Unlike LiveScatterPrediction, this class evaluates the model on the same fixed set
        of points throughout training, making it easier to track changes in specific regions.
    """

    def __init__(
        self,
        save_dir: str,
        period: int,
        points: torch.Tensor,
        save_mode: Literal["html", "png", "pt", "show"] = "html",
        output_index: int = 0,
        style: dict = None,
    ):
        super().__init__(save_dir, period, save_mode, output_index, style)
        self.save_dir = save_dir + "/mesh_scatter_pred_plots"  # override
        self.points = torch.Tensor(points)

    def __call__(self, trainer: Trainer):
        if trainer.current_epoch % self.period == 0 and trainer.current_epoch != 0:
            self.calculate_and_draw_on_points(trainer, self.points)
