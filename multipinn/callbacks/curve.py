import gc
from typing import Callable, List, Literal, Sequence, Union

import numpy as np
import plotly.graph_objects as go
import torch

from ..trainer import Trainer
from .base_callback import BaseCallback, BaseCallbackWithGrad
from .grid import Grid, GridWithGrad
from .save import BaseImageSave


class BasicCurve(BaseImageSave):
    """Base class for creating and saving training curves.

    Provides core functionality for drawing and saving various training metrics as curves.
    Supports multiple output formats and customizable styling.

    Attributes:
        curve_names (List[str]): Names of curves to be plotted
        save_name (str): Base name for saving the curve files
        metric_history (List): History of metric values to plot
        style (dict): Plotly figure styling options
    """

    def __init__(
        self,
        save_dir: str,
        period: int,
        save_name: str,
        save_mode: Literal["html", "png", "pt", "show"] = "html",
        style: dict = None,
    ):
        """Initialize the curve plotting class.

        Args:
            save_dir (str): Directory where curve plots will be saved
            period (int): How often to update the curve (in epochs)
            save_name (str): Base name for saving curve files
            save_mode (Literal["html", "png", "pt", "show"]): Output format for curves
            style (dict, optional): Plotly figure styling options. Defaults to {}.
        """
        super().__init__(period, save_dir, save_mode=save_mode)
        self.curve_names = []
        self.save_name = save_name
        self.metric_history = []
        self.style = style if style is not None else {}

    def draw(
        self,
        values: Sequence,
        v_names: List[str],
        coord: np.ndarray,
    ):
        """Draw curves using plotly.

        Args:
            values (Sequence): Values to plot for each curve
            v_names (List[str]): Names for each curve
            coord (np.ndarray): X-axis coordinates
        """
        fig = go.Figure()
        for val, name in zip(values, v_names):
            fig.add_trace(go.Scatter(x=coord, y=val, mode="lines", name=name))
        fig.update(**self.style)
        self.save_fig(fig, self.save_name)
        del fig
        gc.collect()

    @staticmethod
    def dict_data(fig):
        """Extract Y values from plotly figure data.

        Args:
            fig (go.Figure): Plotly figure object

        Returns:
            dict: Dictionary mapping curve names to Y values
        """
        data = {}
        for d in fig.data:
            data[d.name] = d.y
        return data

    def init_curve_names(self, conditions):
        """Initialize curve names based on conditions.

        Args:
            conditions: Training conditions defining curve labels
        """
        for counter, cond in enumerate(conditions):
            for i in range(cond.output_len):
                self.curve_names.append(f"Condition {counter}.{i}")
        self.curve_names.append("Total loss")

    def reset(self, new_save_dir: str = None):
        """Reset the curve's state and optionally change save directory.

        Args:
            new_save_dir (str, optional): New directory for saving curves
        """
        super().reset(new_save_dir)
        self.metric_history = []


class GridResidualCurve(BaseCallbackWithGrad, BasicCurve):
    """Callback for plotting residual values on a grid during training.

    Calculates and visualizes residuals at grid points for a specific condition.
    Requires gradient computation for accurate residual calculation.
    """

    def __init__(
        self,
        save_dir: str,
        period: int,
        grid: Union[Grid, GridWithGrad],
        save_mode: Literal["html", "png", "pt", "show"] = "html",
        condition_index: int = 0,
        style: dict = None,
    ):
        """Initialize the grid residual curve plotter.

        Args:
            save_dir (str): Directory for saving plots
            period (int): Plot update frequency in epochs
            grid (Union[Grid, GridWithGrad]): Grid for residual calculation
            save_mode (Literal["html", "png", "pt", "show"]): Output format
            condition_index (int): Index of condition to plot residuals for
            style (dict, optional): Custom plotting style options
        """
        style = style if style is not None else {}
        default_style = {
            "layout_yaxis_type": "log",
            "layout_title": f"Grid Residual {condition_index}",
            "layout_xaxis_title": "Epoch",
            "layout_yaxis_title": "Residual",
        }
        BasicCurve.__init__(
            self,
            save_dir + "/grid_residual",
            period,
            f"grid_residual_condition_{condition_index}",
            save_mode=save_mode,
            style=default_style | style,
        )

        if isinstance(grid, GridWithGrad):
            self.grid = grid
        elif isinstance(grid, Grid):
            self.grid = GridWithGrad.from_Grid(grid)
        else:
            raise ValueError("Please provide an instance of Grid or GridWithGrad")
        self.condition_index = condition_index

    def __call__(self, trainer: Trainer) -> None:
        """Update and plot grid residuals during training.

        Args:
            trainer (Trainer): Current trainer instance
        """
        loss_arr = []
        cond = trainer.pinn.conditions[self.condition_index]
        loss_arr += trainer.pinn.calculate_loss_on_points(cond, self.grid.points)
        self.metric_history.append(torch.stack(loss_arr, dim=0).detach().cpu())

        if trainer.current_epoch == 0:
            for i in range(cond.output_len):
                self.curve_names.append(f"Condition {self.condition_index}.{i}")
        elif trainer.current_epoch % self.period == 0:
            loss_arr_tensor = torch.stack(self.metric_history, dim=1)
            BasicCurve.draw(
                self,
                loss_arr_tensor,
                self.curve_names,
                np.arange(trainer.current_epoch + 1),
            )


class LossCurve(BaseCallback, BasicCurve):
    """Callback for plotting training loss curves.

    Creates and updates plots showing the evolution of training losses over epochs.
    Supports multiple loss components and total loss visualization.
    """

    def __init__(
        self,
        save_dir: str,
        period: int,
        save_mode: Literal["html", "png", "pt", "show"] = "html",
        style: dict = None,
    ):
        """Initialize the loss curve plotter.

        Args:
            save_dir (str): Directory for saving plots
            period (int): Plot update frequency in epochs
            save_mode (Literal["html", "png", "pt", "show"]): Output format
            style (dict, optional): Custom plotting style options
        """
        style = style if style is not None else {}
        default_style = {
            "layout_yaxis_type": "log",
            "layout_title": "Loss curve",
            "layout_xaxis_title": "Epoch",
            "layout_yaxis_title": "Loss",
        }
        BasicCurve.__init__(
            self,
            save_dir + "/loss_curves",
            period,
            "loss_curve",
            save_mode=save_mode,
            style=default_style | style,
        )

    def __call__(self, trainer: Trainer) -> None:
        """Update and plot loss curves during training.

        Args:
            trainer (Trainer): Current trainer instance
        """
        self.metric_history.append(trainer.epoch_loss_detailed.detach().cpu())
        if trainer.current_epoch == 0:
            self.init_curve_names(trainer.pinn.conditions)
        elif trainer.current_epoch % self.period == 0:
            loss_arr_tensor = torch.stack(self.metric_history, dim=1)
            loss_arr_tensor = torch.cat(
                (loss_arr_tensor, torch.sum(loss_arr_tensor, dim=0, keepdim=True))
            )
            BasicCurve.draw(
                self,
                loss_arr_tensor,
                self.curve_names,
                np.arange(trainer.current_epoch + 1),
            )


class LearningRateCurve(BaseCallback, BasicCurve):
    """Callback for plotting learning rate changes during training.

    Visualizes the evolution of learning rate over epochs, useful for
    monitoring learning rate schedules and decay.
    """

    def __init__(
        self,
        save_dir: str,
        period: int,
        save_mode: Literal["html", "png", "pt", "show"] = "html",
        style: dict = None,
    ):
        """Initialize the learning rate curve plotter.

        Args:
            save_dir (str): Directory for saving plots
            period (int): Plot update frequency in epochs
            save_mode (Literal["html", "png", "pt", "show"]): Output format
            style (dict, optional): Custom plotting style options
        """
        style = style if style is not None else {}
        default_style = {
            "layout_yaxis_type": "log",
            "layout_title": "Learning rate",
            "layout_xaxis_title": "Epoch",
            "layout_yaxis_title": "Learning Rate",
        }
        BasicCurve.__init__(
            self,
            save_dir + "/lr_curves",
            period,
            "lr_curve",
            save_mode=save_mode,
            style=default_style | style,
        )

    def __call__(self, trainer: Trainer) -> None:
        """Update and plot learning rate curve during training.

        Args:
            trainer (Trainer): Current trainer instance
        """
        self.metric_history.append(trainer.current_lr)
        if trainer.current_epoch % self.period == 0 and trainer.current_epoch != 0:
            BasicCurve.draw(
                self,
                [self.metric_history],
                ["learning rate"],
                np.arange(trainer.current_epoch + 1),
            )


class ErrorCurve(BaseCallback, BasicCurve):
    """Callback for plotting error metrics during training.

    Creates and updates plots showing the evolution of error metrics comparing
    model predictions to known solutions across different outputs and conditions.

    Note:
        The provided solution function must return outputs for all components
        being predicted by the model.
    """

    def __init__(
        self,
        save_dir: str,
        period: int,
        full_solution: Callable[[torch.Tensor], torch.Tensor],
        save_mode: Literal["html", "png", "pt", "show"] = "html",
        style: dict = None,
    ):
        """Initialize the error curve plotter.

        Args:
            save_dir (str): Directory for saving plots
            period (int): Plot update frequency in epochs
            full_solution (Callable[[torch.Tensor], torch.Tensor]): Function providing ground truth
            save_mode (Literal["html", "png", "pt", "show"]): Output format
            style (dict, optional): Custom plotting style options
        """
        style = style if style is not None else {}
        default_style = {
            "layout_yaxis_type": "log",
            "layout_title": "Error",
            "layout_xaxis_title": "Epoch",
            "layout_yaxis_title": "Error",
        }
        BasicCurve.__init__(
            self,
            save_dir + "/error_curves",
            period,
            "error_curve",
            save_mode=save_mode,
            style=default_style | style,
        )
        self.solution = full_solution

    def __call__(self, trainer: Trainer):
        """Update and plot error curves during training.

        Args:
            trainer (Trainer): Current trainer instance
        """
        error = []
        for condition in trainer.pinn.conditions:
            points = condition.points
            predicted = trainer.pinn.model(points).detach().cpu()
            exact = self.solution(points).detach().cpu()
            error.append((predicted - exact).abs().mean(dim=0))
        error = torch.concat(error)
        self.metric_history.append(error)

        if trainer.current_epoch == 0:
            for cond in range(len(trainer.pinn.conditions)):
                for out in range(predicted.shape[1]):
                    self.curve_names.append(f"Output {out} on Condition {cond}")
        elif trainer.current_epoch % self.period == 0:
            error_tensor = torch.stack(self.metric_history, dim=1)
            BasicCurve.draw(
                self,
                error_tensor,
                self.curve_names,
                np.arange(trainer.current_epoch + 1),
            )


class MeshErrorCurve(BaseCallback, BasicCurve):
    """A callback that creates and plots error curves during training using predefined mesh points.

    This class calculates the absolute mean error between model predictions and ground truth values
    on a fixed set of mesh points, and visualizes the error progression over training epochs.

    Args:
        save_dir (str): Directory path where the error curve plots will be saved.
        period (int): Number of epochs between plot updates.
        points (torch.Tensor): Input mesh points of shape (num_points, input_dim).
        values (torch.Tensor): Ground truth values at mesh points of shape (num_points, output_dim).
        save_mode (Literal["html", "png", "pt", "show"]): Format for saving plots. Defaults to "html".
        style (dict, optional): Custom styling options for the plots. Defaults to None.

    Attributes:
        points (torch.Tensor): Stored mesh points for error calculation.
        values (torch.Tensor): Stored ground truth values for error calculation.
        curve_names (list): Names of the output components for plotting.
        metric_history (list): History of error values over epochs.
    """

    def __init__(
        self,
        save_dir: str,
        period: int,
        points: torch.Tensor,
        values: torch.Tensor,
        save_mode: Literal["html", "png", "pt", "show"] = "html",
        style: dict = None,
    ):
        style = style if style is not None else {}
        default_style = {
            "layout_yaxis_type": "log",
            "layout_title": "Mesh Error",
            "layout_xaxis_title": "Epoch",
            "layout_yaxis_title": "Error",
        }

        BasicCurve.__init__(
            self,
            save_dir + "/error_curves",
            period,
            "mesh_error_curve",
            save_mode,
            default_style | style,
        )
        assert points.shape[0] == values.shape[0]
        self.points = points
        self.values = values

    @classmethod
    def from_file(
        cls,
        save_dir: str,
        period: int,
        file_points: str,
        file_values,
        save_mode: Literal["html", "png", "pt", "show"] = "html",
        style: dict = None,
    ):
        """Creates a MeshErrorCurve instance by loading mesh points and values from files.

        Args:
            save_dir (str): Directory path where the error curve plots will be saved.
            period (int): Number of epochs between plot updates.
            file_points (str): Path to CSV file containing mesh points.
            file_values (str): Path to CSV file containing ground truth values.
            save_mode (Literal["html", "png", "pt", "show"]): Format for saving plots. Defaults to "html".
            style (dict, optional): Custom styling options for the plots. Defaults to None.

        Returns:
            MeshErrorCurve: A new instance initialized with the loaded data.
        """
        points = torch.Tensor(
            np.genfromtxt(file_points, delimiter=",", dtype="float32")
        )  # default device
        values = torch.from_numpy(
            np.genfromtxt(file_values, delimiter=",", dtype="float32")
        )  # always CPU
        return cls(save_dir, period, points, values, save_mode, style)

    def __call__(self, trainer: Trainer):
        predicted = trainer.pinn.model(self.points).detach().cpu()
        error = (predicted - self.values).abs().mean(dim=0)
        self.metric_history.append(error)

        if trainer.current_epoch == 0:
            for out in range(predicted.shape[1]):
                self.curve_names.append(f"Output {out}")
        elif trainer.current_epoch % self.period == 0:
            error_tensor = torch.stack(self.metric_history, dim=1)
            BasicCurve.draw(
                self,
                error_tensor,
                self.curve_names,
                np.arange(trainer.current_epoch + 1),
            )


class GridErrorCurve(MeshErrorCurve):
    """A specialized version of MeshErrorCurve that works with grid points and analytical solutions.

    This class extends MeshErrorCurve to work specifically with grid-based points and calculates
    errors using a provided analytical solution function instead of pre-computed values.

    Args:
        save_dir (str): Directory path where the error curve plots will be saved.
        period (int): Number of epochs between plot updates.
        grid (Grid): Grid object containing the evaluation points.
        full_solution (Callable[[torch.Tensor], torch.Tensor]): Function that computes analytical solution
            values for given grid points. Should take input of shape (num_points, input_dim) and return
            tensor of shape (num_points, output_dim).
        save_mode (Literal["html", "png", "pt", "show"]): Format for saving plots. Defaults to "html".
        style (dict, optional): Custom styling options for the plots. Defaults to None.

    Note:
        The analytical solution function must return values of shape (batch_size, output_dim)
        to maintain compatibility with the error calculation process.
    """

    def __init__(
        self,
        save_dir: str,
        period: int,
        grid: Grid,
        full_solution: Callable[[torch.Tensor], torch.Tensor],
        save_mode: Literal["html", "png", "pt", "show"] = "html",
        style: dict = None,
    ):
        style = style if style is not None else {}
        default_style = {
            "layout_yaxis_type": "log",
            "layout_title": "Grid Error",
            "layout_xaxis_title": "Epoch",
            "layout_yaxis_title": "Error",
        }

        points = grid.points
        values = full_solution(points).cpu()

        assert len(values.shape) == 2  # values.shape == (bs, output_dim,)
        MeshErrorCurve.__init__(
            self, save_dir, period, points, values, save_mode, default_style | style
        )
        self.save_name = "grid_error_curve"  # override
