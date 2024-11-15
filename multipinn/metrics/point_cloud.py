from dataclasses import dataclass
from typing import Callable, List, Tuple, Union

import numpy as np
import torch

from ..mesh.comsol_reader import read_comsol_file
from .base_metric import BaseMetric
from .errors import norm_error


@dataclass
class PointCloud:
    """A container for point cloud data used in model evaluation.

    Stores coordinates, corresponding field values, and optional time values for both
    stationary and time-dependent problems.

    Attributes:
        points (Union[torch.Tensor, List[torch.Tensor]]): For stationary problems, a single tensor
            of point coordinates with shape (n_points, n_dims). For time-dependent problems,
            a list of tensors, each with shape (n_points, n_dims) for different time steps.
        values (torch.Tensor): Field values at each point. For stationary problems, shape is
            (n_points, n_fields). For time-dependent problems, shape is (n_points, n_timesteps).
        t_values (Union[torch.Tensor, None]): Time values for time-dependent problems.
            Shape is (n_timesteps,). None for stationary problems.
    """

    points: Union[torch.Tensor, List[torch.Tensor]]
    values: torch.Tensor
    t_values: Union[torch.Tensor, None] = None


class PointCloudMetric(BaseMetric):
    """A metric for evaluating model predictions against point cloud data.

    Computes error metrics between model predictions and known values at specified points
    in space (and time for time-dependent problems).

    Args:
        point_cloud (PointCloud): The reference point cloud data
        metric (Callable, optional): Error metric function. Defaults to norm_error
        field_names (List[str], optional): Names for each field being measured.
            If None, auto-generated names will be used.

    Example:
        >>> points = torch.tensor([[0,0], [1,1]])
        >>> values = torch.tensor([[1.0], [2.0]])
        >>> cloud = PointCloud(points, values)
        >>> metric = PointCloudMetric(cloud)
        >>> error = metric(model)

    Attributes:
        point_cloud (PointCloud): The reference point cloud data
        metric (Callable): The error metric function
        is_stationary (bool): Whether the problem is stationary or time-dependent
        field_names (List[str]): Names of the fields being measured
    """

    def __init__(
        self,
        point_cloud: PointCloud,
        metric: Callable = norm_error,
        field_names: List[str] = None,
    ) -> None:
        """Initialize the point cloud metric.

        Args:
            point_cloud (PointCloud): The reference point cloud data
            metric (Callable, optional): Error metric function. Defaults to norm_error
            field_names (List[str], optional): Names for each field. Defaults to None
        """
        self.point_cloud = point_cloud
        self.metric = metric
        self.is_stationary = self.point_cloud.t_values is None

        if field_names is None:
            if self.is_stationary:
                self.field_names = [
                    f"Field {i}" for i in range(self.point_cloud.values.shape[-1])
                ]
            else:
                self.field_names = [
                    f"Time {i}" for i in range(len(self.point_cloud.t_values))
                ]
        else:
            self.field_names = field_names

    @property
    def num_fields(self):
        """Number of fields being measured.

        Returns:
            int: The number of fields
        """
        return len(self.field_names)

    def __call__(self, model: torch.nn.Module) -> Union[List[float], List[List[float]]]:
        if self.is_stationary:
            return self._compute_stationary_metric(model)
        else:
            return self._compute_time_dependent_metric(model)

    def _compute_stationary_metric(self, model: torch.nn.Module) -> List[float]:
        """Compute metrics for stationary problems.

        Args:
            model (torch.nn.Module): The model to evaluate

        Returns:
            List[float]: List of metric values for each field
        """
        with torch.no_grad():
            predicted = model(self.point_cloud.points).detach()

        return [
            self.metric(self.point_cloud.values[:, i], predicted[:, i]).item()
            for i in range(predicted.shape[1])
        ]

    def _compute_time_dependent_metric(
        self, model: torch.nn.Module
    ) -> List[List[float]]:
        """Compute metrics for time-dependent problems.

        Args:
            model (torch.nn.Module): The model to evaluate

        Returns:
            List[List[float]]: List of metric values for each time step
        """
        metrics = []
        for i, points in enumerate(self.point_cloud.points):
            with torch.no_grad():
                predicted = model(points).detach()

            metric_values = self.metric(
                self.point_cloud.values[:, i], predicted[:, 0]
            ).item()

            metrics.append(metric_values)

        return metrics

    @classmethod
    def from_files(
        cls,
        points_file: str,
        values_file: str,
        metric: Callable = norm_error,
        field_names: List[str] = None,
    ):
        """Create a PointCloudMetric from CSV files.

        Args:
            points_file (str): Path to CSV file containing point coordinates
            values_file (str): Path to CSV file containing field values
            metric (Callable, optional): Error metric function. Defaults to norm_error
            field_names (List[str], optional): Names for each field. Defaults to None

        Returns:
            PointCloudMetric: A new metric instance
        """
        points = torch.tensor(
            np.genfromtxt(points_file, delimiter=",", skip_header=1, dtype="float32")
        )
        values = torch.tensor(
            np.genfromtxt(values_file, delimiter=",", skip_header=1, dtype="float32")
        )

        return cls(PointCloud(points, values), metric, field_names=field_names)

    @classmethod
    def from_comsol_file(
        cls,
        filepath: str,
        metric: Callable = norm_error,
        field_names: List[str] = None,
        is_stationary: bool = None,
    ):
        """Create a PointCloudMetric from a COMSOL export file.

        Args:
            filepath (str): Path to the COMSOL export file
            metric (Callable, optional): Error metric function. Defaults to norm_error
            field_names (List[str], optional): Names for each field. Defaults to None
            is_stationary (bool, optional): Whether the problem is stationary.
                If None, will be determined from the file. Defaults to None

        Returns:
            PointCloudMetric: A new metric instance
        """
        t_values, points, values = read_comsol_file(filepath, is_stationary)

        if t_values is None:  # Stationary problem
            points = torch.tensor(points, dtype=torch.float32)
            values = torch.tensor(values, dtype=torch.float32)
            point_cloud = PointCloud(points, values)
        else:  # Time-dependent problem
            points = [torch.tensor(p, dtype=torch.float32) for p in points]
            values = torch.tensor(values, dtype=torch.float32)
            t_values = torch.tensor(t_values, dtype=torch.float32)
            point_cloud = PointCloud(points, values, t_values)

        return cls(point_cloud, metric, field_names)
