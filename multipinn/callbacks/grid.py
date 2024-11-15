from typing import Sequence, Union

import numpy as np
import torch

from ..PINN import PINN


class Grid:
    """A regular grid for generating heatmap plots and evaluating solutions.

    Creates a structured grid of points in N-dimensional space for visualization
    and evaluation purposes. The grid can be used for plotting heatmaps, computing
    solution values at regular intervals, and analyzing model behavior across the domain.

    Attributes:
        coord (List[np.ndarray]): Grid coordinates along each dimension as numpy arrays
        points (torch.Tensor): Flattened grid points as a tensor

    Example:
        >>> grid = Grid([0, 0], [1, 1], [10, 10])
        >>> # Creates a 10x10 grid in the unit square
        >>> grid.points.shape
        torch.Size([100, 2])
    """

    def __init__(
        self, low: np.array, high: np.array, n_points: Union[Sequence[int], int]
    ):
        """Initialize a regular grid.

        Args:
            low (np.array): Lower bounds for each dimension
            high (np.array): Upper bounds for each dimension
            n_points (Union[Sequence[int], int]): Number of points per dimension.
                If an integer is provided, points are distributed proportionally
                across dimensions using smart_volume.

        Raises:
            AssertionError: If dimensions of low and high don't match or
                if n_points sequence length doesn't match dimension count
        """
        assert len(low) == len(high)
        if isinstance(n_points, int):
            n_points = self.smart_volume(low, high, n_points)
        assert len(low) == len(n_points)

        lin = []
        for dim in range(len(low)):
            lin.append(torch.linspace(low[dim], high[dim], n_points[dim]))

        self.coord = torch.meshgrid(*lin, indexing="ij")
        coord = torch.stack(self.coord)
        coord = torch.movedim(coord, 0, -1)

        # Flatten grid points to 2D tensor (num_points, dim)
        self.points = torch.reshape(coord, (-1, len(low)))

        # Convert coordinates to numpy for visualization
        self.coord = [array.cpu().numpy() for array in self.coord]

    @classmethod
    def from_pinn(cls, pinn: PINN, n_points: Union[Sequence[int], int]) -> "Grid":
        """Create a grid encompassing the domain of a PINN model.

        Creates a grid that covers the union of all geometry bounding boxes
        defined in the PINN's conditions.

        Args:
            pinn (PINN): PINN model containing geometry information
            n_points (Union[Sequence[int], int]): Number of points per dimension

        Returns:
            Grid: A grid covering the entire PINN domain
        """
        low, high = [], []
        for cond in pinn.conditions:
            geom = cond.geometry.bbox
            low.append(geom[0])
            high.append(geom[1])
        low = np.min(np.array(low), axis=0)
        high = np.max(np.array(high), axis=0)
        return cls(low, high, n_points)

    @classmethod
    def from_condition(cls, condition, n_points: Union[Sequence[int], int]) -> "Grid":
        """Create a grid covering a single condition's geometry.

        Args:
            condition: Condition containing geometry information
            n_points (Union[Sequence[int], int]): Number of points per dimension

        Returns:
            Grid: A grid covering the condition's domain
        """
        geom = condition.geometry
        low = geom.bbox[0]
        high = geom.bbox[1]
        return cls(low, high, n_points)

    @staticmethod
    def smart_volume(low, high, total_points: int) -> Sequence[int]:
        """Distribute points across dimensions proportionally to domain size.

        Computes an optimal distribution of points across dimensions based on the
        relative size of each dimension's range, ensuring that the total number
        of points approximately matches the requested amount.

        Args:
            low: Lower bounds for each dimension
            high: Upper bounds for each dimension
            total_points (int): Target total number of grid points

        Returns:
            Sequence[int]: Number of points to use in each dimension

        Note:
            The actual total number of points may differ slightly from the
            requested amount due to rounding to integer values.
        """
        low = np.array(low)
        high = np.array(high)
        size = high - low
        pos_size = size[size > 0]
        volume = np.prod(pos_size)
        n = len(pos_size)
        k = (total_points / volume) ** (1 / n)
        n_points = np.round(size * k).astype(int)
        n_points[n_points <= 0] = 1
        return n_points


class GridWithGrad(Grid):
    """A grid that supports gradient computation through its points.

    Extends the base Grid class by making the points tensor require gradients,
    which is necessary for computing derivatives or optimizing point locations.
    This is particularly useful for gradient-based analysis or optimization.

    Example:
        >>> grad_grid = GridWithGrad([0, 0], [1, 1], [10, 10])
        >>> grad_grid.points.requires_grad
        True
    """

    def __init__(
        self, low: np.array, high: np.array, n_points: Union[Sequence[int], int]
    ):
        """Initialize a grid with gradient-enabled points.

        Args:
            low (np.array): Lower bounds for each dimension
            high (np.array): Upper bounds for each dimension
            n_points (Union[Sequence[int], int]): Number of points per dimension
        """
        Grid.__init__(self, low, high, n_points)
        self.points.requires_grad = True

    @classmethod
    def from_Grid(cls, grid: Grid) -> "GridWithGrad":
        """Convert a regular Grid to a GridWithGrad.

        Creates a new GridWithGrad instance from an existing Grid,
        enabling gradient computation for the points.

        Args:
            grid (Grid): Existing grid to convert

        Returns:
            GridWithGrad: New grid with gradient computation enabled

        Raises:
            ValueError: If input is not a Grid instance
        """
        if isinstance(grid, Grid):
            result = cls.__new__(cls)
            result.points = grid.points.clone()
            result.coord = grid.coord.copy()
            result.points.requires_grad = True
            return result
        else:
            raise ValueError("Please provide an instance of Grid.")
