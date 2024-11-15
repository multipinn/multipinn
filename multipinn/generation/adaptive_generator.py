from typing import Callable, Optional

import numpy as np
import torch

from ..geometry.geometry import Geometry
from .generator import Condition, Generator


class AdaptiveGenerator(Generator):
    """
    Base class for adaptive point generators that adjust sampling based on error metrics.

    This generator creates points by first generating a dense set of candidate points,
    then selecting points based on a specified selection strategy.

    Args:
        n_points (int): Number of points to generate
        selection_strategy (Callable): Function that selects points based on error values
        sampler (str, optional): Sampling strategy ("pseudo", "LHS", etc.). Defaults to "pseudo"
        density_factor (int, optional): Factor to multiply n_points for candidate generation.
            Defaults to 8
    """

    def __init__(
        self,
        n_points: int,
        selection_strategy: Callable,
        sampler: str = "pseudo",
        density_factor: int = 8,
    ):
        super().__init__(n_points, sampler)
        self.selection_strategy = selection_strategy
        self.density_generator = Generator(n_points * density_factor, sampler)

    def generate(self, geometry: Geometry, condition: Condition, model) -> torch.Tensor:
        """Generate points using the adaptive strategy"""
        if condition.points is None:
            return super().generate(geometry, condition, model)

        # Generate candidate points and calculate errors
        candidates = self.density_generator.generate(geometry, condition, model)
        errors = self._calculate_errors(candidates, condition, model)

        # Select points based on strategy
        selected_indices = self.selection_strategy(errors, self.n_points)
        selected_points = candidates[selected_indices]

        return selected_points.detach().requires_grad_()

    def _calculate_errors(
        self, points: torch.Tensor, condition: Condition, model
    ) -> np.ndarray:
        """Calculate error values for given points"""
        residuals = condition.get_residual_fn(model)(points)
        residual = torch.stack(residuals, dim=1).abs().sum(dim=1)
        return residual.cpu().detach().numpy()


class RAD(AdaptiveGenerator):
    """
    Generates points by sampling according to error probability density.
    Points with higher error values have higher probability of being selected.

    Args:
        n_points (int): Number of points to generate
        power_coeff (float): Power coefficient for error scaling
        bias (float): Bias term added to scaled errors
        sampler (str, optional): Sampling strategy. Defaults to "pseudo"
        density_factor (int, optional): Density multiplication factor. Defaults to 8
    """

    def __init__(
        self,
        n_points: int,
        power_coeff: float = 3.0,
        bias: float = 1.0,
        sampler: str = "pseudo",
        density_factor: int = 8,
    ):
        def density_strategy(errors: np.ndarray, n: int) -> np.ndarray:
            # Scale errors and add bias
            scaled_errors = (
                np.power(errors, power_coeff) / np.mean(np.power(errors, power_coeff))
                + bias
            )
            probabilities = scaled_errors / np.sum(scaled_errors)

            # Sample points according to probability
            return np.random.choice(
                a=len(errors), size=n, replace=False, p=probabilities
            )

        super().__init__(n_points, density_strategy, sampler, density_factor)


class RAG(AdaptiveGenerator):
    """
    Generates points by selecting locations with maximum error values.
    Simply selects the n_points points with highest error values.

    Args:
        n_points (int): Number of points to generate
        sampler (str, optional): Sampling strategy. Defaults to "pseudo"
        density_factor (int, optional): Density multiplication factor. Defaults to 8
    """

    def __init__(self, n_points: int, sampler: str = "pseudo", density_factor: int = 8):
        def max_error_strategy(errors: np.ndarray, n: int) -> np.ndarray:
            return np.argpartition(errors, -n)[-n:]

        super().__init__(n_points, max_error_strategy, sampler, density_factor)


class RollingGenerator(AdaptiveGenerator):
    """
    Generator that maintains a rolling window of points, replacing only a portion
    of points in each iteration while keeping the rest.

    Args:
        n_points (int): Total number of points to maintain
        selection_strategy (Callable): Strategy to select new points
        roll_size (int, optional): Number of points to replace in each iteration.
            Defaults to n_points//32
        sampler (str, optional): Sampling strategy. Defaults to "pseudo"
        density_factor (int, optional): Density multiplication factor. Defaults to 1
    """

    def __init__(
        self,
        n_points: int,
        selection_strategy: Callable,
        roll_size: Optional[int] = None,
        sampler: str = "pseudo",
        density_factor: int = 1,
    ):
        super().__init__(n_points, selection_strategy, sampler, density_factor)
        self.roll_size = roll_size or max(1, n_points // 32)

    def generate(self, geometry: Geometry, condition: Condition, model) -> torch.Tensor:
        """Generate points using rolling window strategy"""
        if condition.points is None:
            return super().generate(geometry, condition, model)

        # Generate and select new points for replacement
        candidates = self.density_generator.generate(geometry, condition, model)
        errors = self._calculate_errors(candidates, condition, model)
        selected_indices = self.selection_strategy(errors, self.roll_size)
        new_points = candidates[selected_indices]

        # Combine with existing points
        points = torch.cat(
            [condition.points[self.roll_size :], new_points], dim=0
        ).detach()

        return points.requires_grad_()


class RAR_D(RollingGenerator):
    """
    Rolling window generator that selects new points based on error density.

    Args:
        n_points (int): Total number of points to maintain
        power_coeff (float): Power coefficient for error scaling
        bias (float): Bias term added to scaled errors
        roll_size (int, optional): Number of points to replace each iteration
        sampler (str, optional): Sampling strategy. Defaults to "pseudo"
        density_factor (int, optional): Density multiplication factor. Defaults to 1
    """

    def __init__(
        self,
        n_points: int,
        power_coeff: float = 3.0,
        bias: float = 1.0,
        roll_size: Optional[int] = None,
        sampler: str = "pseudo",
        density_factor: int = 1,
    ):
        def density_strategy(errors: np.ndarray, n: int) -> np.ndarray:
            scaled_errors = (
                np.power(errors, power_coeff) / np.mean(np.power(errors, power_coeff))
                + bias
            )
            probabilities = scaled_errors / np.sum(scaled_errors)
            return np.random.choice(len(errors), size=n, replace=False, p=probabilities)

        super().__init__(n_points, density_strategy, roll_size, sampler, density_factor)


class RAR_G(RollingGenerator):
    """
    Rolling window generator that selects new points with maximum error values.

    Args:
        n_points (int): Total number of points to maintain
        roll_size (int, optional): Number of points to replace each iteration
        sampler (str, optional): Sampling strategy. Defaults to "pseudo"
        density_factor (int, optional): Density multiplication factor. Defaults to 1
    """

    def __init__(
        self,
        n_points: int,
        roll_size: Optional[int] = None,
        sampler: str = "pseudo",
        density_factor: int = 1,
    ):
        def max_error_strategy(errors: np.ndarray, n: int) -> np.ndarray:
            return np.argpartition(errors, -n)[-n:]

        super().__init__(
            n_points, max_error_strategy, roll_size, sampler, density_factor
        )
