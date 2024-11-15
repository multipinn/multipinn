from typing import Any, Iterable, Union

import torch

from multipinn.condition.condition import Condition


class Generator:
    """A class for generating and managing training points in Physics-Informed Neural Networks (PINNs).

    The Generator class is responsible for creating sets of points that will be used during
    the training process of PINNs. It supports both pseudo-random and quasi-random sampling
    strategies and can be associated with one or multiple boundary conditions.

    Args:
        n_points (int): Number of points to generate for each condition.
        sampler (str, optional): Sampling strategy to use. Options include:

            - "pseudo": Pseudo-random sampling
            - "LHS": Latin Hypercube Sampling
            - "Halton": Halton sequence
            - "Hammersley": Hammersley sequence
            - "Sobol": Sobol sequence
            Defaults to "pseudo".

    Attributes:
        n_points (int): Number of points to generate.
        sampler (str): Selected sampling strategy.
    """

    def __init__(self, n_points, sampler="pseudo"):
        self.n_points = n_points
        self.sampler = sampler

    def generate(self, condition: Condition, model) -> Any:
        """Generates points within the geometry specified by the condition.

        Creates a set of points using the specified sampling strategy within
        the condition's geometry. The generated points are converted to PyTorch tensors
        with gradient tracking enabled.

        Args:
            condition (Condition): The boundary condition associated with these points.
            model: The neural network model (unused in base implementation).

        Returns:
            torch.Tensor: Generated points as a tensor with shape (n_points, dimension)
                and gradients enabled.
        """
        numpy_points = condition.geometry.random_points(self.n_points, self.sampler)
        torch_points = torch.tensor(numpy_points, requires_grad=True)
        return torch_points

    def use_for(self, condition: Union[Condition, Iterable[Condition]]) -> None:
        """Associates this generator with one or more conditions.

        Links this generator to specified conditions, allowing it to generate
        points for those conditions during training.

        Args:
            condition (Union[Condition, Iterable[Condition]]): A single condition or
                an iterable of conditions to associate with this generator.

        Example:
            >>> generator.use_for(boundary_condition)  # Single condition
            >>> generator.use_for([bc1, bc2, bc3])    # Multiple conditions
        """
        if isinstance(condition, Condition):
            condition.generator = self
        else:
            for cond in condition:
                cond.generator = self
