from .generator import Condition, Generator


class GradBasedGenerator(Generator):
    """A generator that moves points along loss gradients to focus on high-error regions.

    This generator implements an adaptive sampling strategy that uses gradient information
    to move points towards regions where the model has higher loss. Points are shifted
    in the direction that would increase the loss, helping to focus training on
    challenging areas of the domain.

    Warning:
        This generator should not be used with Shell geometries as it may lead to
        undefined behavior or incorrect results.

    Args:
        n_points (int): Number of points to generate for each condition.
        sampler (str, optional): Sampling strategy for initial point generation.
            See base Generator class for available options. Defaults to "pseudo".
        anti_lr (float, optional): Learning rate for gradient-based point movement.
            Larger values result in larger point movements. Defaults to 1.

    Attributes:
        anti_lr (float): Step size for gradient-based updates.

    Notes:
        - On first call (when condition.points is None), generates points using the
          base Generator's strategy.
        - On subsequent calls, moves existing points using gradient information.
        - Points that would move outside the geometry are reset to their original
          positions.

    Example:
        >>> generator = GradBasedGenerator(n_points=1000, anti_lr=0.1)
        >>> generator.use_for(boundary_condition)
        >>> points = generator.generate(condition, model)
    """

    def __init__(self, n_points, sampler="pseudo", anti_lr=1):
        super().__init__(n_points, sampler)
        self.anti_lr = anti_lr

    def generate(self, condition: Condition, model):
        """Generates or updates points using gradient information.

        Args:
            condition (Condition): The boundary condition associated with these points.
            model: The neural network model (unused in this implementation).

        Returns:
            torch.Tensor: Generated or updated points as a tensor with shape
                (n_points, dimension) and gradients enabled.

        Notes:
            - If condition.points is None (first call), generates initial points
              using the parent class's generate method.
            - Otherwise, updates existing points by moving them in the direction
              of increasing loss.
            - Points that would move outside the geometry are reset to their
              previous positions.
        """
        if condition.points is None:
            return super().generate(condition, model)
        else:
            new_points = (
                condition.points + condition.points.grad * self.anti_lr
            ).detach()
            outside = ~condition.geometry.inside(new_points.cpu().numpy())
            new_points[outside] = condition.points[outside].detach()
            return new_points.requires_grad_()
