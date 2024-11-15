import torch


class BaseMetric:
    """Base class for implementing training metrics.

    This class serves as an abstract base for creating custom metrics that can be
    evaluated during model training. Each metric should implement the __call__ method
    to compute specific measurements on the model's outputs.

    Args:
        metric (callable): The metric computation function
        num_fields (int): Number of fields/outputs this metric will measure
        field_names (List[str]): Names of the fields being measured

    Example:
        >>> class AccuracyMetric(BaseMetric):
        ...     def __init__(self):
        ...         super().__init__(accuracy_fn, 1, ['accuracy'])
        ...     def __call__(self, model):
        ...         return self.metric(model.predictions, model.targets)
    """

    def __init__(self, metric: callable, num_fields: int, field_names) -> None:
        """Initialize the base metric.

        Args:
            metric (callable): The metric computation function
            num_fields (int): Number of fields/outputs this metric will measure
            field_names: Names of the fields being measured
        """
        self.metric = metric
        self.num_fields = num_fields
        self.field_names = field_names

    def __call__(self, model: torch.nn.Module) -> torch.Tensor:
        """Compute the metric value for the given model.

        This method should be implemented by subclasses to define how the metric
        is calculated.

        Args:
            model (torch.nn.Module): The model to evaluate

        Returns:
            torch.Tensor: The computed metric value(s)

        Raises:
            NotImplementedError: If the subclass doesn't implement this method
        """
        raise NotImplementedError("Subclasses must implement __call__")
