from abc import ABC, abstractmethod


class BasicLosses(ABC):
    """
    Abstract base class for calculating loss with regularizations in PINN training.

    This class serves as a template for different loss weighting strategies used in
    Physics-Informed Neural Networks (PINNs). Subclasses implement specific weighting
    schemes such as constant weights or adaptive weights.

    All subclasses must implement the __call__ method which takes a trainer instance
    and returns the weighted losses.
    """

    def __init__(self):
        """Initialize the BasicLosses base class."""
        super().__init__()

    @abstractmethod
    def __call__(self, trainer):
        """
        Calculate the weighted sum of losses.

        Args:
            trainer: The PINN trainer instance containing the model and training data.

        Returns:
            tuple: A tuple containing:
                - total_loss (torch.Tensor): The weighted sum of all losses
                - individual_losses (torch.Tensor): Individual loss terms

        Raises:
            NotImplementedError: This is an abstract method that must be implemented
                by subclasses.
        """
        raise NotImplementedError
