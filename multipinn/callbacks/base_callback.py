from abc import ABC, abstractmethod


class BaseCallback(ABC):
    """Abstract base class for callbacks that can be executed during model training.

    This class defines the interface for callbacks that don't require gradient computation.
    Callbacks are useful for implementing functionality that should be executed at specific
    points during training, such as logging, visualization, or model checkpointing.

    All concrete callback classes must implement the __call__ method.
    """

    @abstractmethod
    def __call__(self, trainer: "Trainer") -> None:
        """Execute the callback.

        Args:
            trainer (Trainer): The trainer instance that is executing this callback.
                             Provides access to the model, optimizer, and training state.

        Raises:
            NotImplementedError: Must be implemented by concrete subclasses.
        """
        raise NotImplementedError


class BaseCallbackWithGrad(BaseCallback, ABC):
    """Abstract base class for callbacks that require gradient computation.

    This class extends BaseCallback for cases where the callback needs to access or
    manipulate gradients during training. Examples include gradient clipping, gradient
    logging, or adaptive learning methods based on gradient information.

    All concrete callback classes must implement the __call__ method.
    """

    @abstractmethod
    def __call__(self, trainer: "Trainer") -> None:
        """Execute the callback with gradient computation enabled.

        Args:
            trainer (Trainer): The trainer instance that is executing this callback.
                             Provides access to the model, optimizer, gradients and training state.

        Raises:
            NotImplementedError: Must be implemented by concrete subclasses.
        """
        raise NotImplementedError
