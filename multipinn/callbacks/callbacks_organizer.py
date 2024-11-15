from typing import List

from .base_callback import BaseCallback, BaseCallbackWithGrad
from .save import FileSaver


class CallbacksOrganizer:
    """Organizes and manages training callbacks by separating them into gradient and non-gradient groups.

    This class provides a structured way to handle different types of callbacks during model training.
    It automatically categorizes callbacks into two groups:
    - Base callbacks: Regular callbacks that don't require gradient computation
    - Gradient callbacks: Callbacks that need to access or manipulate gradients

    It also handles initialization of directory structure for callbacks that save files.

    Attributes:
        base_callbacks (List[BaseCallback]): List of callbacks that don't require gradients
        grad_callbacks (List[BaseCallbackWithGrad]): List of callbacks that require gradient computation

    Example:
        >>> callbacks = [LoggingCallback(), GradientClippingCallback(), ModelCheckpoint()]
        >>> organizer = CallbacksOrganizer(callbacks)
        >>> # Now callbacks are organized and directories are created if needed
        >>> len(organizer.base_callbacks)  # Number of regular callbacks
        >>> len(organizer.grad_callbacks)  # Number of gradient-based callbacks
    """

    def __init__(self, callbacks: List[BaseCallback], mkdir: bool = True):
        """Initialize the callbacks organizer.

        Args:
            callbacks (List[BaseCallback]): List of callback instances to organize.
                Can include both regular callbacks and gradient-based callbacks.
            mkdir (bool, optional): Whether to create directories for FileSaver callbacks.
                Defaults to True. Set to False if directories should be created manually.
        """
        self.base_callbacks = []
        self.grad_callbacks = []
        for callback in callbacks:
            if isinstance(callback, BaseCallbackWithGrad):
                self.grad_callbacks.append(callback)
            else:
                self.base_callbacks.append(callback)
            if isinstance(callback, FileSaver) and mkdir:
                callback.mkdir()
