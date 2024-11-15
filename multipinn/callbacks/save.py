from pathlib import Path
from typing import Literal, Union

import torch

from .base_callback import BaseCallback


class FileSaver:
    """Base class for handling file saving operations.

    Provides core functionality for managing save directories and creating
    required directory structures. Handles path management using pathlib
    for cross-platform compatibility.

    Attributes:
        path (Path): Path to the save directory
    """

    def __init__(self, save_dir: str) -> None:
        """Initialize the file saver.

        Args:
            save_dir (str): Directory path where files will be saved
        """
        self.path = Path(save_dir)

    def reset(self, new_save_dir: str = None) -> None:
        """Reset the save directory and optionally change it.

        Creates the new directory if it doesn't exist.

        Args:
            new_save_dir (str, optional): New directory path. If None, keeps current path.
        """
        if new_save_dir is not None:
            self.path = Path(new_save_dir)
            self.path.mkdir(parents=True, exist_ok=True)

    def mkdir(self) -> None:
        """Create the save directory if it doesn't exist.

        Creates parent directories as needed and handles existing directories gracefully.
        """
        self.path.mkdir(parents=True, exist_ok=True)


class SaveModel(FileSaver, BaseCallback):
    """Callback for saving model checkpoints during training.

    Periodically saves model state to disk, allowing for training resumption
    or model deployment. Supports configurable save frequency and file extensions.

    Example:
        >>> saver = SaveModel("./models", ".pth", 100)
        >>> # Will save model every 100 epochs to ./models/checkpoints/
    """

    def __init__(self, save_dir: str, extension: str = ".pth", period: int = 1000):
        """Initialize the model saver.

        Args:
            save_dir (str): Base directory for saving models
            extension (str, optional): File extension for saved models. Defaults to ".pth".
            period (int, optional): How often to save models (in epochs). Defaults to 1000.
        """
        super().__init__(save_dir + "/checkpoints")
        self.period = period
        self.extension = extension

    def __call__(self, trainer: "Trainer") -> None:
        """Save model if current epoch matches save period.

        Args:
            trainer (Trainer): Current trainer instance containing model to save
        """
        if trainer.current_epoch % self.period == 0 and trainer.current_epoch != 0:
            save_path = self.path / Path(f"{trainer.current_epoch}{self.extension}")
            torch.save(trainer.pinn.model, save_path)


class BaseImageSave(FileSaver):
    """Base class for saving visualization outputs.

    Provides functionality for saving plots and figures in various formats.
    Supports multiple output formats including HTML, PNG, PyTorch tensors,
    and direct display.

    Attributes:
        period (int): How often to save images
        save_mode (str): Output format - "html", "png", "pt", or "show"

    Example:
        >>> saver = BaseImageSave(100, "./plots", "png")
        >>> # Will save plots as PNG files every 100 epochs
    """

    def __init__(
        self,
        period: int,
        save_dir: str = None,
        save_mode: Literal["html", "png", "pt", "show"] = "html",
    ):
        """Initialize the image saver.

        Args:
            period (int): How often to save images (in epochs)
            save_dir (str, optional): Directory for saving images. Not required if save_mode is "show".
            save_mode (Literal["html", "png", "pt", "show"], optional): Output format. Defaults to "html".
        """
        self.period = period
        self.save_mode = save_mode

        if self.save_mode != "show" and save_dir is not None:
            super().__init__(save_dir)

    def save_fig(self, fig, file_name: str) -> None:
        """Save a figure in the specified format.

        Args:
            fig: Figure object to save (typically a plotly figure)
            file_name (str): Base name for the saved file (without extension)

        Raises:
            Exception: If save_mode is unknown
            AssertionError: If file_name or path is missing when required
        """
        assert (self.save_mode == "show") or (
            file_name is not None and self.path is not None
        ), "File name and path required for non-display modes"

        if self.save_mode == "png":
            fig.write_image(self.path / Path(file_name + ".png"), scale=2)
        elif self.save_mode == "html":
            fig.write_html(self.path / Path(file_name + ".html"))
        elif self.save_mode == "pt":
            self.save_pt(fig, self.path / Path(file_name + ".pt"))
        elif self.save_mode == "show":
            fig.show()
        else:
            raise Exception(f"Unknown save_mode = {self.save_mode}")

    def save_pt(self, fig, file: Union[str, Path]) -> None:
        """Save figure data as a PyTorch tensor.

        Extracts data from the figure using the dict_data method and saves
        it as a PyTorch tensor file.

        Args:
            fig: Figure object containing data to save
            file (Union[str, Path]): Path for saved tensor file

        Raises:
            AssertionError: If dict_data method is not implemented
        """
        assert hasattr(
            self, "dict_data"
        ), "dict_data method must be implemented for PT saving"
        data = self.dict_data(fig)
        torch.save(data, file)
