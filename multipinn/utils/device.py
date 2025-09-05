import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_device_and_seed(seed, accelerator=None, gpu_id=0):
    """
    Set both the random seed and device configuration for PyTorch.

    Args:
        seed (int): Random seed for reproducibility.
        accelerator (str, optional): Device type to use (e.g., 'cuda', 'cpu').
            If None, will use CUDA if available. Defaults to None.
        gpu_id (int, optional): GPU device ID to use when multiple GPUs are available.
            Defaults to 0.
    """
    set_seed(seed)
    set_device(accelerator, gpu_id)


def set_device(accelerator=None, gpu_id=0):
    """
    Configure the PyTorch device for computation.

    Sets up the device for PyTorch operations, either using the specified accelerator
    or automatically selecting CUDA if available, falling back to CPU if not.
    When using CUDA device 0, also sets the default tensor type to CUDA.

    Args:
        accelerator (str, optional): Device type to use (e.g., 'cuda', 'cpu').
            If None, will use CUDA if available. Defaults to None.
        gpu_id (int, optional): GPU device ID to use when multiple GPUs are available.
            Defaults to 0.

    Returns:
        None
    """
    if accelerator is not None:
        device = torch.device(accelerator)
    else:
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    logger.info(f"Device you are using is {device}")

    if device.type == "cuda":
        torch.cuda.set_device(device) 
        torch.set_default_device(device)
        torch.set_default_dtype(torch.float32)


def set_seed(seed):
    """
    Set random seeds for reproducibility.

    Sets the random seed for both PyTorch and NumPy to ensure
    reproducible results across runs.

    Args:
        seed (int): The random seed value to use.

    Returns:
        None
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
