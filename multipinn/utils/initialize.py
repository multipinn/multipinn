import logging
from typing import Callable, Optional, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict

logger = logging.getLogger(__name__)


def initialize_model(
    cfg: DictConfig, input_dim: int, output_dim: int
) -> torch.nn.Module:
    """
    Initialize a neural network model based on configuration settings.

    This function takes a configuration dictionary and dynamically sets the input
    and output dimensions of the model before instantiation.

    Args:
        cfg (DictConfig): Hydra configuration object containing model specifications.
        input_dim (int): Number of input features for the model.
        output_dim (int): Number of output features for the model.

    Returns:
        torch.nn.Module: Initialized PyTorch model according to the configuration.
    """
    with open_dict(cfg.model.params):
        cfg.model.params.input_dim = input_dim
        cfg.model.params.output_dim = output_dim

    logger.info(f"Model: {cfg.model.type}")
    logger.info(f"Model params: {cfg.model.params}")
    return instantiate(cfg.model_target, **cfg.model.params)


def initialize_regularization(cfg: DictConfig) -> Tuple[Optional[Callable], str]:
    """
    Initialize the loss regularization function based on configuration settings.

    This function sets up the regularization scheme for training. If no regularization
    is specified, it falls back to the trainer's default loss calculation method.

    Args:
        cfg (DictConfig): Hydra configuration object containing regularization specifications.

    Returns:
        Tuple[Optional[Callable], str]: A tuple containing:
            - calc_loss (Optional[Callable]): The loss calculation function or None
            - loss_type (str): Description of the loss calculation method
    """
    if cfg.regularization.type != "None" and cfg.regularization.type is not None:
        logger.info(f"Regularization type: {cfg.regularization.type}")
        logger.info(f"Regularization parameters: {cfg.regularization.params}")
        calc_loss = instantiate(cfg.regularization_target, **cfg.regularization.params)

    else:
        logger.info(f"Using calc_closs={cfg.trainer.calc_loss}")
        calc_loss = cfg.trainer.calc_loss  # "mean" or "legacy"

    return calc_loss
