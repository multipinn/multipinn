from typing import List

import torch

from .basic import BasicLosses


class ConstantLosses(BasicLosses):
    """
    Class with regularization with constant lambdas.

    Args:
        lambdas (List[float]): constant lambdas.
    """

    def __init__(self, lambdas=List[float]):
        super().__init__()
        self.lambdas = lambdas

    """
    Multiply losses on constant lambdas and return new losses.
    Requirement: First loss must relate to inner area.
    """

    def __call__(self, trainer):
        losses = trainer.pinn.calculate_loss()
        if len(losses) != len(self.lambdas):
            raise Exception(
                f"The number of lambdas(number = {len(self.lambdas)}) does not match the number of losses(number = {len(losses)})"
            )

        for ind, los in enumerate(losses):
            losses[ind] = los * self.lambdas[ind]
        losses = torch.stack(losses)
        total_loss = torch.sum(losses)
        return total_loss, losses
