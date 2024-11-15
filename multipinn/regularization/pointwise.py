import itertools
from typing import Any, List

import torch

from .basic import BasicLosses


class PointwiseLosses(BasicLosses):
    """
    Class with regularization Investigating and Mitigating Failure Modes in Physicsinformed Neural Networks (PINNs)
    github authors //github.com/shamsbasir/investigating_mitigating_failure_modes_in_pinns/blob/main/Helmholtz/Helmholtz.ipynb

    Args:
        ignored_indexes (List[int]): list indexes equations without regularization.
    """

    def __init__(
        self,
        ignored_indexes: List[int] = None,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        lr: float = 0.01,
    ):
        super().__init__()
        self.optimizer_def = optimizer
        self.optimizers = []
        self.lr = lr
        self.models = []
        self.ignored_indexes = ignored_indexes if ignored_indexes is not None else []
        self.ignored_mask = None

    def __reset_lambdas(self, trainer):
        """
        Reset lambda parameters if conditions for update are met.

        Args:
            trainer: The PINN trainer instance containing current epoch and batch information.
        """
        if (
            trainer.current_epoch % trainer.update_grid_every == 0
            and trainer.current_batch == 0
        ):
            self.ignored_mask = None
            self.models = []

    def __call__(self, trainer):
        """
        Calculate losses with pointwise regularization.

        Args:
            trainer: The PINN trainer instance.

        Returns:
            tuple: A tuple containing:
                - total_loss (torch.Tensor): The regularized sum of all losses
                - real_losses (torch.Tensor): Original unregularized losses
        """
        self.__reset_lambdas(trainer)
        self.__update_params(trainer)
        real_losses = self.__generate_real_loss(trainer)
        return self.__calculate_losses_with_regularization(trainer), real_losses

    def __update_params(self, trainer):
        """
        Update or initialize the regularization parameters.

        Args:
            trainer: The PINN trainer instance.
        """
        if len(self.models) < trainer.num_batches:
            self.models.append([])
            self.__first_launch(trainer)
        else:
            self.__update_lambdas(trainer)

    def __generate_real_loss(self, trainer):
        """
        Calculate the original unregularized losses.

        Args:
            trainer: The PINN trainer instance.

        Returns:
            torch.Tensor: Detached tensor of original losses.
        """
        return torch.stack(trainer.pinn.calculate_loss(), dim=0).detach()

    def __calculate_losses_with_regularization(self, trainer):
        """
        Calculate the total loss with pointwise regularization applied.

        Args:
            trainer: The PINN trainer instance.

        Returns:
            torch.Tensor: Sum of regularized losses.
        """
        losses = []
        index = 0
        for condition in trainer.pinn.conditions:
            list_residuals = condition.get_residual(trainer.pinn.model)
            for residual in list_residuals:
                if self.ignored_mask[index]:
                    losses.append((residual**2).sum())
                else:
                    losses.append(
                        self.models[trainer.current_batch][index](residual**2)[0]
                    )
                index += 1
        return torch.sum(torch.stack(losses, dim=0))

    def __first_launch(self, trainer):
        """
        Initialize models and optimizers for the first time.

        Creates linear models for each non-ignored residual and sets up their optimizers.

        Args:
            trainer: The PINN trainer instance.
        """
        index = 0
        self.ignored_mask = []
        params = []
        for condition in trainer.pinn.conditions:
            list_residuals = condition.get_residual(trainer.pinn.model)
            for residual in list_residuals:
                residual.detach()
                if index in self.ignored_indexes:
                    self.ignored_mask.append(True)
                    self.models[trainer.current_batch].append(None)
                else:
                    self.ignored_mask.append(False)
                    model = torch.nn.Linear(residual.shape[0], 1, bias=False)
                    model.weight.data.fill_(1.0)
                    self.models[trainer.current_batch].append(model)
                    params.append(model.parameters())
                index += 1
        self.optimizers.append(
            self.optimizer_def(
                params=itertools.chain(*params), lr=self.lr, maximize=True
            )
        )

    def __update_lambdas(self, trainer):
        """
        Update lambda parameters using the optimizer.

        Args:
            trainer: The PINN trainer instance.
        """
        self.optimizers[trainer.current_batch].step()
        self.optimizers[trainer.current_batch].zero_grad()
