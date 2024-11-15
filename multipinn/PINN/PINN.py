from typing import List

import torch

from multipinn.condition.condition import Condition


class PINN:
    def __init__(
        self,
        model: torch.nn.Module,
        conditions: List[Condition],
    ) -> None:
        """Initialize a Physics-Informed Neural Network (PINN).

        Args:
            model (torch.nn.Module): Neural network model to be used as the function approximator
            conditions (List[Condition]): List of physical conditions (equations, boundary conditions, etc.)
                that the network needs to satisfy

        The PINN class manages the interaction between the neural network and the physical conditions,
        handling loss calculations and data updates during training.
        """
        self.model = model
        self.conditions = conditions

        self.model.eval()
        for cond in self.conditions:
            cond.init_output_len(self.model)

    def update_data(self) -> None:
        """Update training points for all conditions.

        Generates new training points for each condition and updates their values.
        This is typically called periodically during training to sample new points
        from the domain.
        """
        for cond in self.conditions:
            cond.update_points(model=self.model)

    def select_batch(self, i) -> None:
        """Select the i-th batch of points for all conditions.

        Args:
            i (int): Batch index to select
        """
        for cond in self.conditions:
            cond.select_batch(i)

    def condition_loss(self, cond: Condition) -> List[torch.Tensor]:
        """Calculate squared residuals for a given condition.

        Args:
            cond (Condition): Physical condition to evaluate

        Returns:
            List[torch.Tensor]: List of squared residuals for each component of the condition
        """
        return [r**2 for r in cond.get_residual(self.model)]

    @staticmethod
    def count(loss_arr):
        """Count number of points in each loss component.

        Args:
            loss_arr: List of loss tensors

        Returns:
            List[int]: Number of points for each loss component
        """
        return [len(x) for x in loss_arr]

    @staticmethod
    def mean_loss(loss_arr):
        """Calculate mean of each loss component.

        Args:
            loss_arr: List of loss tensors

        Returns:
            List[torch.Tensor]: Mean value for each loss component
        """
        return [torch.mean(x) for x in loss_arr]

    @staticmethod
    def sum_loss(loss_arr):
        """Calculate sum of each loss component.

        Args:
            loss_arr: List of loss tensors

        Returns:
            List[torch.Tensor]: Sum for each loss component
        """
        return [torch.sum(x) for x in loss_arr]

    def calculate_loss(self) -> List[torch.Tensor]:
        """Calculate mean loss for each condition component.

        Returns:
            List[torch.Tensor]: List of mean losses for each component of each condition
        """
        mean_losses = []
        for cond in self.conditions:
            loss = self.condition_loss(cond)
            mean_losses += self.mean_loss(loss)
        return mean_losses

    def calculate_mean_loss(self):
        """Calculate total mean loss and individual condition means.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                - Total mean loss across all conditions
                - List of mean losses for each condition component
        """
        sum_losses = []
        counts = []
        for cond in self.conditions:
            loss = self.condition_loss(cond)
            sum_losses += self.sum_loss(loss)
            counts += self.count(loss)
        total_mean = torch.sum(torch.stack(sum_losses)) / sum(counts)
        mean_losses = [s / c for s, c in zip(sum_losses, counts)]
        return total_mean, mean_losses

    def calculate_loss_and_count(self):
        """Calculate mean loss and point count for each condition component.

        Returns:
            Tuple[List[torch.Tensor], List[int]]:
                - List of mean losses for each condition component
                - List of point counts for each component
        """
        mean_losses = []
        counts = []
        for cond in self.conditions:
            loss = self.condition_loss(cond)
            mean_losses += self.mean_loss(loss)
            counts += self.count(loss)
        return mean_losses, counts

    def calculate_loss_on_points(
        self, cond: Condition, points: torch.Tensor
    ) -> List[torch.Tensor]:
        """Calculate residual losses on specific points for a given condition.

        Args:
            cond (Condition): Physical condition to evaluate
            points (torch.Tensor): Points at which to evaluate the residuals

        Returns:
            List[torch.Tensor]: Mean squared residuals for each component of the condition
        """
        get_residual_fn = cond.get_residual_fn(self.model)
        list_of_residuals = get_residual_fn(points)
        losses = [torch.mean(r**2) for r in list_of_residuals]
        return losses
