import torch

from .basic import BasicLosses


class AdaptiveConditionsLosses(BasicLosses):
    """
    Implements adaptive loss weighting for multiple loss terms in PINN training.

    This class automatically adjusts the weights of different loss components during training
    using a gradient-based optimization approach. The weights are learned by maximizing
    the weighted sum of losses, encouraging balanced training across different conditions.

    Args:
        lr (float, optional): Learning rate for the weight optimizer. Defaults to 1e-3.

    Attributes:
        lr (float): Learning rate for the weight optimizer
        optimizer (torch.optim.Optimizer): Optimizer for training the weights
        train_weight (torch.Tensor): Trainable log-weights before softmax
        weight (torch.Tensor): Actual weights after applying softmax to train_weight
    """

    def __init__(
        self,
        # optimizer: torch.optim.Optimizer = torch.optim.Adam,
        lr: float = 1e-3,
    ):
        super().__init__()
        # self.optimizer_cls = optimizer
        self.lr = lr
        self.optimizer = None
        self.train_weight = None
        self.weight = None

    def __call__(self, trainer: "Trainer"):
        """
        Calculate the weighted sum of losses using adaptive weights.

        Args:
            trainer (Trainer): The PINN trainer instance containing the model and data

        Returns:
            tuple: (total_loss, individual_losses) where:
                - total_loss (torch.Tensor): Weighted sum of all losses
                - individual_losses (torch.Tensor): Individual loss terms before weighting
        """
        if self.optimizer is None:
            return self.__first_launch(trainer)
        # if trainer.current_epoch % 500 == 0:
        #     print(self.weight)
        return self.__calc_losses(trainer)

    def __calc_losses(self, trainer: "Trainer"):
        """
        Calculate weighted losses for the current iteration.

        Args:
            trainer (Trainer): The PINN trainer instance

        Returns:
            tuple: (total_loss, individual_losses)
        """
        self.__update_weight()
        mean_losses = trainer.pinn.calculate_loss()
        return self.__weight_sum(mean_losses)

    def __first_launch(self, trainer):
        """
        Initialize the adaptive weights on the first call.

        The initial weights are set based on the log ratio of the number of points
        in each condition to encourage balanced training.

        Args:
            trainer (Trainer): The PINN trainer instance

        Returns:
            tuple: (total_loss, individual_losses)
        """
        mean_losses, counts = trainer.pinn.calculate_loss_and_count()
        self.train_weight = torch.log(
            torch.Tensor(counts) / sum(counts)
        ).requires_grad_()
        self.optimizer = torch.optim.Adam(
            params=(self.train_weight,), lr=self.lr, maximize=True, weight_decay=1e-2
        )
        # self.optimizer = self.optimizer_cls(params=self.train_weight, lr=self.lr, maximize=True)
        return self.__weight_sum(mean_losses)

    def __weight_sum(self, mean_losses):
        """
        Calculate the weighted sum of losses using softmax-normalized weights.

        Args:
            mean_losses (List[torch.Tensor]): List of individual loss terms

        Returns:
            tuple: (total_loss, individual_losses)
        """
        self.weight = torch.softmax(self.train_weight, dim=0)
        total_loss = torch.sum(torch.stack(mean_losses) * self.weight)
        return total_loss, torch.stack(mean_losses).detach()

    def __update_weight(self):
        """
        Update the adaptive weights using the optimizer.
        """
        self.optimizer.step()
        self.optimizer.zero_grad()
