import torch

from .basic import BasicLosses


class GradientLosses(BasicLosses):
    """
    Class with regularization UNDERSTANDING AND MITIGATING GRADIENT FLOW PATHOLOGIES IN PHYSICS-INFORMED NEURAL NETWORKS
    github authors https://github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs/tree/master

    Args:
        alpha (float): hyperparametr, authors recommended value = 0.9.
    """

    def __init__(self, alpha: float = 0.9):
        super().__init__()
        self.lambda_regularization = None
        self.alpha = alpha

    """
    Calculate regularization param and return losses multiply on it.
    Requirement: First loss must relate to inner area.
    """

    def __call__(self, trainer):
        losses = trainer.pinn.calculate_loss()
        losses_true = losses.copy()
        if len(trainer.pinn.conditions) != len(losses):
            raise Exception("Regularization does not supported system ODE")

        if self.lambda_regularization is None:
            self.lambda_regularization = torch.ones((len(losses) - 1))
        losses[0].backward(retain_graph=True)
        grad_of_params = []
        for parameter in trainer.pinn.model.parameters():
            if parameter.grad is not None:
                grad_of_params.append(torch.max(torch.abs(parameter.grad)))
        max_f = max(grad_of_params)
        trainer.optimizer.zero_grad()

        for ind, los in enumerate(losses[1:]):
            los.backward(retain_graph=True)
            grad_of_params = []
            for parameter in trainer.pinn.model.parameters():
                if parameter.grad is not None:
                    grad_of_params.append(torch.abs(parameter.grad).mean())
            regularization = max_f / torch.tensor(grad_of_params).mean()
            trainer.optimizer.zero_grad()
            regularization = (1 - self.alpha) * self.lambda_regularization[
                ind
            ] + self.alpha * regularization
            losses[ind + 1] = los * regularization
            self.lambda_regularization[ind] = regularization
        return (
            torch.sum(torch.stack(losses, dim=0)),
            torch.stack(losses_true, dim=0).detach(),
        )
