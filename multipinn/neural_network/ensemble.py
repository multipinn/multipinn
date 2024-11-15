from typing import List, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ExponentialLR, LRScheduler

from multipinn import PINN, Generator, Trainer
from multipinn.condition import Condition

from .activation_function import GELU, Sine


class PretrainedEnsemble(nn.Module):
    """
    An ensemble of pretrained models for solving differential equations.
    """

    def __init__(self, layers_all: list, pretrained_models: List[nn.Module]):
        """
        Args:
            layers_all (list): List of layer sizes including input and output dimensions
            pretrained_models (List[nn.Module]): List of pretrained neural network models
        """
        super().__init__()
        self.pretrained_models = pretrained_models
        layers = layers_all[1:-1]

        linears = [
            nn.Sequential(nn.Linear(layers_all[0], layers[0], bias=False), Sine())
        ]

        for layer, next_layer in zip(layers[:-1], layers[1:]):
            linears.append(
                nn.Sequential(nn.Linear(layer, next_layer, bias=False), GELU())
            )

        linears.append(nn.Linear(layers[-1], layers_all[-1], bias=False))

        self.linears = nn.ModuleList(linears)

    def _freeze_model(self, trained_model):
        for param in trained_model.parameters():
            param.requires_grad = False

    def forward(self, input: torch.Tensor):
        predict_models = 0

        for trained_model in self.pretrained_models[:-1]:
            self._freeze_model(trained_model)
            predict_models += trained_model(input)

        t = torch.clone(predict_models)

        for layer in self.linears:
            t = layer(t)
        return t


class TrainableEnsemble(nn.Module):
    """
    An ensemble of trainable models for solving differential equations.
    """

    def __init__(self, train_model_list: List[nn.Module]):
        """
        Args:
            train_model_list (List[nn.Module]): List of neural network models to be trained
        """
        super(TrainableEnsemble, self).__init__()
        self.train_model_list = train_model_list
        for idx_model in range(len(self.train_model_list)):
            self.train_model_list[idx_model].fc = nn.Identity()
        self.fc1 = nn.LazyLinear(1)

    def forward(self, input: torch.Tensor):
        prediction_list = []
        for model in self.train_model_list:
            prediction_list.append(model(input))
        concat_prediction = torch.cat(prediction_list, dim=1)
        return self.fc1(concat_prediction)


class EnsembleInstance:
    """
    A container class that holds configuration for a single model in the ensemble.
    """

    def __init__(
        self, model_name: str, pinn: PINN, optimizer: Optimizer, scheduler: LRScheduler
    ) -> None:
        """
        Args:
            model_name (str): Name identifier for the model
            pinn (PINN): Physics-informed neural network instance
            optimizer (Optimizer): Optimizer for model training
            scheduler (LRScheduler): Learning rate scheduler
        """
        self.model_name = model_name
        self.pinn = pinn
        self.optimizer = optimizer
        self.scheduler = scheduler

    def get_parameters(self) -> Tuple[str, PINN, Optimizer, LRScheduler]:
        return self.model_name, self.pinn, self.optimizer, self.scheduler


class EnsembleTrainer(Trainer):
    """
    Trainer class for handling the training of multiple models in an ensemble.
    """

    def __init__(
        self, ensemble_config: List[EnsembleInstance], output_dim: int, **kwargs
    ) -> None:
        """
        Args:
            ensemble_config (List[EnsembleInstance]): List of ensemble model configurations
            output_dim (int): Output dimension of the ensemble model
            **kwargs: Additional arguments passed to parent Trainer class
        """
        super().__init__(pinn=None, optimizer=None, scheduler=None, **kwargs)
        self.ensemble_config = ensemble_config
        self.output_dim = output_dim
        self.pretrained_models = []

    def train(self):
        for ensemble_instance in self.ensemble_config:
            (
                model_name,
                self.pinn,
                self.optimizer,
                self.scheduler,
            ) = ensemble_instance.get_parameters()

            print(f"---- training {model_name} ----")
            self.callbacks_organizer.reset_callbacks(model_name)
            super().train()

            self.pretrained_models.append(self.pinn.model)

        torch.cuda.empty_cache()

        model_name = "Ensemble"
        self.pinn.model = PretrainedEnsemble(
            [self.output_dim, 64, self.output_dim], self.pretrained_models
        )
        self.optimizer = torch.optim.Adam(self.pinn.model.parameters(), lr=1e-4)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.999)

        print(f"---- training {model_name} ----")
        self.callbacks_organizer.reset_callbacks(model_name)
        super().train()


def ensemble_builder(
    models: List[torch.nn.Module],
    generatorss_domain: List[Generator],
    generators_bound: List[Generator],
    condition_idx: List[int],
    optimizers: List[Optimizer],
    schedulers: List[LRScheduler],
    conditions: List[Condition],
) -> List[EnsembleInstance]:
    """
    Builds ensemble configuration from individual model components.

    Args:
        models (List[torch.nn.Module]): List of neural network models
        generatorss_domain (List[Generator]): List of domain data generators
        generators_bound (List[Generator]): List of boundary data generators
        condition_idx (List[int]): List of condition indices
        optimizers (List[Optimizer]): List of optimizers for each model
        schedulers (List[LRScheduler]): List of learning rate schedulers
        conditions (List[Condition]): List of physical conditions

    Returns:
        List[EnsembleInstance]: List of configured ensemble instances
    """
    ensemble_config = []

    for i, ensemble_part in enumerate(
        zip(models, generatorss_domain, generators_bound, optimizers, schedulers)
    ):
        (model, generator_domain, generator_bound, optimizer, scheduler) = ensemble_part
        model_name = model.__class__.__name__ + f"_{i+1}"
        generator_bound.use_for(conditions)
        for j in condition_idx:
            generator_domain.use_for(conditions[j])

        pinn = PINN(model=model, conditions=conditions)
        ensemble_config.append(EnsembleInstance(model_name, pinn, optimizer, scheduler))

    return ensemble_config
