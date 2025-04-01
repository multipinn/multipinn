import copy
from typing import List

import torch

from multipinn import *
from multipinn.condition import *
from multipinn.geometry import *

from .trainer import Trainer


class MarchingTrainer:
    def __init__(
        self,
        steps: List[float],
        trainer: Trainer,
        epochs_per_iter: List[int],
        divide,
    ) -> None:
        self.steps = steps
        self.trainer = trainer
        self.epochs_per_iter = epochs_per_iter
        self.divide = divide

        self.current_epoch = 0
        self.previous_iter_epoch = 0
        self.common_lr = trainer.optimizer.param_groups[0]["lr"]

        self.previous_model: torch.nn.Module = self.trainer.pinn.model
        self.not_trained_model: torch.nn.Module = copy.deepcopy(self.trainer.pinn.model)
        self.heatmap_callback = None
        self.stage = 0

    def reset_weights(self, model):
        def init_weights(m):
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                m.reset_parameters()

        model.apply(init_weights)

    def march_trainer(self):
        first_step = True
        for step, next_step in zip(self.steps[0:-1], self.steps[1:]):
            self.divide(
                self.trainer.pinn.conditions,
                step,
                next_step,
                first_step,
                self.previous_model,
            )
            print(f"Start marching on subdomain: {step}, {next_step}")

            if not first_step:
                for param_group in self.trainer.optimizer.param_groups:
                    param_group["lr"] = self.common_lr

            self.train()
            self.previous_model = copy.deepcopy(self.trainer.pinn.model)

            self.trainer.pinn.model.eval()
            self.trainer.pinn.model.train()
            self.reset_weights(self.trainer.pinn.model)

            first_step = False
            self.stage += 1

    def train(self):
        self.trainer.pinn.update_data()
        for self.trainer.current_epoch in range(
            self.previous_iter_epoch, self.epochs_per_iter[self.stage] + 1
        ):
            self.trainer._train_epoch()
            if self.trainer.scheduler is not None:
                self.trainer.scheduler.step()
                self.trainer.current_lr = self.trainer.optimizer.param_groups[0]["lr"]

            self.trainer.pinn.model.eval()
            with torch.no_grad():
                for callback in self.trainer.callbacks_organizer.base_callbacks:
                    callback(self.trainer)

            self.trainer.pinn.model.train()
            for callback in self.trainer.callbacks_organizer.grad_callbacks:
                callback(self.trainer)

            self.previous_iter_epoch = self.trainer.current_epoch + 1
