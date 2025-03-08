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
        save_dir,
        steps: List[float],
        trainer: Trainer,
        epochs_per_iter: List[int],
        basic_symbols,
    ) -> None:
        self.save_dir = save_dir + "/marching"
        self.steps = steps
        self.trainer = trainer
        self.epochs_per_iter = epochs_per_iter
        self.basic_symbols = basic_symbols

        self.current_epoch = 0
        self.previous_iter_epoch = 0
        self.common_lr = trainer.optimizer.param_groups[0]["lr"]

        self.previous_model: torch.nn.Module = self.trainer.pinn.model
        self.not_trained_model: torch.nn.Module = copy.deepcopy(self.trainer.pinn.model)
        self.heatmap_callback = None
        self.stage = 0

    def ic_new(self, model, arg):
        _, u, _, _ = self.basic_symbols(model, arg)
        _, u_prev, _, _ = self.basic_symbols(self.previous_model, arg)
        return [u - u_prev]

    def divide(self, step, next_step, first_iter=False):
        ## conditions for covection_1D
        new_domain = Hypercube(low=[step, 0], high=[next_step, torch.pi * 2])
        x_min = Hypercube(low=[step, 0], high=[next_step, 0])
        t_min = Hypercube(low=[step, 0], high=[step, torch.pi * 2])

        if first_iter:
            self.trainer.pinn.conditions[0].geometry = new_domain
            self.trainer.pinn.conditions[1].geometry = x_min
            self.trainer.pinn.conditions[2].geometry = t_min

        else:
            self.trainer.pinn.conditions[0].geometry = new_domain
            self.trainer.pinn.conditions[1].geometry = x_min
            self.trainer.pinn.conditions[2].geometry = t_min
            self.trainer.pinn.conditions[2].function = self.ic_new

        ## conditions for heat_2D_1C
        # new_domain = Hypercube(low=[0, step], high=[2, next_step])
        # x_min = Hypercube(low=[0, step], high=[0, next_step])
        # x_max = Hypercube(low=[2, step], high=[2, next_step])
        # t_min = Hypercube(low=[0, step], high=[2, step])

        # if first_iter:
        #     self.trainer.pinn.conditions[0].geometry = new_domain
        #     self.trainer.pinn.conditions[1].geometry = x_min
        #     self.trainer.pinn.conditions[2].geometry = x_max
        #     self.trainer.pinn.conditions[3].geometry = t_min

        # else:
        #     self.trainer.pinn.conditions[0].geometry = new_domain
        #     self.trainer.pinn.conditions[1].geometry = x_min
        #     self.trainer.pinn.conditions[2].geometry = x_max
        #     self.trainer.pinn.conditions[3].geometry = t_min
        #     self.trainer.pinn.conditions[3].function = self.ic_new

        self.trainer.pinn.conditions[0].points = None

    def reset_weights(self, model):
        def init_weights(m):
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                m.reset_parameters()

        model.apply(init_weights)

    def march_trainer(self):
        first_step = True
        for step, next_step in zip(self.steps[0:-1], self.steps[1:]):
            self.divide(step, next_step, first_step)
            print(f"Start marching on subdomain: {step}, {next_step}")

            if not first_step:
                for param_group in self.trainer.optimizer.param_groups:
                    param_group["lr"] = self.common_lr

            self.train()
            self.previous_model = copy.deepcopy(self.trainer.pinn.model)

            self.trainer.pinn.model.eval()
            with torch.no_grad():
                torch.save(
                    self.trainer.pinn.model,
                    self.save_dir + f"/{self.trainer.current_epoch}_saved_model.pth",
                )
                self.heatmap_callback(self.trainer)

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

            self.previous_iter_epoch = self.trainer.current_epoch
