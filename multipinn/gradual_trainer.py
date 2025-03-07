from __future__ import annotations

from typing import Callable, List

import torch
import torch.distributed as dist

from .PINN import PINN
from .trainer import Trainer


class GradualTrainer:
    def __init__(
        self,
        trainer: Trainer,
        param_setter: Callable,
        list_of_params: List[float],
        g_l_schedule: List[int],
    ) -> None:
        self.trainer = trainer
        self.param_setter = param_setter
        self.list_of_params = list_of_params
        self.g_l_schedule = g_l_schedule
        assert len(self.list_of_params) == len(self.g_l_schedule)
        self.g_l_stage = 0

    def train(self):
        print("Start of gradual learning")
        for self.trainer.current_epoch in range(self.trainer.num_epochs + 1):
            if (
                self.g_l_stage < len(self.g_l_schedule)
                and self.trainer.current_epoch == self.g_l_schedule[self.g_l_stage]
            ):
                new_param = self.list_of_params[self.g_l_stage]
                self.param_setter(new_param)
                print(f"New parameter value: {new_param}")

                self.g_l_stage += 1

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
