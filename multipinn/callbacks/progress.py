from string import Formatter

import numpy as np
import tqdm

from ..trainer import Trainer
from .base_callback import BaseCallback


class ProgressBar(BaseCallback):
    def __init__(self, template: str, period: int = 10):
        self.template = template
        self.period = period
        np.set_printoptions(formatter={"float": "{:.2e}".format})
        self.params = {}
        for e in Formatter().parse(self.template):
            self.params[e[1]] = e[2]

    def __call__(self, trainer: Trainer) -> None:
        if trainer.current_epoch % self.period == 0:
            print(self.make_message(trainer))

    def make_message(self, trainer: Trainer) -> str:
        loss_detailed = trainer.epoch_loss_detailed.cpu().numpy()
        kwargs = {}
        if self.params.get("loss_eq") is not None:
            kwargs["loss_eq"] = self.split_by_cond(
                loss_detailed, trainer.pinn.conditions
            )
        if self.params.get("loss_cond") is not None:
            kwargs["loss_cond"] = self.sum_by_cond(
                loss_detailed, trainer.pinn.conditions
            )
        return self.template.format(
            epoch=trainer.current_epoch,
            lr=trainer.current_lr,
            total_loss=trainer.total_loss.cpu().numpy(),
            loss_detailed=loss_detailed,
            **kwargs,
        )

    @staticmethod
    def split_by_cond(data, conditions):
        result = ""
        i = 0
        for cond in conditions:
            loss = data[i : i + cond.output_len]
            result += f"{loss}"
            i += cond.output_len
        return result

    @staticmethod
    def sum_by_cond(data, conditions):
        result = "["
        i = 0
        for cond in conditions:
            loss = data[i : i + cond.output_len].sum()
            result += f"{loss:.2e} "
            i += cond.output_len
        result = result[:-1] + "]"
        return result


class TqdmBar(ProgressBar):
    def __init__(self, template: str, period: int = 5):
        super().__init__(template, period)
        self.tracker = None

    def __call__(self, trainer: Trainer) -> None:
        if trainer.current_epoch == 0:
            self.tracker = tqdm.tqdm(
                total=trainer.num_epochs, desc=self.make_message(trainer)
            )
        else:
            self.tracker.update()
            if self.tracker.n >= self.tracker.total:
                self.tracker.set_description(self.make_message(trainer))
                self.tracker.close()
            elif trainer.current_epoch % self.period == 0:
                self.tracker.set_description(self.make_message(trainer))
