from typing import Callable, List, Union

import torch
import torch.distributed as dist

from multipinn.PINN import PINN


class Trainer:
    def __init__(
        self,
        pinn: PINN,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        num_epochs: int = 1000,
        num_batches: int = 1,
        update_grid_every: int = 1,
        calc_loss: Union[str, Callable[[], List[torch.Tensor]]] = "mean",
        callbacks_organizer: "CallbacksOrganizer" = None,
        mixed_training: bool = False,
    ) -> None:
        """Initialize a trainer for Physics-Informed Neural Networks.

        Args:
            pinn (PINN): The PINN model to train
            optimizer (torch.optim.Optimizer): Optimizer for model parameters
            scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler
            num_epochs (int, optional): Number of training epochs. Defaults to 1000.
            num_batches (int, optional): Number of batches per epoch. Defaults to 1.
            update_grid_every (int, optional): Frequency of grid point updates. Defaults to 1.
            calc_loss (Union[str, Callable], optional): Loss calculation method. Can be "mean", "legacy",
                or a custom function. Defaults to "mean".
            callbacks_organizer (CallbacksOrganizer, optional): Organizer for training callbacks.
                Defaults to None.
            mixed_training (bool, optional): Whether to use mixed precision training. Only works with CUDA.
                Defaults to False.
        """
        self.pinn = pinn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.update_grid_every = (
            update_grid_every if update_grid_every is not None else num_epochs + 1
        )
        if isinstance(calc_loss, str):
            if calc_loss == "mean":
                self.calc_loss = self.mean_calc_loss
            elif calc_loss == "legacy":
                self.calc_loss = self.sum_of_means_calc_loss
            else:
                raise ValueError
        else:
            self.calc_loss = calc_loss
        self.callbacks_organizer = callbacks_organizer
        self.mixed_training = mixed_training and torch.cuda.is_available()

        self.current_epoch = 0
        self.current_batch = 0
        self.inum_batches = 1.0 / self.num_batches
        self.epoch_loss_detailed = None
        self.total_loss = None
        self.current_lr = None

        if self.num_batches == 1 and type(self) is Trainer:
            print(
                "Optimisation opportunity - you can use TrainerOneBatch instead of Train, since num_batches==1"
            )
        for cond in self.pinn.conditions:
            cond.set_batching(self.num_batches)
        if self.mixed_training:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)

    def train(self):
        """Execute the main training loop.

        Runs the training process for the specified number of epochs. For each epoch:
        1. Performs training step
        2. Updates learning rate via scheduler
        3. Executes callbacks for monitoring and visualization
        """
        for self.current_epoch in range(self.num_epochs + 1):
            self._train_epoch()

            if self.scheduler is not None:
                self.scheduler.step()
                self.current_lr = self.optimizer.param_groups[0]["lr"]

            self.pinn.model.eval()
            with torch.no_grad():
                for callback in self.callbacks_organizer.base_callbacks:
                    callback(self)

            self.pinn.model.train()
            for callback in self.callbacks_organizer.grad_callbacks:
                callback(self)

    def _train_epoch(self):
        """Execute a single training epoch.

        Updates training points if needed, then performs forward and backward passes
        through the network. Handles both regular and mixed precision training.
        """
        self.pinn.model.train()

        if self.current_epoch % self.update_grid_every == 0:
            self.pinn.update_data()

        class Zero:  # TODO find better solution
            def __iadd__(self, other: torch.Tensor):
                self = other
                return self

        self.epoch_loss_detailed = Zero()
        self.total_loss = 0

        self._train_batches()

    def _train_batches(self):
        """Process all batches for the current epoch.

        Performs the following for each batch:
        1. Selects the batch data
        2. Calculates loss
        3. Performs backward pass
        4. Accumulates gradients

        Finally updates model parameters using accumulated gradients.
        """
        self.optimizer.zero_grad()  # out of the loop to accumulate gradients
        for self.current_batch in range(self.num_batches):
            self.pinn.select_batch(self.current_batch)
            batch_loss, losses = self.calc_loss(self)
            batch_loss.backward()
            self.epoch_loss_detailed += losses.detach()
            self.total_loss += batch_loss.detach()
        self.optimizer.step()  # out of the loop to accumulate gradients
        self.epoch_loss_detailed *= self.inum_batches
        self.total_loss *= self.inum_batches

    @staticmethod
    def sum_of_means_calc_loss(self: "Trainer"):
        """Calculate loss using sum of mean losses for each condition.

        Legacy loss calculation method that computes mean loss for each condition
        separately and then sums them.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Total loss and individual losses stack
        """
        losses = torch.stack(self.pinn.calculate_loss())
        total_loss = torch.sum(losses)
        return total_loss, losses

    @staticmethod
    def mean_calc_loss(self: "Trainer"):
        """Calculate loss using mean across all conditions.

        Preferred loss calculation method that computes a single mean across
        all condition points.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Total loss and individual losses stack
        """
        total_loss, losses_list = self.pinn.calculate_mean_loss()
        return total_loss, torch.stack(losses_list)


class TrainerMultiGPU(Trainer):
    def __init__(
        self,
        pinn: PINN,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        num_epochs: int = 1000,
        num_batches: int = 1,
        update_grid_every: int = 1,
        calc_loss: Union[str, Callable[[], List[torch.Tensor]]] = "mean",
        callbacks_organizer: "CallbacksOrganizer" = None,
        mixed_training: bool = False,
        rank: int = 0,
    ) -> None:
        """Initialize a multi-GPU trainer for distributed training.

        Extends the base Trainer for distributed training across multiple GPUs.

        Args:
            rank (int, optional): Rank of current process in distributed setup. Defaults to 0.
            **kwargs: All other arguments are passed to base Trainer
        """
        super().__init__(
            pinn,
            optimizer,
            scheduler,
            num_epochs,
            num_batches,
            update_grid_every,
            calc_loss,
            callbacks_organizer,
            mixed_training,
        )
        self.rank = rank
        assert dist.get_world_size() == num_batches

    def _train_batches(self):
        """Process batches in distributed training mode.

        Each GPU processes its assigned batch, then gradients are synchronized
        across all devices before parameter updates.
        """
        self.optimizer.zero_grad()

        self.pinn.select_batch(self.rank)
        batch_loss, losses = self.calc_loss(self)
        batch_loss.backward()
        self.epoch_loss_detailed += losses.detach()
        self.total_loss += batch_loss.detach()

        self.reduce_gradients()
        self.optimizer.step()

    def reduce_gradients(self):
        """Synchronize gradients across all GPUs.

        Performs all-reduce operation to average gradients across all processes
        in the distributed training setup.
        """
        size = float(dist.get_world_size())
        # t0 = clock()
        # s = 0
        for param in self.pinn.model.parameters():
            # s += param.grad.data.nelement() * param.grad.data.element_size()
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size
        # t1 = clock()
        # print(f'{os.environ["RANK"]}: reducing {s} bytes in {i+1} calls took {t1-t0:.3f} sec')


class TrainerOneBatch(Trainer):
    def __init__(
        self,
        pinn: PINN,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        num_epochs: int = 1000,
        update_grid_every: int = 1,
        calc_loss: Union[str, Callable[[], List[torch.Tensor]]] = "mean",
        callbacks_organizer: "CallbacksOrganizer" = None,
        mixed_training: bool = False,
    ) -> None:
        """Initialize a single-batch trainer.

        Optimized version of Trainer for cases where only one batch is used.

        Args:
            Same as base Trainer, except num_batches which is fixed to 1
        """
        super().__init__(
            pinn,
            optimizer,
            scheduler,
            num_epochs,
            1,
            update_grid_every,
            calc_loss,
            callbacks_organizer,
            mixed_training,
        )

    def _train_batches(self):
        """Process single batch training step.

        Simplified training loop optimized for single batch training.
        """
        self.pinn.select_batch(0)
        self.optimizer.zero_grad()
        batch_loss, losses = self.calc_loss(self)
        batch_loss.backward()
        self.epoch_loss_detailed += losses.detach()
        self.total_loss += batch_loss.detach()
        self.optimizer.step()
