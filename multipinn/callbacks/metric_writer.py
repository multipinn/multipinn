import csv
import os
import time
from typing import List

from ..metrics import BaseMetric
from ..trainer import Trainer
from .base_callback import BaseCallback
from .save import FileSaver


class MetricWriter(BaseCallback, FileSaver):
    """A callback for recording training metrics to a CSV file.

    Records various training metrics including elapsed time, epoch number,
    total loss, custom metrics, and per-condition losses. Metrics are saved
    periodically to a CSV file for later analysis.

    Features:
        - Automatic CSV header generation based on metrics and conditions
        - Periodic saving with configurable frequency
        - Support for custom metrics
        - Time tracking and formatting
        - Automatic file management and cleanup

    Attributes:
        metrics (List[BaseMetric]): List of metric objects to evaluate
        filename (str): Name of the output CSV file
        period (int): How often to record metrics (in epochs)
        file_path (str): Full path to the output file
        is_inited_tabel (bool): Whether CSV headers have been written
        start_time (float): Training start time for elapsed time calculation

    Example:
        >>> metrics = [AccuracyMetric(), ErrorMetric()]
        >>> writer = MetricWriter(metrics, "./logs", "training_metrics.csv")
        >>> trainer.add_callback(writer)
    """

    def __init__(
        self,
        metrics: List[BaseMetric],
        save_dir: str,
        filename: str = "metrics.csv",
        period: int = 10,
    ) -> None:
        """Initialize the metric writer.

        Args:
            metrics (List[BaseMetric]): List of metric objects to evaluate
            save_dir (str): Directory where the CSV file will be saved
            filename (str, optional): Name of the CSV file. Defaults to "metrics.csv"
            period (int, optional): How often to record metrics (in epochs). Defaults to 10.
        """

        super().__init__(save_dir=save_dir)
        super().mkdir()

        self.metrics = metrics
        self.filename = filename
        self.period = period
        self.file_path = os.path.join(save_dir, filename)
        self.file = open(self.file_path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.is_inited_tabel = False
        self.start_time = time.time()

    def __del__(self):
        """Ensure the CSV file is properly closed when the object is destroyed."""
        self.file.close()

    def __call__(self, trainer: Trainer) -> None:
        """Record metrics at specified intervals during training.

        Args:
            trainer (Trainer): Current trainer instance
        """
        if trainer.current_epoch % self.period == 0:
            self.write_metrics(trainer)
            self.file.flush()

    def write_metrics(self, trainer: Trainer) -> None:
        """Write current metrics to the CSV file.

        Collects all metrics including elapsed time, epoch number, loss values,
        custom metrics, and condition-specific losses. Initializes the CSV headers
        on first call.

        Args:
            trainer (Trainer): Current trainer instance
        """
        metrics_values = []

        elapsed_seconds = time.time() - self.start_time
        elapsed_time = self.format_time(elapsed_seconds)

        for m in self.metrics:
            values = m(trainer.pinn.model)
            metrics_values.extend(values)

        if not self.is_inited_tabel:
            self.init_table(trainer.pinn.conditions)

        loss_detailed = trainer.epoch_loss_detailed.cpu().numpy()

        loss_by_cond = list(loss_detailed)
        self.writer.writerow(
            [
                elapsed_time,
                trainer.current_epoch,
                trainer.total_loss.cpu().numpy(),
                *metrics_values,
                *loss_by_cond,
            ]
        )

    def init_table(self, conditions):
        """Initialize the CSV file headers.

        Creates column headers for all tracked values including timestamps,
        epoch numbers, losses, custom metrics, and per-condition metrics.

        Args:
            conditions: Training conditions defining output dimensions
        """
        col_list = ["Time Elapsed", "Epoch", "Total Loss"]

        for m in self.metrics:
            for i in range(m.num_fields):
                col_list.append(m.metric.__name__ + f"_{m.field_names[i]}")

        all_conds = []

        for i, cond in enumerate(conditions):
            for j in range(cond.output_len):
                all_conds.append(f"{i}_{j}")

        col_list += all_conds
        self.writer.writerow(col_list)
        self.is_inited_tabel = True

    @staticmethod
    def format_time(seconds: float) -> str:
        """Format a duration in seconds to a human-readable string.

        Args:
            seconds (float): Number of seconds

        Returns:
            str: Formatted time string in "HH:MM:SS" format
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    @staticmethod
    def split_by_cond(data, conditions):
        """Split data by condition into a formatted string.

        Args:
            data: Data to split
            conditions: Training conditions defining splits

        Returns:
            str: Comma-separated string of values
        """
        result = ""
        i = 0
        for cond in conditions:
            loss = data[i : i + cond.output_len]
            result += ", ".join(loss)
            i += cond.output_len
        return result

    @staticmethod
    def sum_by_cond(data, conditions):
        """Calculate sums for each condition's data.

        Args:
            data: Data to sum
            conditions: Training conditions defining groups

        Returns:
            str: Formatted string of sums in scientific notation
        """
        result = "["
        i = 0
        for cond in conditions:
            loss = data[i : i + cond.output_len].sum()
            result += f"{loss:.2e} "
            i += cond.output_len
        result = result[:-1] + "]"
        return result
