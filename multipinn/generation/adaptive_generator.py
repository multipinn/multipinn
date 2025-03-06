from __future__ import annotations

from typing import List

import numpy as np
import torch

from .generator import Generator
from ..condition import Condition


class AdaptiveGeneratorRect(Generator):
    def __init__(self, n_points, power_coeff=3, add_coeff=1, density_rec_points_num=None,
                 add_points=None, n_points_up_bnd=None):
        super().__init__(n_points)

        self.power_coeff = power_coeff
        self.add_coeff = add_coeff
        self.density_rec_points_num = density_rec_points_num if density_rec_points_num is not None else n_points * 8
        self.add_points = add_points if add_points is not None else n_points // 64
        self.n_points_up_bnd = n_points_up_bnd if n_points_up_bnd is not None else density_rec_points_num

    def set_batching(self, new_num_batches):
        self.n_points = self.n_points // self.num_batches * new_num_batches
        self.add_points = self.add_points // self.num_batches * new_num_batches
        self.density_rec_points_num = self.density_rec_points_num // self.num_batches * new_num_batches
        self.n_points_up_bnd = self.n_points_up_bnd // self.num_batches * new_num_batches
        self.num_batches = new_num_batches

    def calc_error_field(self, geometry, condition: Condition, model):
        self.device = next(model.parameters()).device
        density_points = torch.tensor(
            np.random.uniform(
                low=geometry.low,
                high=geometry.high,
                size=(self.density_rec_points_num, len(geometry.low)),
            ),
            dtype=torch.float32,
            requires_grad=True
        )
        residuals = condition.get_residual_fn(model)(density_points)
        residual = torch.stack(residuals, dim=1).abs().sum(dim=1)
        error = residual.cpu().detach().numpy()
        density_points = density_points.detach()
        return density_points, error

    def sample_from_density(self, errors, points_num):
        err_eq = np.power(errors, self.power_coeff) / np.mean(np.power(errors, self.power_coeff)) + self.add_coeff
        err_eq_norm = err_eq / np.sum(err_eq)
        pnts_ids = np.random.choice(a=err_eq_norm.shape[0], size=points_num, replace=False, p=err_eq_norm)
        return pnts_ids

    @staticmethod
    def sample_max_error(error, points_num):
        return np.argpartition(error, -points_num)[-points_num:]

    @staticmethod
    def sample_min_error(error, points_num):
        return np.argpartition(error, points_num)[points_num:]


class AdaptiveGeneratorRectRAR_D(AdaptiveGeneratorRect):
    def __init__(self, n_points, power_coeff=3, add_coeff=1, density_rec_points_num=10000,
                 add_points=50, n_points_up_bnd=None):
        """RAR-D adaptive generator

        Args:
            n_points (_type_): number of points to sample
            power_coeff (_type_): https://arxiv.org/abs/2207.10289 coefficient in power
            add_coeff (_type_): https://arxiv.org/abs/2207.10289 addtional coefficient
            density_rec_points_num (int, optional): number of points neccessery to recover error field. Defaults to 10000.
            add_points (int, optional): number of points to add. Defaults to 50.
            n_points_up_bnd (int, optional): upper limit on points number.
        """
        super().__init__(n_points, power_coeff, add_coeff, density_rec_points_num)
        self.add_points = add_points if add_points is not None else n_points // 64
        self.n_points_up_bnd = n_points_up_bnd if n_points_up_bnd is not None else density_rec_points_num
    
    def generate(self, condition: Condition, model):
        """ Samples points uniformly at the begining, then adds some points
        according to the error probability during every update step.

        Returns:
            torch.tensor: training points tensor
        """
        if condition.points is None:
            points = super().generate(condition, model)
            self.points = points
            return points
        if condition.points.shape[0] >= self.n_points_up_bnd:
            self.points = condition.points
            return condition.points

        density_points, error = self.calc_error_field(condition.geometry, condition, model)
        chosen_points_id = self.sample_from_density(error, self.add_points)
        chosen_points = density_points[chosen_points_id].detach()
        del density_points
        delete_points_id = self.sample_min_error(
            torch.stack(condition.get_residual(model), dim=1).abs().sum(dim=1).cpu().detach().numpy(), self.add_points)
        points = torch.cat([chosen_points, condition.points[delete_points_id]], dim=0).detach().requires_grad_()
        points.update = chosen_points.clone().detach().cpu().numpy()
        self.points = points
        return points


class AdaptiveGeneratorRectRAR_G(AdaptiveGeneratorRect):
    def __init__(self, n_points, power_coeff=3, add_coeff=1, density_rec_points_num=10000,
                 add_points=50, n_points_up_bnd=None):
        """RAR-G adaptive generator

        Args:
            n_points (_type_): number of points to sample
            power_coeff (_type_): https://arxiv.org/abs/2207.10289 coefficient in power
            add_coeff (_type_): https://arxiv.org/abs/2207.10289 addtional coefficient
            density_rec_points_num (int, optional): number of points neccessery to recover error field. Defaults to 10000.
            add_points (int, optional): number of points to add. Defaults to 50.
            n_points_up_bnd (int, optional): upper limit on points number.
        """
        super().__init__(n_points, power_coeff, add_coeff, density_rec_points_num)
        self.add_points = add_points if add_points is not None else n_points // 64
        self.n_points_up_bnd = n_points_up_bnd if n_points_up_bnd is not None else density_rec_points_num

    def generate(self, condition: Condition, model):
        """ Samples points uniformly at the begining, then adds some points
        with the biggest error during every update step.

        Returns:
            torch.tensor: training points tensor
        """
        if condition.points is None:
            points = super().generate(condition, model)
            self.points = points
            return points
        if condition.points.shape[0] >= self.n_points_up_bnd:
            self.points = condition.points
            return condition.points
        density_points, error = self.calc_error_field(condition.geometry, condition, model)
        chosen_points_id = self.sample_max_error(error, self.add_points)
        chosen_points = density_points[chosen_points_id].detach()
        # delete_points_id = self.sample_min_error(torch.stack(condition.get_residual(model), dim=1).abs().sum(dim=1).cpu().detach().numpy(), self.add_points)
        # points = torch.cat([chosen_points, condition.points[delete_points_id]], dim=0).detach().requires_grad_()
        points = torch.cat([chosen_points, condition.points], dim=0)[:self.n_points].detach().requires_grad_()
        points.update = chosen_points.clone().detach().cpu().numpy()
        # points.update = chosen_points.clone().detach().cpu().numpy()
        self.points = points
        return points


class AdaptiveGeneratorRectRAD(AdaptiveGeneratorRect):
    def __init__(self, n_points, power_coeff=3, add_coeff=1, density_rec_points_num=None,
                ):
        """RAD adaptive generator

        Args:
            n_points (_type_): number of points to sample
            power_coeff (_type_): https://arxiv.org/abs/2207.10289 coefficient in power
            add_coeff (_type_): https://arxiv.org/abs/2207.10289 addtional coefficient
            density_rec_points_num (int, optional): number of points neccessery to recover error field. Defaults to 10000.
        """
        super().__init__(n_points, power_coeff, add_coeff, density_rec_points_num)

    def generate(self, condition: Condition, model):

        """ Samples points according error probability. Fully updates training set.

        Returns:
            torch.tensor: training points tensor
        """
        if condition.points is None:
            points = super().generate(condition, model)
            self.points = points
            return points

        density_points, error = self.calc_error_field(condition.geometry, condition, model)
        chosen_points_id = self.sample_from_density(error, self.n_points)
        chosen_points = density_points[chosen_points_id].requires_grad_()
        del density_points
        chosen_points.update = chosen_points.clone().detach().cpu().numpy()
        self.points = chosen_points
        return chosen_points


class AdaptiveGeneratorRectRAG(AdaptiveGeneratorRect):
    def __init__(self, n_points, power_coeff=3, add_coeff=1, density_rec_points_num=None,
                ):
        """RAG adaptive generator

        Args:
            n_points (_type_): number of points to sample
            power_coeff (_type_): https://arxiv.org/abs/2207.10289 coefficient in power
            add_coeff (_type_): https://arxiv.org/abs/2207.10289 addtional coefficient
            density_rec_points_num (int, optional): number of points neccessery to recover error field. Defaults to 10000.
        """
        super().__init__(n_points, power_coeff, add_coeff, density_rec_points_num)
    

    def generate(self, condition: Condition, model):
        """
        Samples points with the biggest error during every update step. Fully updates training set.

        Returns:
            torch.tensor: training points tensor
        """

        if condition.points is None:
            points = super().generate(condition, model)
            self.points = points
            return points
        density_points, error = self.calc_error_field(condition.geometry, condition, model)
        chosen_points_id = self.sample_max_error(error, self.n_points)
        chosen_points = density_points[chosen_points_id].requires_grad_()
        chosen_points.update = chosen_points.clone().detach().cpu().numpy()
        self.points = chosen_points
        return chosen_points
