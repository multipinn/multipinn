import os

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from examples.surfaces_elliptic_adaptive.problem import sphere
from multipinn import *
from multipinn.utils import (
    initialize_model,
    initialize_regularization,
    save_config,
    set_device_and_seed,
)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig):
    config_save_path = os.path.join(cfg.paths.save_dir, "used_config.yaml")
    save_config(cfg, config_save_path)

    conditions, input_dim, output_dim = sphere()
    set_device_and_seed(cfg.trainer.random_seed)

    model = initialize_model(cfg, input_dim, output_dim)

    calc_loss = initialize_regularization(cfg)

    generator_surf = AdaptiveGeneratorRAR_G(
        cfg.generator.domain_points,
        power_coeff=1,
        add_coeff=0,
        add_points=20000,
        n_points_up_bnd=1000000,
        density_rec_points_num=200000,
    )

    generator_other = Generator(
        n_points=1, sampler='pseudo'
    )

    generator_surf.use_for(conditions[0])
    generator_other.use_for(conditions[1])

    pinn = PINN(model=model, conditions=conditions)

    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    grid = heatmap.Grid.from_pinn(pinn, cfg.visualization.grid_plot_points)

    callbacks = [
        progress.TqdmBar(
            "Epoch {epoch} lr={lr:.2e} Loss={loss_eq} Total={total_loss:.2e}"
        ),
        curve.LossCurve(
            cfg.paths.save_dir, cfg.visualization.save_period, save_mode="html"
        ),
        curve.LearningRateCurve(
            cfg.paths.save_dir, cfg.visualization.save_period),
        save.SaveModel(cfg.paths.save_dir,
                       period=cfg.visualization.save_period)
    ]

    trainer = Trainer(
        pinn=pinn,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=cfg.trainer.num_epochs,
        num_batches=1,
        update_grid_every=cfg.trainer.grid_update,
        calc_loss=calc_loss,
        callbacks_organizer=CallbacksOrganizer(callbacks),
    )

    trainer.train()


if __name__ == "__main__":
    train()
