import os
import torch
import torch.distributed as dist

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from examples.navier_stokes_2D_obstacle_multigpu.problem import (
    problem,
)
from multipinn import *
from multipinn.trainer.trainer import TrainerMultiGPU
from multipinn.utils import (
    initialize_model,
    initialize_regularization,
    save_config,
    set_device_and_seed,
)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig):
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
    
    num_gpus = int(os.environ.get('NUM_GPUS_TRAIN', dist.get_world_size()))
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    if rank == 0:
        config_save_path = os.path.join(cfg.paths.save_dir, "used_config.yaml")
        save_config(cfg, config_save_path)

    conditions, input_dim, output_dim = problem(re=cfg.problem.re)
    set_device_and_seed(cfg.trainer.random_seed + rank, accelerator=f'cuda:{local_rank}')

    model = initialize_model(cfg, input_dim, output_dim, rank=rank).to(device)
    calc_loss = initialize_regularization(cfg, rank=rank)

    total_domain_points = cfg.generator.domain_points * num_gpus
    total_bound_points = cfg.generator.bound_points * num_gpus
    
    generator_domain = Generator(
        n_points=total_domain_points, sampler=cfg.generator.sampler
    )
    generator_bound = Generator(
        n_points=total_bound_points, sampler=cfg.generator.sampler
    )

    generator_domain.use_for(conditions[0])
    generator_bound.use_for(conditions[1:])

    pinn = PINN(model=model, conditions=conditions)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    callbacks = []
    if rank == 0:
        callbacks = [
            progress.TqdmBar(
                "Epoch {epoch} lr={lr:.2e} Loss={loss_eq} Total={total_loss:.2e}"
            ),
            curve.LossCurve(cfg.paths.save_dir, cfg.visualization.save_period),
            save.SaveModel(cfg.paths.save_dir, period=cfg.visualization.save_period),
        ]

    trainer = TrainerMultiGPU(
        pinn=pinn,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=cfg.trainer.num_epochs,
        num_batches=num_gpus,  
        update_grid_every=cfg.trainer.grid_update,
        calc_loss=calc_loss,
        callbacks_organizer=CallbacksOrganizer(callbacks),
        mixed_training=cfg.trainer.get('mixed_training', False),
        rank=rank,
    )

    if rank == 0:
        print(f"Starting training on {num_gpus} GPUs")
        print(f"Points per GPU: {total_domain_points//num_gpus} domain + {total_bound_points//num_gpus} boundary")

    trainer.train()
    dist.destroy_process_group()


if __name__ == "__main__":
    train() 