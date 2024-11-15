import os

import yaml
from omegaconf import DictConfig, OmegaConf


def save_config(cfg: DictConfig, save_path: str):
    """
    Save a Hydra configuration object to a YAML file.

    This function takes a Hydra configuration object and saves it to disk in YAML format.
    It automatically creates any necessary parent directories if they don't exist.

    Args:
        cfg (DictConfig): Hydra configuration object to save.
        save_path (str): File path where the configuration should be saved.
            The path can include directories that don't exist yet.

    Returns:
        None

    Example:
        >>> cfg = DictConfig({"model": {"type": "mlp", "layers": [64, 32]}})
        >>> save_config(cfg, "configs/model_config.yaml")
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(
            OmegaConf.to_container(cfg, resolve=True), f, default_flow_style=False
        )
