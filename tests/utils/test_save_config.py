import os
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml
from omegaconf import DictConfig, OmegaConf

from multipinn.utils import save_config


@pytest.fixture
def config():
    return OmegaConf.create(
        {
            "param1": "value1",
            "param2": "value2",
            "nested": {
                "param3": "value3",
            },
        }
    )


@pytest.fixture
def cleanup_dirs():
    dirs_to_cleanup = []

    yield dirs_to_cleanup

    for dir_path in dirs_to_cleanup:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)


def test_save_config_valid_path(config, tmp_path):
    save_path = tmp_path / "config.yaml"
    save_config(config, str(save_path))
    assert os.path.exists(save_path)
    with open(save_path, "r") as f:
        loaded_config = yaml.safe_load(f)
    assert loaded_config == {
        "param1": "value1",
        "param2": "value2",
        "nested": {
            "param3": "value3",
        },
    }


def test_save_config_creates_directories(config, cleanup_dirs):
    temp_dir = tempfile.mkdtemp()
    non_existent_dir = os.path.join(temp_dir, "non_existent_dir")
    save_path = os.path.join(non_existent_dir, "config.yaml")

    cleanup_dirs.append(temp_dir)

    save_config(config, save_path)
    assert os.path.exists(non_existent_dir)
    assert os.path.exists(save_path)

    with open(save_path, "r") as f:
        loaded_config = yaml.safe_load(f)
    assert loaded_config == {
        "param1": "value1",
        "param2": "value2",
        "nested": {
            "param3": "value3",
        },
    }


def test_save_config_empty(config, tmp_path):
    empty_config = OmegaConf.create()
    save_path = tmp_path / "empty_config.yaml"
    save_config(empty_config, str(save_path))
    with open(save_path, "r") as f:
        loaded_config = yaml.safe_load(f)
    assert loaded_config == {}
