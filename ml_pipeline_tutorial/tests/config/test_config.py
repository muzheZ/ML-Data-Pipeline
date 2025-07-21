import pathlib

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

from omegaconf import DictConfig, OmegaConf

from ml_pipeline.config import Config


def get_config(config_type: "pathlib.Path") -> DictConfig:
    common_config_str = """
        common_item_a: a
    """
    dataset_config_str = """
        dataset_item_a:
          dataset_item_a_a: 0
    """
    project_config_str = """
        project_item_a: 0
        project_item_b: b
        project_list_item:
          - x
          - y
          - z
    """
    if config_type == pathlib.Path("config/common.yaml"):
        return OmegaConf.create(common_config_str)
    elif config_type == pathlib.Path("config/datasets.yaml"):
        return OmegaConf.create(dataset_config_str)
    elif config_type == pathlib.Path("config/projects/project.yaml"):
        return OmegaConf.create(project_config_str)


def test_load(monkeypatch: "pytest.MonkeyPatch") -> None:
    monkeypatch.setattr(OmegaConf, "load", get_config)

    config = Config("config", "project")
    config.load()
    expected_config_str = """
        common_item_a: a
        dataset_item_a:
          dataset_item_a_a: 0
        project_item_a: 0
        project_item_b: b
        project_list_item:
          - x
          - y
          - z
    """
    assert config.items == OmegaConf.create(expected_config_str)