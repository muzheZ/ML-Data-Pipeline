"""Pipeline configuration.

This module loads configuration information from files in the config directory.
"""

import pathlib

from omegaconf import OmegaConf


class Config:
    """Pipeline configuration."""

    def __init__(self, config_dir: str, project: str) -> None:
        """Instantiates the config object.

        Args:
            config_dir (str): Path to config directory.
            project (str): Project name. Must correspond to a YAML file.
        """
        self.config_dir = pathlib.Path(config_dir)
        self.project = project

    def load(self) -> None:
        """Loads configuration into the config object.

        Raises:
            FileNotFoundError: File not found.
            PermissionError: Insufficient permissions.
            IsADirectoryError: Project config path points to a directory.
            yaml.scanner.ScannerError: Invalid YAML.
        """
        self.items = OmegaConf.merge(
            OmegaConf.load(self.config_dir / "common.yaml"),
            OmegaConf.load(self.config_dir / "datasets.yaml"),
            OmegaConf.load(self.config_dir / f"projects/{self.project}.yaml"),
        )

    def __repr__(self):
        """Printable representation of an object of this type."""
        return OmegaConf.to_yaml(self.items)