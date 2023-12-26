"""Defines a class that loads parameters from config.yml file."""
from typing import Dict

import yaml


class PrettySafeLoader(yaml.SafeLoader):
    """Custom loader for reading YAML files"""

    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


PrettySafeLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", PrettySafeLoader.construct_python_tuple
)


class Config:
    """A class that loads parameters from a yaml file.

    Args:
    config_path (str): path of the config .yml file.
    """

    def __init__(self, config_path: str):
        assert config_path.endswith(".yml")
        self.config_path = config_path

        try:
            with open(self.config_path, "r", encoding="UTF-8") as f:
                self.params = yaml.load(f, Loader=PrettySafeLoader)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"{config_path} doesn't exist.") from exc

    @staticmethod
    def _check_params(params: Dict):
        """Check all required values exist."""
        assert "description" in params, "description is not included in config file"
        assert "data" in params, "data is not included in config file"

        # Check beta value (primarily used to compare models)
        if params["data"]["params"]["raw_dataset_source"] == "none":
            raise ValueError("raw_dataset_source must be specified!")

        if params["data"]["params"]["pk_col_name"] == "none":
            raise ValueError("pk_col_name must be specified!")

        if (params["data"]["params"]["num_col_names"] == "none") and (
            params["data"]["params"]["cat_col_names"] == "none"
        ):
            raise ValueError("Neither categorical nor numerical are specified!")
