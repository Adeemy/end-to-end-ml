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
        """Check all training exp values"""
        assert "description" in params, "description is not included in config file"
        assert "data" in params, "data is not included in config file"
        assert "train" in params, "train is not included in config file"
        assert "modelregistry" in params, "modelregistry is not included in config file"

        # Check beta value (primarily used to compare models)
        if params["train"]["params"]["fbeta_score_beta_val"] != "none":
            try:
                fbeta_score_beta_val = float(
                    params["train"]["params"]["fbeta_score_beta_val"]
                )
            except ValueError as e:
                raise ValueError("fbeta_score_beta_val must be a float type!") from e

            assert (
                fbeta_score_beta_val > 0
            ), f"fbeta_score_beta_val must be > 0. Got {fbeta_score_beta_val}"
            params["train"]["params"][
                "comparison_metric"
            ] = f"f_{fbeta_score_beta_val}_score"

        # Check if comparison metric is a valid value
        comparison_metric = params["train"]["params"]["comparison_metric"]
        comparison_metrics = ("recall", "precision", "f1", "roc_auc", "fbeta_score")
        assert (
            comparison_metric in comparison_metrics
        ), f"Supported metrics are {comparison_metrics}. Got {comparison_metric}!"

        # Check if voting rule a valid value
        voting_rule = params["train"]["params"]["voting_rule"]
        assert voting_rule in (
            "soft",
            "hard",
        ), f"Voting rule in Voting Ensemble must be 'soft' or 'hard'. Got {voting_rule}!"
