"""Defines a class that loads parameters from config.yml file."""

import yaml


class PrettySafeLoader(yaml.SafeLoader):
    """Custom loader for reading YAML files.

    Attributes:
        None.
    """

    def construct_python_tuple(self, node) -> tuple:
        """Override the default constructor to create tuples instead of lists.

        Args:
            node (str): yaml node.

        Returns:
            tuple: python tuple.
        """

        return tuple(self.construct_sequence(node))


PrettySafeLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", PrettySafeLoader.construct_python_tuple
)


class Config:
    """A class that loads parameters from a yaml file.

    Args:
        config_path (str): path of the config .yml file.
    """

    def __init__(self, config_path: str) -> None:
        """Creates a Config instance.

        Args:
            config_path (str): path of the config .yml file.

        Raises:
            FileNotFoundError: if config file doesn't exist.
        """

        if not config_path.endswith(".yml"):
            raise ValueError("Config path must end with '.yml'")

        self.config_path = config_path

        try:
            with open(self.config_path, "r", encoding="UTF-8") as f:
                self.params = yaml.load(f, Loader=PrettySafeLoader)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"{config_path} doesn't exist.") from exc

    @staticmethod
    def _check_params(params: dict) -> None:
        """Checks all training exp values.

        Args:
            params (dict): dictionary of config parameters.

        Raises:
            ValueError: if description or data is not included in config file.
            ValueError: if data parameters are not included in config file.
            ValueError: if train parameters are not included in config file.
            ValueError: if modelregistry parameters are not included in config file.
            ValueError: if split_rand_seed is not an integer.
            ValueError: if split_type is not 'random' or 'time'.
            ValueError: if fbeta_score_beta_val is not a positive float.
            ValueError: if comparison_metric is not a valid value.
            ValueError: if split_date_col_format is not a valid date format.
            ValueError: if train_test_split_curoff_date or train_valid_split_curoff_date is not a date and split_type is 'time'.
            ValueError: if voting_rule is not 'soft' or 'hard'.
        """

        if "description" not in params:
            raise ValueError("description is not included in config file")
        if "data" not in params:
            raise ValueError("data is not included in config file")
        if "train" not in params:
            raise ValueError("train is not included in config file")
        if "modelregistry" not in params:
            raise ValueError("modelregistry is not included in config file")

        # Check data split params are of correct types
        try:
            _ = int(params["data"]["split_rand_seed"])
        except ValueError as exc:
            raise ValueError(
                f"split_rand_seed must be integer type. Got {params['data']['split_rand_seed']}"
            ) from exc

        split_type = params.get("data", {}).get("split_type", None)
        if split_type not in ["random", "time"]:
            raise ValueError(
                f"split_type must be either 'random' or 'time'. Got {split_type}"
            )

        # Check beta value (primarily used to compare models)
        fbeta_score_beta_val = params.get("train", {}).get("fbeta_score_beta_val", 1.0)
        try:
            fbeta_score_beta_val = float(fbeta_score_beta_val)
            if fbeta_score_beta_val <= 0:
                raise ValueError(
                    f"fbeta_score_beta_val must be > 0. Got {fbeta_score_beta_val}"
                )
        except ValueError as exc:
            raise ValueError(
                f"fbeta_score_beta_val must be a positive float. Got {fbeta_score_beta_val}"
            ) from exc

        # Check if comparison metric is a valid value
        comparison_metric = params.get("train", {}).get("comparison_metric", None)
        supported_comparison_metrics = [
            "recall",
            "precision",
            "f1",
            "roc_auc",
            "fbeta_score",
        ]
        if comparison_metric not in supported_comparison_metrics:
            raise ValueError(
                f"Supported metrics are {supported_comparison_metrics}. Got {comparison_metric}!"
            )

        # Check if input split cutoff date (if split_type == "time") is in proper date format
        date_format = params.get("data", {}).get("split_date_col_format", None)
        train_test_split_curoff_date = params.get("data", {}).get(
            "train_test_split_curoff_date", None
        )
        train_valid_split_curoff_date = params.get("data", {}).get(
            "train_valid_split_curoff_date", None
        )
        if split_type == "time" and (
            train_test_split_curoff_date == "none"
            or train_valid_split_curoff_date == "none"
        ):
            raise ValueError(
                f"train_test_split_curoff_date and train_valid_split_curoff_date must be a date (format {date_format}) or None if split type is 'random'."
            )

        # Check if voting rule a valid value
        voting_rule = params.get("train", {}).get("voting_rule", None)
        if voting_rule not in ("soft", "hard"):
            raise ValueError(
                f"Voting rule in Voting Ensemble must be 'soft' or 'hard'. Got {voting_rule}!"
            )
