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

        assert config_path.endswith(".yml")
        self.config_path = config_path

        print(f"\n\nDirectory of training config file: {self.config_path}\n\n")

        try:
            with open(self.config_path, "r", encoding="UTF-8") as f:
                self.params = yaml.load(f, Loader=PrettySafeLoader)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"{config_path} doesn't exist.") from exc

    @staticmethod
    def _check_params(params: dict) -> None:
        """Check all training exp values.

        Args:
            params (dict): dictionary of config parameters.

        Raises:
            AssertionError: if any required parameter is missing.
            ValueError: if any parameter is of invalid type.
        """

        assert "description" in params, "description is not included in config file"
        assert "data" in params, "data is not included in config file"
        assert "train" in params, "train is not included in config file"
        assert "modelregistry" in params, "modelregistry is not included in config file"

        # Check data split params are of correct types
        if not isinstance(int(params["data"]["params"]["split_rand_seed"]), int):
            raise ValueError(
                f"split_rand_seed must be integer type. Got {params['data']['params']['split_rand_seed']}"
            )

        if params["data"]["params"]["split_type"] not in ["random", "time"]:
            raise ValueError(
                f"split_type must be either 'random' or 'time'. Got {params['data']['params']['split_type']}"
            )

        # Check beta value (primarily used to compare models)
        if isinstance(float(params["train"]["params"]["fbeta_score_beta_val"]), float):
            fbeta_score_beta_val = float(
                params["train"]["params"]["fbeta_score_beta_val"]
            )

            assert (
                fbeta_score_beta_val > 0
            ), f"fbeta_score_beta_val must be > 0. Got {fbeta_score_beta_val}"

        else:
            raise ValueError(
                f"fbeta_score_beta_val must be float type. Got {params['train']['params']['fbeta_score_beta_val']}"
            )

        # Check if comparison metric is a valid value
        comparison_metric = params["train"]["params"]["comparison_metric"]
        comparison_metrics = ("recall", "precision", "f1", "roc_auc", "fbeta_score")
        assert (
            comparison_metric in comparison_metrics
        ), f"Supported metrics are {comparison_metrics}. Got {comparison_metric}!"

        # Check if input split cutoff date (if split_type == "time") is in proper date format
        if params["data"]["params"]["split_type"] == "time" and (
            params["data"]["params"]["train_test_split_curoff_date"] == "none"
            or params["data"]["params"]["train_valid_split_curoff_date"] == "none"
        ):
            raise ValueError(
                f"train_test_split_curoff_date and train_valid_split_curoff_date must be a date (format {params['data']['params']['split_date_col_format']}) or None if split type is 'random'."
            )

        # Check if voting rule a valid value
        voting_rule = params["train"]["params"]["voting_rule"]
        assert voting_rule in (
            "soft",
            "hard",
        ), f"Voting rule in Voting Ensemble must be 'soft' or 'hard'. Got {voting_rule}!"
