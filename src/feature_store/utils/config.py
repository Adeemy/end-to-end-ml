"""Defines a class that loads parameters from config.yml file."""

import yaml


class PrettySafeLoader(yaml.SafeLoader):
    """A YAML loader that loads mappings into ordered dictionaries."""

    def construct_python_tuple(self, node: str) -> tuple:
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
    """Loads parameters from config.yml file.

    Attributes:
        config_path (str): path of the config .yml file.
    """

    def __init__(self, config_path: str) -> None:
        """Creates a Config instance.

        Args:
            config_path (str): path of the config .yml file.

        Raises:
            ValueError: if config path doesn't end with '.yml'.
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
        """Checks all required values exist.

        Args:
            params (dict): dictionary of parameters loaded from config file.

        Raises:
            ValueError: if description or data is not included in config file.
            ValueError: if data parameters are not included in config file.
            ValueError: if raw_dataset_source is not specified.
            ValueError: if pk_col_name is not specified.
            ValueError: if neither categorical nor numerical are specified.
            ValueError: if uci_dataset_id is not integer type.
        """

        if "description" not in params:
            raise ValueError("description is not included in config file")
        if "data" not in params:
            raise ValueError("data is not included in config file")

        # Check beta value (primarily used to compare models)
        raw_dataset_source = params.get("data", {}).get("raw_dataset_source", None)
        if raw_dataset_source is None or raw_dataset_source == "none":
            raise ValueError("raw_dataset_source must be specified!")

        pk_col_name = params.get("data", {}).get("pk_col_name", None)
        if pk_col_name is None or pk_col_name == "none":
            raise ValueError("pk_col_name must be specified!")

        num_col_names = params.get("data", {}).get("num_col_names", None)
        cat_col_names = params.get("data", {}).get("cat_col_names", None)
        if (num_col_names is None or num_col_names == "none") and (
            cat_col_names is None or cat_col_names == "none"
        ):
            raise ValueError("Neither categorical nor numerical are specified!")

        uci_raw_data_num = params.get("data", {}).get("uci_raw_data_num", None)
        try:
            _ = int(uci_raw_data_num)
        except ValueError as exc:
            raise ValueError(
                f"uci_dataset_id must be integer type. Got {uci_raw_data_num}"
            ) from exc
