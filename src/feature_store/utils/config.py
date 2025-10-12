"""Defines a class that loads parameters from config.yml file."""

import yaml


class PrettySafeLoader(yaml.SafeLoader):
    """A YAML loader that loads mappings into ordered dictionaries.

    Attributes:
        None.
    """

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
            FileNotFoundError: if config file doesn't exist.
            ValueError: if config file is not a .yml file.
        """

        if not config_path.endswith(".yml"):
            raise ValueError("Config file must be a .yml file")
        else:
            self.config_path = config_path

        try:
            with open(self.config_path, "r", encoding="UTF-8") as f:
                self.params = yaml.load(f, Loader=PrettySafeLoader)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"{config_path} doesn't exist.") from exc

    def check_params(self) -> None:
        """Checks all required values exist.

        Raises:
            AssertionError: if any required value is missing.
            ValueError: if any required description is missing.
            ValueError: if any required data parameter is missing.
            ValueError: if raw_dataset_source is not specified.
            ValueError: if pk_col_name is not specified.
            ValueError: if neither categorical nor numerical columns are specified.
            ValueError: if uci_raw_data_num is not an integer.
        """

        if "description" not in self.params:
            raise ValueError("description is not included in config file")

        if "data" not in self.params:
            raise ValueError("data is not included in config file")

        # Check beta value (primarily used to compare models)
        if self.params["data"]["raw_dataset_source"] == "none":
            raise ValueError("raw_dataset_source must be specified!")

        if self.params["data"]["pk_col_name"] == "none":
            raise ValueError("pk_col_name must be specified!")

        if (self.params["data"]["num_col_names"] == "none") and (
            self.params["data"]["cat_col_names"] == "none"
        ):
            raise ValueError("Neither categorical nor numerical are specified!")

        if not isinstance(self.params["data"]["uci_raw_data_num"], int):
            raise ValueError(
                f"uci_dataset_id must be integer type. Got {self.params['data']['params']['uci_raw_data_num']}"
            )
