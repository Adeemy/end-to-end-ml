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
        """

        assert config_path.endswith(".yml")
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
            AssertionError: if any required value is missing.
        """

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

        if not isinstance(params["data"]["params"]["uci_raw_data_num"], int):
            raise ValueError(
                f"uci_dataset_id must be integer type. Got {params['data']['params']['uci_raw_data_num']}"
            )
