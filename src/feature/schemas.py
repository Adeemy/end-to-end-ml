"""Feature store configuration loader and dataclasses.

Handles loading and validation of feature store configuration from YAML files.
Defines dataclasses for different configuration sections.
"""

from dataclasses import dataclass
from pathlib import PosixPath
from typing import Any, Dict, List

import yaml

from src.utils.config_loader import map_to_dataclass
from src.utils.logger import LoggerConfig, get_console_logger

module_name: str = PosixPath(__file__).stem
logger = get_console_logger(module_name)


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
            KeyError:
               if 'description' is not included in config file.
               if 'data' is not included in config file.
               if 'raw_dataset_source' is not specified.
               if 'pk_col_name' is not specified.
            ValueError:
               if neither categorical nor numerical columns are specified.
               if 'uci_raw_data_num' is not an integer.
        """

        if "description" not in self.params:
            raise KeyError("description is not included in config file")

        if "data" not in self.params:
            raise KeyError("data is not included in config file")

        # Check beta value (primarily used to compare models)
        if self.params["data"]["raw_dataset_source"] == "none":
            raise KeyError("raw_dataset_source must be specified!")

        if self.params["data"]["pk_col_name"] == "none":
            raise KeyError("pk_col_name must be specified!")

        if (self.params["data"]["num_col_names"] == "none") and (
            self.params["data"]["cat_col_names"] == "none"
        ):
            raise ValueError("Neither categorical nor numerical are specified!")

        if not isinstance(self.params["data"]["uci_raw_data_num"], int):
            raise ValueError(
                f"uci_dataset_id must be integer type. Got {self.params['data']['params']['uci_raw_data_num']}"
            )


@dataclass(frozen=True)
class DataConfig:
    """Configuration for data handling."""

    uci_raw_data_num: int
    raw_dataset_source: str
    pk_col_name: str
    class_col_name: str
    pos_class: str
    date_col_names: List[str]
    datetime_col_names: List[str]
    inference_set_ratio: float
    original_split_type: str
    random_seed: int
    event_timestamp_col_name: str
    num_col_names: List[str]
    cat_col_names: List[str]
    entity_name: str
    entity_description: str
    feature_view_name: str
    feature_view_description: str
    target_view_name: str
    target_view_description: str
    view_tags_name_1: str
    view_tags_value_1: str
    ttl_duration_in_days: int


@dataclass(frozen=True)
class FeatureMappingsConfig:
    """Configuration for feature mappings."""

    mappings: Dict[str, Dict[str, str]]


@dataclass(frozen=True)
class FilesConfig:
    """Configuration for file paths."""

    raw_dataset_file_name: str
    inference_set_file_name: str
    preprocessed_data_features_file_name: str
    preprocessed_data_target_file_name: str


@dataclass(frozen=True)
class FeatureStoreConfig:
    """Main configuration for the feature store."""

    description: str
    logger: LoggerConfig
    data: DataConfig
    feature_mappings: FeatureMappingsConfig
    files: FilesConfig


def build_feature_store_config(params: Dict[str, Any]) -> FeatureStoreConfig:
    """Builds the FeatureStoreConfig dataclass from the configuration parameters.

    Args:
        params (Dict[str, Any]): The configuration parameters.

    Returns:
        FeatureStoreConfig: The feature store configuration as a dataclass instance.
    """

    return FeatureStoreConfig(
        description=params["description"],
        logger=map_to_dataclass(LoggerConfig, params["logger"]),
        data=map_to_dataclass(DataConfig, params["data"]),
        feature_mappings=map_to_dataclass(
            FeatureMappingsConfig, {"mappings": params["feature_mappings"]}
        ),
        files=map_to_dataclass(FilesConfig, params["files"]),
    )
