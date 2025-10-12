"""Utility to load and parse configuration files for the feature store."""

from dataclasses import dataclass, fields
from typing import Any, Dict, List

from src.feature_store.utils.config import Config


@dataclass(frozen=True)
class LoggerConfig:
    """Configuration for logging."""

    entity: str
    project: str


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
class ClassMappingsConfig:
    """Configuration for class mappings."""

    class_column: str
    class_values: Dict[str, str]


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
    class_mappings: ClassMappingsConfig
    files: FilesConfig


def load_config(config_path: str) -> FeatureStoreConfig:
    """Load the feature store configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        FeatureStoreConfig: The loaded configuration as a dataclass instance.
    """

    # Load and validate the configuration using the Config class
    config = Config(config_path=config_path)
    config.check_params()  # Validate the parameters

    return FeatureStoreConfig(
        description=config.params["description"],
        logger=_map_to_dataclass(LoggerConfig, config.params["logger"]),
        data=_map_to_dataclass(DataConfig, config.params["data"]),
        feature_mappings=_map_to_dataclass(
            FeatureMappingsConfig, {"mappings": config.params["feature_mappings"]}
        ),
        class_mappings=_map_to_dataclass(
            ClassMappingsConfig, config.params["class_mappings"]
        ),
        files=_map_to_dataclass(FilesConfig, config.params["files"]),
    )


def _map_to_dataclass(dataclass_type: Any, config_dict: Dict[str, Any]) -> Any:
    """Map a dictionary to a dataclass, ensuring all fields are populated.

    Args:
        dataclass_type (Any): The dataclass type to map to.
        config_dict (Dict[str, Any]): The configuration dictionary.

    Returns:
        Any: An instance of the dataclass populated with values from the dictionary.
    """
    dataclass_fields = {field.name for field in fields(dataclass_type)}
    filtered_dict = {
        key: value for key, value in config_dict.items() if key in dataclass_fields
    }
    return dataclass_type(**filtered_dict)
