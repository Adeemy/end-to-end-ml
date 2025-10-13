"""Utility to load and parse configuration files."""

from dataclasses import fields
from typing import Any, Callable, Dict, Type, TypeVar

T = TypeVar("T")  # Generic type for any config dataclass


def map_to_dataclass(dataclass_type: Type[T], config_dict: Dict[str, Any]) -> T:
    """Map a dictionary to a dataclass, ensuring all fields are populated."""
    dataclass_fields = {field.name: field.default for field in fields(dataclass_type)}
    filtered_dict = {
        key: config_dict.get(key, default) for key, default in dataclass_fields.items()
    }
    return dataclass_type(**filtered_dict)


def load_config(config_class: Type, builder_func: Callable, config_path: str) -> Any:
    """Generic function to load any config type.

    Args:
        config_class: The config class to use for loading and validation
        builder_func: Function to build the final config object
        config_path: Path to the config YAML file

    Returns:
        The built configuration object
    """
    # Load and validate the configuration
    config = config_class(config_path=config_path)
    config.check_params()

    # Map to the target dataclass
    return builder_func(config.params)
