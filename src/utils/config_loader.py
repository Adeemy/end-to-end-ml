"""Utility to load and parse configuration files."""

from dataclasses import fields
from pathlib import PosixPath
from typing import Any, Callable, Dict, Type, TypeVar

from src.utils.logger import get_console_logger

T = TypeVar("T")  # Generic type for any config dataclass

module_name: str = PosixPath(__file__).stem
logger = get_console_logger(module_name)


def map_to_dataclass(dataclass_type: Type[T], config_dict: Dict[str, Any]) -> T:
    """Map a dictionary to a dataclass, ensuring all fields are populated.

    Keys present in ``config_dict`` that do not correspond to a dataclass field
    are ignored, but a warning is logged so typos in the config surface instead
    of silently vanishing.
    """
    dataclass_fields = {field.name: field.default for field in fields(dataclass_type)}

    unknown_keys = set(config_dict) - set(dataclass_fields)
    if unknown_keys:
        logger.warning(
            "Ignoring unknown config key(s) for %s: %s",
            dataclass_type.__name__,
            ", ".join(sorted(unknown_keys)),
        )

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
