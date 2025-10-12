"""Utility to load and parse configuration files for the feature store."""

from dataclasses import fields
from typing import Any, Dict

from src.feature_store.utils.config import (
    ClassMappingsConfig,
    Config,
    DataConfig,
    FeatureMappingsConfig,
    FeatureStoreConfig,
    FilesConfig,
)
from src.training.utils.config import (
    IncludedModelsConfig,
    LGBMConfig,
    LogisticRegressionConfig,
    ModelRegistryConfig,
    RandomForestConfig,
    TrainFeaturesConfig,
    TrainFilesConfig,
    TrainingConfig,
    TrainParams,
    TrainPreprocessingConfig,
    XGBoostConfig,
)
from src.utils.logger import LoggerConfig


def load_data_and_train_config(
    config_path: str,
) -> tuple[FeatureStoreConfig, TrainingConfig]:
    """Loads and parses the feature store and training configurations from a YAML file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        tuple[FeatureStoreConfig, TrainingConfig]: A tuple containing the feature store configuration
        and the training configuration as dataclass instances.
    """
    # Load and validate the configuration using the Config class
    config = Config(config_path=config_path)
    config.check_params()  # Validate the parameters

    # Map the configuration to the FeatureStoreConfig and TrainingConfig dataclasses
    feature_store_config = _build_feature_store_config(config.params)
    training_config = _build_training_config(config.params)

    return feature_store_config, training_config


def _build_feature_store_config(params: Dict[str, Any]) -> FeatureStoreConfig:
    """Builds the FeatureStoreConfig dataclass from the configuration parameters.

    Args:
        params (Dict[str, Any]): The configuration parameters.

    Returns:
        FeatureStoreConfig: The feature store configuration as a dataclass instance.
    """
    return FeatureStoreConfig(
        description=params["description"],
        logger=_map_to_dataclass(LoggerConfig, params["logger"]),
        data=_map_to_dataclass(DataConfig, params["data"]),
        feature_mappings=_map_to_dataclass(
            FeatureMappingsConfig, {"mappings": params["feature_mappings"]}
        ),
        class_mappings=_map_to_dataclass(ClassMappingsConfig, params["class_mappings"]),
        files=_map_to_dataclass(FilesConfig, params["files"]),
    )


def _build_training_config(params: Dict[str, Any]) -> TrainingConfig:
    """Builds the TrainingConfig dataclass from the configuration parameters.

    Args:
        params (Dict[str, Any]): The configuration parameters.

    Returns:
        TrainingConfig: The training configuration as a dataclass instance.
    """
    included_models_params = params.get(
        "included_models", {}
    )  # Fallback to an empty dictionary
    return TrainingConfig(
        description=params["description"],
        logger=_map_to_dataclass(LoggerConfig, params["logger"]),
        data=_map_to_dataclass(TrainFeaturesConfig, params["data"]),
        preprocessing=_map_to_dataclass(
            TrainPreprocessingConfig, params.get("preprocessing", {})
        ),
        train_params=_map_to_dataclass(TrainParams, params.get("train_params", {})),
        logistic_regression=_map_to_dataclass(
            LogisticRegressionConfig, params.get("logistic_regression", {})
        ),
        random_forest=_map_to_dataclass(
            RandomForestConfig, params.get("random_forest", {})
        ),
        lightgbm=_map_to_dataclass(LGBMConfig, params.get("lightgbm", {})),
        xgboost=_map_to_dataclass(XGBoostConfig, params.get("xgboost", {})),
        files=_map_to_dataclass(TrainFilesConfig, params["files"]),
        modelregistry=_map_to_dataclass(
            ModelRegistryConfig, params.get("modelregistry", {})
        ),
        included_models=_map_to_dataclass(IncludedModelsConfig, included_models_params),
    )


def _map_to_dataclass(dataclass_type: Any, config_dict: Dict[str, Any]) -> Any:
    """Map a dictionary to a dataclass, ensuring all fields are populated.

    Args:
        dataclass_type (Any): The dataclass type to map to.
        config_dict (Dict[str, Any]): The configuration dictionary.

    Returns:
        Any: An instance of the dataclass populated with values from the dictionary.
    """
    dataclass_fields = {field.name: field.default for field in fields(dataclass_type)}
    filtered_dict = {
        key: config_dict.get(key, default) for key, default in dataclass_fields.items()
    }
    return dataclass_type(**filtered_dict)
