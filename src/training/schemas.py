"""
Defines a class that loads parameters from config.yml file and dataclasses for
configuration sections. It includes validation logic to ensure required parameters
are present and correctly typed.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import PosixPath
from typing import Any, Dict, List, Union

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
                - if 'description' is not included in config file.
                - if 'data' is not included in config file.
                - if 'train' is not included in config file.
                - if 'modelregistry' is not included in config file.
            ValueError:
                - if 'split_rand_seed' is not an integer.
                - if 'split_type' is not either 'random' or 'time'.
                - if 'fbeta_score_beta_val' is not a float > 0.
                - if 'comparison_metric' is not one of 'recall', 'precision', '
                    'f1', 'roc_auc', 'fbeta_score'.
                - if 'train_test_split_curoff_date' or 'train_valid_split_curoff_date'
                    is 'none' when 'split_type' is 'time'.
                - if 'voting_rule' is not either 'soft' or 'hard'.
        """

        if "description" not in self.params:
            raise KeyError("description is not included in config file")

        if "data" not in self.params:
            raise KeyError("data is not included in config file")

        if "train" not in self.params:
            raise KeyError("train is not included in config file")

        if "modelregistry" not in self.params:
            raise KeyError("modelregistry is not included in config file")

        # Check data split params are of correct types
        if not isinstance(int(self.params["data"]["split_rand_seed"]), int):
            raise ValueError(
                f"split_rand_seed must be integer type. Got {self.params['data']['params']['split_rand_seed']}"
            )

        if self.params["data"]["split_type"] not in ["random", "time"]:
            raise ValueError(
                f"split_type must be either 'random' or 'time'. Got {self.params['data']['params']['split_type']}"
            )

        # Check beta value (primarily used to compare models)
        if isinstance(float(self.params["train"]["fbeta_score_beta_val"]), float):
            fbeta_score_beta_val = float(self.params["train"]["fbeta_score_beta_val"])

            if fbeta_score_beta_val <= 0:
                raise ValueError(
                    f"fbeta_score_beta_val must be > 0. Got {fbeta_score_beta_val}"
                )

        else:
            raise ValueError(
                f"fbeta_score_beta_val must be float type. Got {self.params['train']['params']['fbeta_score_beta_val']}"
            )

        # Check if comparison metric is a valid value
        comparison_metric = self.params["train"]["comparison_metric"]
        comparison_metrics = ("recall", "precision", "f1", "roc_auc", "fbeta_score")
        if comparison_metric not in comparison_metrics:
            raise ValueError(
                f"Supported metrics are {comparison_metrics}. Got {comparison_metric}!"
            )

        # Check if input split cutoff date (if split_type == "time") is in proper date format
        if self.params["data"]["split_type"] == "time" and (
            self.params["data"]["train_test_split_curoff_date"] == "none"
            or self.params["data"]["train_valid_split_curoff_date"] == "none"
        ):
            raise ValueError(
                f"train_test_split_curoff_date and train_valid_split_curoff_date must be a date (format {self.params['data']['params']['split_date_col_format']}) or None if split type is 'random'."
            )

        # Check if voting rule is a valid value
        voting_rule = self.params["train"]["voting_rule"]
        if voting_rule not in ("soft", "hard"):
            raise ValueError(
                f"Voting rule in Voting Ensemble must be 'soft' or 'hard'. Got {voting_rule}!"
            )


@dataclass(frozen=True)
class TrainFeaturesConfig:
    """Configuration for defining training features."""

    raw_dataset_source: str
    split_type: str
    split_rand_seed: str
    split_date_col_name: str
    train_test_split_curoff_date: str
    train_valid_split_curoff_date: date
    split_date_col_format: str
    cat_features_nan_replacement: str
    train_set_size: float
    pk_col_name: str
    class_col_name: str
    pos_class: str
    date_col_names: List[str]
    datetime_col_names: List[str]
    num_col_names: List[str]
    cat_col_names: List[str]
    historical_features: List[str]


@dataclass(frozen=True)
class TrainPreprocessingConfig:
    """Configuration for preprocessing training data."""

    num_features_imputer: str = "mean"
    num_features_scaler: str = "standard"
    scaler_params: Dict[str, Any] = None
    cat_features_imputer: str = "most_frequent"
    cat_features_ohe_handle_unknown: str = "error"
    cat_features_nans_replacement: str = "Unknown"
    var_thresh_val: float = 0.0


@dataclass(frozen=True)
class TrainParams:
    """Main configuration for training parameters."""

    initiate_comet_project: bool = False
    experiment_tracker: str = "comet"
    project_name: str = "default-project"
    workspace_name: str = "comet-workspace-name"
    search_max_iters: int = 10
    parallel_jobs_count: int = 1
    exp_timout_secs: int = 3600
    cross_val_folds: int = 5
    fbeta_score_beta_val: float = 0.5
    comparison_metric: str = "fbeta_score"
    voting_rule: str = "soft"
    deployment_score_thresh: float = 0.8
    max_eval_experiments: int = 10


@dataclass(frozen=True)
class LogisticRegressionConfig:
    """Configuration for Logistic Regression."""

    params: Dict[str, Union[int, str]] = None
    search_space_params: Dict[str, List[Union[float, List[str], bool]]] = None


@dataclass(frozen=True)
class RandomForestConfig:
    """Configuration for Random Forest."""

    params: Dict[str, Union[int, str]] = None
    search_space_params: Dict[str, List[Union[int, float, List[str], bool]]] = None


@dataclass(frozen=True)
class LGBMConfig:
    """Configuration for LightGBM."""

    params: Dict[str, Union[int, float, str]] = None
    search_space_params: Dict[str, List[Union[int, float, bool]]] = None


@dataclass(frozen=True)
class XGBoostConfig:
    """Configuration for XGBoost."""

    params: Dict[str, str] = None
    search_space_params: Dict[str, List[Union[int, float, bool]]] = None


@dataclass(frozen=True)
class TrainFilesConfig:
    """Configuration for file paths."""

    historical_data_file_name: str
    preprocessed_dataset_target_file_name: str
    preprocessed_dataset_features_file_name: str
    train_set_file_name: str
    valid_set_file_name: str
    test_set_file_name: str


@dataclass(frozen=True)
class ModelRegistryConfig:
    """Configuration for model registry."""

    lr_registered_model_name: str = "default-lr-model"
    rf_registered_model_name: str = "default-rf-model"
    lgbm_registered_model_name: str = "default-lgbm-model"
    xgb_registered_model_name: str = "default-xgb-model"
    voting_ensemble_registered_model_name: str = "default-voting-ensemble-model"
    champion_model_name: str = "default-champion-model"


@dataclass(frozen=True)
class SupportedModelsConfig:
    """Configuration for supported models in ModelOptimizer.

    Note: When adding a new model, update the search space definition
    in the ModelOptimizer.generate_trial_params method.
    """

    models: tuple

    def is_supported(self, model_name: str) -> bool:
        """Check if a model name is in the supported models list.

        Args:
            model_name: Name of the model to validate.

        Returns:
            bool: True if model is supported, False otherwise.
        """
        return model_name in self.models


@dataclass(frozen=True)
class IncludedModelsConfig:
    """Configuration for included models."""

    include_logistic_regression: bool = True
    include_random_forest: bool = True
    include_lightgbm: bool = True
    include_xgboost: bool = True
    include_voting_ensemble: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    """Main configuration for training experiment."""

    description: str = "Default training experiment"
    logger: LoggerConfig = None
    data: TrainFeaturesConfig = None
    preprocessing: TrainPreprocessingConfig = None
    train_params: TrainParams = None
    logistic_regression: LogisticRegressionConfig = None
    random_forest: RandomForestConfig = None
    lightgbm: LGBMConfig = None
    xgboost: XGBoostConfig = None
    files: TrainFilesConfig = None
    modelregistry: ModelRegistryConfig = None
    included_models: IncludedModelsConfig = None
    supported_models: SupportedModelsConfig = None


def build_training_config(params: Dict[str, Any]) -> TrainingConfig:
    """Builds the TrainingConfig dataclass from the configuration parameters.

    Args:
        params (Dict[str, Any]): The configuration parameters.

    Returns:
        TrainingConfig: The training configuration as a dataclass instance.
    """
    included_models_params = params.get(
        "included_models", {}
    )  # Fallback to an empty dictionary

    # Build supported models tuple from modelregistry
    modelregistry_params = params["modelregistry"]
    models = (
        modelregistry_params["lr_registered_model_name"],
        modelregistry_params["rf_registered_model_name"],
        modelregistry_params["lgbm_registered_model_name"],
        modelregistry_params["xgb_registered_model_name"],
    )
    supported_models_config = SupportedModelsConfig(models=models)

    return TrainingConfig(
        description=params["description"],
        logger=map_to_dataclass(LoggerConfig, params["logger"]),
        data=map_to_dataclass(TrainFeaturesConfig, params["data"]),
        preprocessing=map_to_dataclass(
            TrainPreprocessingConfig, params.get("preprocessing", {})
        ),
        train_params=map_to_dataclass(TrainParams, params.get("train", {})),
        logistic_regression=map_to_dataclass(
            LogisticRegressionConfig, params.get("logistic_regression", {})
        ),
        random_forest=map_to_dataclass(
            RandomForestConfig, params.get("random_forest", {})
        ),
        lightgbm=map_to_dataclass(LGBMConfig, params.get("lightgbm", {})),
        xgboost=map_to_dataclass(XGBoostConfig, params.get("xgboost", {})),
        files=map_to_dataclass(TrainFilesConfig, params["files"]),
        modelregistry=map_to_dataclass(
            ModelRegistryConfig, params.get("modelregistry", {})
        ),
        included_models=map_to_dataclass(IncludedModelsConfig, included_models_params),
        supported_models=supported_models_config,
    )
