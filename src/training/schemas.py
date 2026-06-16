"""
Defines a class that loads parameters from config.yml file and dataclasses for
configuration sections. It includes validation logic to ensure required parameters
are present and correctly typed.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import PosixPath
from typing import Any, Dict, List, Optional, Union

import yaml

from src.utils.config_loader import map_to_dataclass
from src.utils.logger import LoggerConfig, get_logger

module_name: str = PosixPath(__file__).stem
logger = get_logger(module_name)


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
        try:
            int(self.params["data"]["split_rand_seed"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"split_rand_seed must be integer type. Got {self.params['data']['split_rand_seed']}"
            ) from exc

        if self.params["data"]["split_type"] not in ["random", "time"]:
            raise ValueError(
                f"split_type must be either 'random' or 'time'. Got {self.params['data']['split_type']}"
            )

        # Check beta value (primarily used to compare models)
        try:
            fbeta_score_beta_val = float(self.params["train"]["fbeta_score_beta_val"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"fbeta_score_beta_val must be float type. Got {self.params['train']['fbeta_score_beta_val']}"
            ) from exc
        if fbeta_score_beta_val <= 0:
            raise ValueError(
                f"fbeta_score_beta_val must be > 0. Got {fbeta_score_beta_val}"
            )

        # Validate task type (optional; defaults to binary) and the metrics that
        # are valid for it. comparison_metric is the optimization metric;
        # selection_metric is optional and falls back to comparison_metric.
        task_type = self.params["train"].get("task_type", "binary")
        supported_tasks = ("binary", "multiclass", "regression")
        if task_type not in supported_tasks:
            raise ValueError(
                f"task_type must be one of {supported_tasks}. Got {task_type}!"
            )

        classification_metrics = (
            "recall",
            "precision",
            "f1",
            "roc_auc",
            "fbeta_score",
            "average_precision",
            "log_loss",
            "brier_score",
        )
        regression_metrics = ("mae", "rmse", "r2", "mape")
        valid_metrics = (
            regression_metrics if task_type == "regression" else classification_metrics
        )
        for metric_key in ("comparison_metric", "selection_metric"):
            metric_value = self.params["train"].get(metric_key)
            # selection_metric is optional; empty/None means "use comparison_metric".
            if metric_key == "selection_metric" and not metric_value:
                continue
            if metric_value not in valid_metrics:
                raise ValueError(
                    f"{metric_key} must be one of {valid_metrics} for "
                    f"task_type='{task_type}'. Got {metric_value}!"
                )

        # Check if input split cutoff date (if split_type == "time") is in proper date format
        if self.params["data"]["split_type"] == "time" and (
            self.params["data"]["train_test_split_curoff_date"] == "none"
            or self.params["data"]["train_valid_split_curoff_date"] == "none"
        ):
            raise ValueError(
                f"train_test_split_curoff_date and train_valid_split_curoff_date must be a date (format {self.params['data']['split_date_col_format']}) or None if split type is 'random'."
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
    train_set_size: float
    pk_col_name: str
    class_col_name: str
    pos_class: str
    date_col_names: List[str]
    datetime_col_names: List[str]
    num_col_names: List[str]
    cat_col_names: List[str]
    historical_features: List[str]
    # Proportion of the (post-validation) training set kept as training data; the
    # complement becomes the dedicated calibration set used to calibrate the
    # champion and tune its decision threshold (disjoint from selection data).
    calibration_set_size: float = 0.8
    # Cutoff date for carving the calibration set when split_type == "time".
    train_calib_split_curoff_date: Optional[str] = None


@dataclass(frozen=True)
class TrainPreprocessingConfig:
    """Configuration for preprocessing training data."""

    num_features_imputer: str = "mean"
    num_features_scaler: str = "standard"
    scaler_params: Dict[str, Any] = None
    cat_features_imputer: str = "most_frequent"
    cat_features_ohe_handle_unknown: str = "error"
    cat_features_nans_replacement: str = "Unknown"
    cat_features_min_frequency: float = 0.01
    var_thresh_val: float = 0.0


@dataclass(frozen=True)
class TrainParams:
    """Main configuration for training parameters."""

    initiate_comet_project: bool = False
    experiment_tracker: str = "comet"
    project_name: str = "default-project"
    workspace_name: str = "comet-workspace-name"
    # Task the pipeline solves. Drives evaluator/metric selection and which
    # classification-only steps (label encoding, calibration, threshold tuning)
    # run. One of "binary", "multiclass", "regression".
    task_type: str = "binary"
    search_max_iters: int = 10
    parallel_jobs_count: int = 1
    exp_timout_secs: int = 3600
    # Number of stratified CV folds used to estimate the metric during the
    # search and champion selection. cross_val_folds > 1 enables CV; a value of
    # 1 (or less) falls back to the single train/valid holdout.
    cross_val_folds: int = 5
    fbeta_score_beta_val: float = 0.5
    # Metric the Optuna search optimizes (the "optimization metric").
    comparison_metric: str = "fbeta_score"
    # Metric used to rank candidates for the champion and gate deployment. Kept
    # separate from comparison_metric to break the tuning/selection circularity;
    # empty string means "fall back to comparison_metric".
    selection_metric: str = ""
    voting_rule: str = "soft"
    deployment_score_thresh: float = 0.8
    max_eval_experiments: int = 10
    decision_threshold: float = 0.5
    tune_decision_threshold: bool = False
    encoded_pos_class_label: int = 1


@dataclass(frozen=True)
class ModelSpecConfig:
    """Config-driven definition of one trainable model.

    The set of models to train is the YAML ``models:`` list; each entry maps to
    one of these. ``estimator`` is the importable class path (e.g.
    ``lightgbm.LGBMClassifier``) that the model factory resolves and
    instantiates, so adding a model is purely a new YAML entry -- no code change.

    Attributes:
        name: Registered model name and experiment key (e.g. ``lightgbm``).
        estimator: Importable estimator class path passed to the model factory.
        enabled: Whether this model is trained.
        params: Fixed estimator kwargs. A value of ``"${name}"`` is replaced at
            runtime with a data-derived value (e.g. ``scale_pos_weight``).
        search_space_params: Optuna search space; ``[min, max, log]`` for numeric
            or ``[[choices], false]`` for categorical params.
    """

    name: str
    estimator: str
    enabled: bool = False
    params: Dict[str, Any] = None
    search_space_params: Dict[str, List[Union[int, float, List[str], bool]]] = None


@dataclass(frozen=True)
class TrainFilesConfig:
    """Configuration for file paths."""

    historical_data_file_name: str
    preprocessed_dataset_target_file_name: str
    preprocessed_dataset_features_file_name: str
    train_set_file_name: str
    valid_set_file_name: str
    test_set_file_name: str
    calibration_set_file_name: str = "calibration.parquet"


@dataclass(frozen=True)
class ModelRegistryConfig:
    """Cross-cutting model registry names.

    Per-model registered names now come from each entry's ``name`` in the
    ``models:`` list; only the ensemble and champion names live here.
    """

    voting_ensemble_registered_model_name: str = "default-voting-ensemble-model"
    champion_model_name: str = "default-champion-model"


@dataclass(frozen=True)
class SupportedModelsConfig:
    """Names of the models defined in the config ``models:`` list.

    Used by ModelOptimizer to validate a ``registered_model_name``. The search
    space is consumed generically (ModelOptimizer.generate_trial_params), so a
    new model needs no code change here -- only a new ``models:`` entry.
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
class EnsembleConfig:
    """Configuration for the voting ensemble over the enabled base models."""

    enabled: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    """Main configuration for training experiment."""

    description: str = "Default training experiment"
    logger: LoggerConfig = None
    data: TrainFeaturesConfig = None
    preprocessing: TrainPreprocessingConfig = None
    train_params: TrainParams = None
    models: List[ModelSpecConfig] = None
    files: TrainFilesConfig = None
    modelregistry: ModelRegistryConfig = None
    ensemble: EnsembleConfig = None
    supported_models: SupportedModelsConfig = None


def build_training_config(params: Dict[str, Any]) -> TrainingConfig:
    """Builds the TrainingConfig dataclass from the configuration parameters.

    Args:
        params (Dict[str, Any]): The configuration parameters.

    Returns:
        TrainingConfig: The training configuration as a dataclass instance.
    """
    # Surface top-level section typos instead of silently loading section
    # defaults.
    known_sections = {
        "description",
        "logger",
        "data",
        "preprocessing",
        "train",
        "models",
        "files",
        "modelregistry",
        "ensemble",
        "inference",
    }
    unexpected_sections = set(params) - known_sections
    if unexpected_sections:
        logger.warning(
            "Unexpected top-level config section(s) (ignored): %s",
            ", ".join(sorted(unexpected_sections)),
        )

    # Build the config-driven model list and the supported-model names from it.
    model_specs = [
        map_to_dataclass(ModelSpecConfig, spec) for spec in params.get("models", [])
    ]
    # Each entry must set a name and an importable estimator path; surface a clear
    # error rather than failing cryptically when one is omitted (map_to_dataclass
    # leaves required fields as a sentinel for missing keys).
    for index, spec in enumerate(model_specs):
        if not isinstance(spec.name, str) or not spec.name:
            raise ValueError(
                f"Config `models:` entry #{index} must set a string `name`."
            )
        if not isinstance(spec.estimator, str) or not spec.estimator:
            raise ValueError(
                f"Config `models:` entry '{spec.name}' must set an `estimator` "
                "class path (e.g. 'lightgbm.LGBMClassifier')."
            )
    supported_models_config = SupportedModelsConfig(
        models=tuple(spec.name for spec in model_specs)
    )

    return TrainingConfig(
        description=params["description"],
        logger=map_to_dataclass(LoggerConfig, params["logger"]),
        data=map_to_dataclass(TrainFeaturesConfig, params["data"]),
        preprocessing=map_to_dataclass(
            TrainPreprocessingConfig, params.get("preprocessing", {})
        ),
        train_params=map_to_dataclass(TrainParams, params.get("train", {})),
        models=model_specs,
        files=map_to_dataclass(TrainFilesConfig, params["files"]),
        modelregistry=map_to_dataclass(
            ModelRegistryConfig, params.get("modelregistry", {})
        ),
        ensemble=map_to_dataclass(EnsembleConfig, params.get("ensemble", {})),
        supported_models=supported_models_config,
    )
