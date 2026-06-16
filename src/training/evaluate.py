"""
Model selection and champion evaluation pipeline.

This script implements proper ML evaluation practices by separating model selection
from final performance assessment:

1. **Model Selection**: Ranks all trained models based on validation set performance
2. **Test Evaluation**: Evaluates ONLY the best selected model on the held-out test set to determine final performance
3. **Champion Registration**: Registers the model as champion if it meets deployment threshold based on test performance

Workflow Details:
- Integrated workflow: Called from train.py with experiment keys passed in-memory
- Standalone workflow: Discovers experiments automatically via tracking backend
- Selection Criteria: Uses validation metrics (e.g., 'valid_f1_score') to avoid test contamination
- Final Assessment: Single test evaluation on the selected model only

Supports both MLflow (default) and Comet ML experiment discovery.
"""

import argparse
import logging
import os
from pathlib import PosixPath
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# IMPORTANT: Set ENABLE_COMET_LOGGING=true in environment if using Comet ML tracker
# This ensures comet_ml is imported before other ML libraries for proper auto-logging
if os.getenv("ENABLE_COMET_LOGGING", "false").lower() == "true":
    import comet_ml  # pylint: disable=unused-import

import numpy as np
import pandas as pd

from src.training.evaluation import metrics
from src.training.evaluation.champion import ModelChampionManager
from src.training.evaluation.orchestrator import create_evaluation_orchestrator
from src.training.evaluation.selector import ModelSelector
from src.training.schemas import Config, build_training_config
from src.training.tracking.experiment import get_tracker_credentials
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.path import (
    ARTIFACTS_DIR,
    DATA_DIR,
    TRAINING_EXPERIMENTS_FILE,
    encoded_split_path,
)

module_name: str = PosixPath(__file__).stem
console_logger = get_logger(module_name)


def _resolve_mlflow_run(
    run_id: str, artifacts_dir: PosixPath, logger: logging.Logger
) -> pd.DataFrame:
    """Materializes the model from a specific MLflow run for evaluation.

    Looks up the registered model version produced by ``run_id``, loads it, and
    writes it to the local artifacts dir so the evaluation workflow scores that
    exact run instead of the most recent one.

    Args:
        run_id: MLflow run id of the training run to evaluate.
        artifacts_dir: Directory where candidate model pkls are read from.
        logger: Logger object.

    Returns:
        pd.DataFrame: a single ``[model_name, run_id]`` row of experiment keys.

    Raises:
        ValueError: If no registered model is associated with ``run_id``.
    """
    import joblib  # pylint: disable=import-outside-toplevel
    import mlflow.sklearn  # pylint: disable=import-outside-toplevel
    from mlflow.tracking import MlflowClient  # pylint: disable=import-outside-toplevel

    versions = MlflowClient().search_model_versions(f"run_id='{run_id}'")
    if not versions:
        raise ValueError(
            f"No registered model found for MLflow run_id '{run_id}'. "
            "Check the run id (browse runs with `make view_mlflow`)."
        )
    version = versions[0]
    model = mlflow.sklearn.load_model(f"models:/{version.name}/{version.version}")
    local_path = artifacts_dir / f"{version.name}.pkl"
    joblib.dump(model, local_path)
    logger.info(
        "Evaluating MLflow run %s: model '%s' v%s materialized to %s",
        run_id,
        version.name,
        version.version,
        local_path,
    )
    return pd.DataFrame([[version.name, run_id]])


def main(
    config_yaml_path: str,
    data_dir: PosixPath,
    artifacts_dir: PosixPath,
    logger: logging.Logger,
    experiment_keys: Optional[pd.DataFrame] = None,
    run_id: Optional[str] = None,
) -> tuple[str, dict]:
    """Evaluates best model on test set and registers as champion.

    By default this evaluates the most recent training run (the models recorded
    in the ``training_experiments.json`` written by train.py). Pass ``run_id`` to
    evaluate a specific MLflow run instead.

    Args:
        config_yaml_path: Path to training config YAML file.
        data_dir: Path to data directory.
        artifacts_dir: Path to artifacts directory.
        logger: Logger object.
        experiment_keys: Optional DataFrame with experiment keys passed in-process
                        from training. If None, the keys are resolved from
                        ``run_id``, then the last-run keys file, then (Comet only)
                        remote discovery.
        run_id: Optional MLflow run id to evaluate a specific run instead of the
                most recent one.

    Returns:
        Tuple of (champion_model_name, test_metrics).

    Raises:
        ValueError: If test score is below deployment threshold or no experiments found.
    """
    logger.info(
        "Directory of training config file: %s", config_yaml_path
    )  # Load configuration using new config system
    training_config = load_config(
        config_class=Config,
        builder_func=build_training_config,
        config_path=config_yaml_path,
    )

    # Resolve which trained models to evaluate. Priority:
    #   1. keys passed in-process from train.py (--run_evaluation),
    #   2. the keys file written by the most recent `make train`,
    #   3. Comet workspace discovery (only when the tracker is Comet).
    # For MLflow with no keys file we fail fast rather than querying Comet.
    tracker_name = training_config.train_params.experiment_tracker.lower()
    if experiment_keys is not None:
        logger.info("Using experiment keys passed in-process from training.")
    elif run_id:
        experiment_keys = _resolve_mlflow_run(run_id, artifacts_dir, logger)
    else:
        keys_path = artifacts_dir / TRAINING_EXPERIMENTS_FILE
        if keys_path.exists():
            experiment_keys = pd.read_json(keys_path, orient="values", dtype=str)
            logger.info(
                "Evaluating the most recent training run: loaded %d model key(s) from %s",
                len(experiment_keys),
                keys_path,
            )
        elif tracker_name == "comet":
            logger.info(
                "No %s found; discovering recent experiments via the Comet workspace.",
                TRAINING_EXPERIMENTS_FILE,
            )  # experiment_keys stays None -> ModelSelector (Comet) discovery
        else:
            raise ValueError(
                f"No '{TRAINING_EXPERIMENTS_FILE}' found in {artifacts_dir}. Run "
                f"'make train' first (experiment_tracker='{tracker_name}' evaluates "
                "the most recent local training run; it has no remote discovery)."
            )

    # Load the feature-selected, label-encoded splits written by train.py
    # (separate "*_encoded.parquet" files; the canonical splits are left intact).
    train_set = pd.read_parquet(
        encoded_split_path(data_dir, training_config.files.train_set_file_name)
    )
    valid_set = pd.read_parquet(
        encoded_split_path(data_dir, training_config.files.valid_set_file_name)
    )
    calib_set = pd.read_parquet(
        encoded_split_path(data_dir, training_config.files.calibration_set_file_name)
    )
    test_set = pd.read_parquet(
        encoded_split_path(data_dir, training_config.files.test_set_file_name)
    )
    logger.info("Loaded train, validation, calibration, and test sets")

    # Prepare data splits
    class_col = training_config.data.class_col_name
    train_features = train_set.drop(class_col, axis=1)
    train_class = np.array(train_set[class_col])
    valid_features = valid_set.drop(class_col, axis=1)
    valid_class = np.array(valid_set[class_col])
    calib_features = calib_set.drop(class_col, axis=1)
    calib_class = np.array(calib_set[class_col])
    test_features = test_set.drop(class_col, axis=1)
    test_class = np.array(test_set[class_col])

    # Task type and the model-preference order used by the 1-SE selection rule
    # (simplest/cheapest first, falling back to the order models are trained in).
    task_type = training_config.train_params.task_type
    model_preference = [
        training_config.modelregistry.lr_registered_model_name,
        training_config.modelregistry.rf_registered_model_name,
        training_config.modelregistry.lgbm_registered_model_name,
        training_config.modelregistry.xgb_registered_model_name,
        training_config.modelregistry.voting_ensemble_registered_model_name,
    ]

    # Create orchestrators
    test_evaluator = create_evaluation_orchestrator(
        tracker_type=training_config.train_params.experiment_tracker,
        train_features=train_features,
        train_class=train_class,
        test_features=test_features,
        test_class=test_class,
        artifacts_path=str(artifacts_dir),
        fbeta_score_beta=training_config.train_params.fbeta_score_beta_val,
        voting_ensemble_name=training_config.modelregistry.voting_ensemble_registered_model_name,
        decision_threshold=training_config.train_params.decision_threshold,
        tune_decision_threshold=training_config.train_params.tune_decision_threshold,
        encoded_pos_class_label=training_config.train_params.encoded_pos_class_label,
        task_type=task_type,
        cv_folds=training_config.train_params.cross_val_folds,
        model_preference=model_preference,
        random_seed=int(training_config.data.split_rand_seed),
    )

    champion_manager = ModelChampionManager(
        champ_model_name=training_config.modelregistry.champion_model_name
    )

    # Resolve the SELECTION metric (decoupled from the optimization metric). It
    # falls back to comparison_metric when unset, and is mapped to the row name
    # the evaluators emit (macro-averaged for multi-class) so selection and the
    # deployment gate find the metric for every task type.
    selection_metric = (
        training_config.train_params.selection_metric
        or training_config.train_params.comparison_metric
    )
    comparison_metric = metrics.selection_metric_row_name(
        selection_metric, task_type, training_config.train_params.fbeta_score_beta_val
    )

    # Add valid_ prefix to ensure the model selection is based on validation set
    valid_comparison_metric = f"valid_{comparison_metric}"

    # Only build the (Comet-backed) ModelSelector when we actually need remote
    # discovery, i.e. no experiment keys were resolved above. Constructing it
    # calls comet_ml.login(), so creating it unconditionally would hit Comet
    # (and fail on networks without access) even for MLflow runs that already
    # have their keys.
    model_selector = None
    if experiment_keys is None:
        model_selector = ModelSelector(
            project_name=training_config.train_params.project_name,
            workspace_name=training_config.train_params.workspace_name,
            comparison_metric=valid_comparison_metric,
        )

    # Run evaluation workflow
    deployment_threshold = float(training_config.train_params.deployment_score_thresh)
    max_eval_experiments = int(training_config.train_params.max_eval_experiments)

    # Setup experiment kwargs for evaluation tracking
    experiment_tracker_type = training_config.train_params.experiment_tracker
    experiment_kwargs = {
        "experiment_tracker_type": experiment_tracker_type,
        "project_name": training_config.train_params.project_name,
        "workspace_name": training_config.train_params.workspace_name,
    }

    # Add tracker-specific credentials using the credential provider
    try:
        credentials = get_tracker_credentials(experiment_tracker_type)
        experiment_kwargs.update(credentials)
    except ValueError as e:
        logger.error(
            "Could not get credentials for tracker %s. Error -> %s",
            experiment_tracker_type,
            e,
        )
        raise

    champion_name, test_metrics = test_evaluator.run_evaluation_workflow(
        model_selector=model_selector,
        valid_features=valid_features,
        valid_class=valid_class,
        calibration_features=calib_features,
        calibration_class=calib_class,
        champion_manager=champion_manager,
        comparison_metric_name=comparison_metric,
        deployment_threshold=deployment_threshold,
        experiment_keys=experiment_keys,
        max_eval_experiments=max_eval_experiments,
        **experiment_kwargs,
    )

    logger.info("Champion model %s deployed successfully", champion_name)

    return champion_name, test_metrics


###########################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained models on test set and register champion model."
    )
    parser.add_argument(
        "--config_yaml_path",
        type=str,
        default="./src/config/training-config.yml",
        help="Path to the training configuration YAML file.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="MLflow run id to evaluate a specific run. Defaults to the most "
        "recent training run (recorded by train.py).",
    )

    args = parser.parse_args()

    console_logger.info("Model Evaluation on Test Set Starts ...")

    # By default evaluates the most recent training run; --run_id targets a specific one.
    champ_name, metrics = main(
        config_yaml_path=args.config_yaml_path,
        data_dir=DATA_DIR,
        artifacts_dir=ARTIFACTS_DIR,
        logger=console_logger,
        run_id=args.run_id,
    )

    console_logger.info("Champion model: %s", champ_name)
    console_logger.info("Test metrics: %s", metrics)
