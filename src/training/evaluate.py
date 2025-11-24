"""
Evaluates trained models on test set and registers champion model.

This script supports two evaluation workflows:
1. Integrated workflow: Called directly from train.py with experiment keys passed in-memory
2. Standalone workflow: Run independently with automatic Comet ML experiment discovery

For standalone evaluation, the script queries Comet ML directly to discover recent experiments
based on naming patterns (e.g., 'lightgbm_', 'random_forest_').

The script selects the best model based on validation performance, evaluates it on the
testing set, and registers it as the champion model if it meets the deployment threshold.
"""

import argparse
import logging
from pathlib import PosixPath
from typing import Optional

# IMPORTANT: Import comet_ml before sklearn to enable auto-logging
import comet_ml  # noqa: F401 # pylint: disable=unused-import
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.training.evaluation.champion import ModelChampionManager
from src.training.evaluation.orchestrator import create_evaluation_orchestrator
from src.training.evaluation.selector import ModelSelector
from src.training.schemas import Config, build_training_config
from src.training.tracking.experiment import get_tracker_credentials
from src.utils.config_loader import load_config
from src.utils.logger import get_console_logger
from src.utils.path import ARTIFACTS_DIR, DATA_DIR

load_dotenv()


def main(
    config_yaml_path: str,
    data_dir: PosixPath,
    artifacts_dir: PosixPath,
    logger: logging.Logger,
    experiment_keys: Optional[pd.DataFrame] = None,
) -> tuple[str, dict]:
    """Evaluates best model on test set and registers as champion.

    Note: This script requires trained models to exist in Comet ML.
    If no experiments are found, run training first: 'make train'

    Args:
        config_yaml_path: Path to training config YAML file.
        data_dir: Path to data directory.
        artifacts_dir: Path to artifacts directory.
        logger: Logger object.
        experiment_keys: Optional DataFrame with experiment keys from training.
                        If None, will query Comet ML directly.

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

    # Use experiment keys passed from training or query Comet ML directly
    if experiment_keys is not None:
        logger.info("Using experiment keys passed from training.")

    # Load datasets
    train_set = pd.read_parquet(data_dir / training_config.files.train_set_file_name)
    valid_set = pd.read_parquet(data_dir / training_config.files.valid_set_file_name)
    test_set = pd.read_parquet(data_dir / training_config.files.test_set_file_name)
    logger.info("Loaded train, validation, and test sets")

    # Prepare data splits
    class_col = training_config.data.class_col_name
    train_features = train_set.drop(class_col, axis=1)
    train_class = np.array(train_set[class_col])
    valid_features = valid_set.drop(class_col, axis=1)
    valid_class = np.array(valid_set[class_col])
    test_features = test_set.drop(class_col, axis=1)
    test_class = np.array(test_set[class_col])

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
    )

    champion_manager = ModelChampionManager(
        champ_model_name=training_config.modelregistry.champion_model_name
    )

    # Determine comparison metric name
    comparison_metric = training_config.train_params.comparison_metric
    if comparison_metric == "fbeta_score":
        comparison_metric = (
            f"f_{training_config.train_params.fbeta_score_beta_val}_score"
        )

    # Add valid_ prefix to ensure the model selection is based on validation set
    valid_comparison_metric = f"valid_{comparison_metric}"

    model_selector = ModelSelector(
        project_name=training_config.train_params.project_name,
        workspace_name=training_config.train_params.workspace_name,
        comparison_metric=valid_comparison_metric,
    )

    # Run evaluation workflow
    deployment_threshold = float(training_config.train_params.deployment_score_thresh)

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
        champion_manager=champion_manager,
        comparison_metric_name=comparison_metric,
        deployment_threshold=deployment_threshold,
        cv_folds=training_config.train_params.cross_val_folds,
        experiment_keys=experiment_keys,
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

    args = parser.parse_args()

    # Get logger
    module_name: str = PosixPath(__file__).stem
    console_logger = get_console_logger(module_name)
    console_logger.info("Model Evaluation on Test Set Starts ...")

    # Run evaluation (without experiment_keys will query Comet ML directly)
    champ_name, metrics = main(
        config_yaml_path=args.config_yaml_path,
        data_dir=DATA_DIR,
        artifacts_dir=ARTIFACTS_DIR,
        logger=console_logger,
    )

    console_logger.info("Champion model: %s", champ_name)
    console_logger.info("Test metrics: %s", metrics)
