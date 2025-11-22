"""
Evaluates trained models on test set and registers champion model.

This script can be run standalone or called from train.py with experiment keys.
It selects the best model based on validation performance, evaluates it on the
test set, and registers it as the champion model if it meets the deployment threshold.
"""

import argparse
import logging
import os
from pathlib import PosixPath
from typing import Optional

# IMPORTANT: Import comet_ml before sklearn to enable auto-logging
import comet_ml  # noqa: F401 # pylint: disable=unused-import
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.training.utils.config.config import Config, build_training_config
from src.training.utils.evaluation.champion import ModelChampionManager
from src.training.utils.evaluation.orchestrator import create_evaluation_orchestrator
from src.training.utils.evaluation.selector import ModelSelector
from src.utils.config_loader import load_config
from src.utils.logger import get_console_logger
from src.utils.path import ARTIFACTS_DIR, DATA_DIR

load_dotenv()


def main(
    config_yaml_path: str,
    comet_api_key: str,
    data_dir: PosixPath,
    artifacts_dir: PosixPath,
    logger: logging.Logger,
    experiment_keys: Optional[pd.DataFrame] = None,
) -> tuple[str, dict]:
    """Evaluates best model on test set and registers as champion.

    Args:
        config_yaml_path: Path to training config YAML file.
        comet_api_key: Comet API key.
        data_dir: Path to data directory.
        artifacts_dir: Path to artifacts directory.
        logger: Logger object.
        experiment_keys: Optional DataFrame with experiment keys from training.
                        If None, will load from CSV file.

    Returns:
        Tuple of (champion_model_name, test_metrics).

    Raises:
        ValueError: If test score is below deployment threshold.
    """
    logger.info("Directory of training config file: %s", config_yaml_path)

    # Load configuration using new config system
    training_config = load_config(
        config_class=Config,
        builder_func=build_training_config,
        config_path=config_yaml_path,
    )

    # Load experiment keys (from train.py or from file)
    if experiment_keys is None:
        exp_keys_path = (
            artifacts_dir / f"{training_config.files.experiments_keys_file_name}.csv"
        )
        experiment_keys = pd.read_csv(exp_keys_path)
        logger.info("Loaded experiment keys from: %s", exp_keys_path)
    else:
        logger.info("Using experiment keys passed from training")

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

    # Add 'valid_' prefix for validation metrics
    comparison_metric = f"valid_{comparison_metric}"

    model_selector = ModelSelector(
        project_name=training_config.train_params.project_name,
        workspace_name=training_config.train_params.workspace_name,
        comparison_metric=comparison_metric,
        comet_api_key=comet_api_key,
    )

    # Run evaluation workflow
    champion_name, test_metrics = test_evaluator.run_evaluation_workflow(
        comet_api_key=comet_api_key,
        experiment_keys=experiment_keys,
        model_selector=model_selector,
        valid_features=valid_features,
        valid_class=valid_class,
        champion_manager=champion_manager,
        comparison_metric_name=comparison_metric,
        deployment_threshold=training_config.train_params.deployment_score_thresh,
        cv_folds=training_config.train_params.cross_val_folds,
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

    # Run evaluation (without experiment_keys will load from file)
    champ_name, metrics = main(
        config_yaml_path=args.config_yaml_path,
        comet_api_key=os.environ["COMET_API_KEY"],
        data_dir=DATA_DIR,
        artifacts_dir=ARTIFACTS_DIR,
        logger=console_logger,
    )

    console_logger.info("Champion model: %s", champ_name)
    console_logger.info("Test metrics: %s", metrics)
