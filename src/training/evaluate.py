"""
Evaluates trained models and selects the best model based on
performance on validation set. The best model is then evaluated
on testing set to assess its generalization capability and it
will be registered as champion model only if its score on the
test set is better than a required threshold value.
"""

import argparse
import logging
import logging.config
import os
from pathlib import PosixPath

import joblib
import numpy as np
import pandas as pd
from comet_ml import ExistingExperiment
from dotenv import load_dotenv

from src.training.utils.config import Config
from src.training.utils.experiment_tracker import CometExperimentTracker
from src.training.utils.model import ModelChampionManager, ModelEvaluator
from src.utils.logger import get_console_logger
from src.utils.path import ARTIFACTS_DIR, DATA_DIR

load_dotenv()


def select_best_model(
    config_yaml_path: str,
    successful_exp_keys: pd.DataFrame,
    champ_model_manager: ModelChampionManager,
    comparison_metric_name: str,
    project_name: str,
    workspace_name: str,
) -> str:
    """Selects the best model based on performance on validation set.

    Args:
        config_yaml_path (str): path to the config yaml file.
        successful_exp_keys (pd.DataFrame): dataframe containing successful experiments keys.
        champ_model_manager (ModelChampionManager): ModelChampionManager object.
        comparison_metric_name (str): name of the comparison metric.
        project_name (str): name of the Comet project.
        workspace_name (str): name of the Comet workspace.

    Returns:
        best_model_name (str): name of the best model.
    """

    # Get configuration parameters
    config = Config(config_path=config_yaml_path)
    project_name = config.params["train"]["project_name"]
    workspace_name = config.params["train"]["workspace_name"]

    if successful_exp_keys.shape[0] == 0:
        raise ValueError(
            "No successful experiments found. Please check the experiment logs."
        )

    # Select the best performing model
    best_model_name = champ_model_manager.select_best_performer(
        project_name=project_name,
        workspace_name=workspace_name,
        comparison_metric=f"valid_{comparison_metric_name}",
        comet_exp_keys=successful_exp_keys,
    )

    return best_model_name


def evaluate_best_model(
    best_model_exp_obj: ExistingExperiment,
    best_model_pipeline: ModelEvaluator,
    train_set: pd.DataFrame,
    test_set: pd.DataFrame,
    class_col_name: str,
    fbeta_score_beta_val: float,
    best_model_name: str,
    ve_registered_model_name: str,
) -> dict:
    """Evaluate the best model on testing set to assess its generalization capability.

    Args:
        best_model_exp_obj (ExistingExperiment): ExistingExperiment object.
        best_model_pipeline (ModelEvaluator): ModelEvaluator object.
        train_set (pd.DataFrame): training set.
        test_set (pd.DataFrame): testing set.
        class_col_name (str): name of the class column.
        fbeta_score_beta_val (float): beta value for fbeta score.
        best_model_name (str): name of the best model.
        ve_registered_model_name (str): name of the voting ensemble registered model.

    Returns:
        test_scores (dict): dictionary containing test scores.
    """

    # Wrap Comet experiment with tracker abstraction
    tracker = CometExperimentTracker(experiment=best_model_exp_obj)

    best_model_evaluator = ModelEvaluator(
        tracker=tracker,
        pipeline=best_model_pipeline,
        train_features=train_set.drop(class_col_name, axis=1),
        train_class=np.array(train_set[class_col_name]),
        valid_features=test_set.drop(class_col_name, axis=1),
        valid_class=np.array(test_set[class_col_name]),
        fbeta_score_beta=fbeta_score_beta_val,
        is_voting_ensemble=(
            True if best_model_name == ve_registered_model_name else False
        ),
    )

    # Evaluate best model on testing set to assess its generalization capability
    (
        _,
        test_scores,
    ) = best_model_evaluator.evaluate_model_perf(
        class_encoder=None,
    )

    test_scores = best_model_evaluator.convert_metrics_from_df_to_dict(
        scores=test_scores, prefix="test_"
    )

    return test_scores


def main(
    config_yaml_path: str,
    comet_api_key: str,
    data_dir: PosixPath,
    artifacts_dir: PosixPath,
    logger: logging.Logger,
) -> None:
    """Selects the best model based on performance on validation set and evaluates it on testing set.

    Args:
        config_yaml_path (str): path to the config yaml file.
        comet_api_key (str): Comet API key.
        data_dir (PosixPath): path to the data directory.
        artifacts_dir (PosixPath): path to the artifacts directory.
        logger (logging.Logger): logger object.

    Raises:
        ValueError: if no successful experiments are found.
        ValueError: if the best model score on the test set is lower than the deployment threshold.
    """

    logger.info("Directory of training config file: %s", config_yaml_path)

    # Get configuration parameters
    config = Config(config_path=config_yaml_path)
    project_name = config.params["train"]["project_name"]
    workspace_name = config.params["train"]["workspace_name"]
    class_col_name = config.params["data"]["class_col_name"]
    fbeta_score_beta_val = config.params["train"]["fbeta_score_beta_val"]
    calib_cv_folds = config.params["train"]["cross_val_folds"]
    comparison_metric_name = config.params["train"]["comparison_metric"]
    exp_keys_file_name = config.params["files"]["experiments_keys_file_name"]
    train_set_file_name = config.params["files"]["train_set_file_name"]
    valid_set_file_name = config.params["files"]["valid_set_file_name"]
    test_set_file_name = config.params["files"]["test_set_file_name"]
    ve_registered_model_name = config.params["modelregistry"][
        "voting_ensemble_registered_model_name"
    ]
    champ_model_name = config.params["modelregistry"]["champion_model_name"]
    deployment_score_thresh = config.params["train"]["deployment_score_thresh"]

    # Rename comparison metric if it's fbeta_score to include beta value
    if comparison_metric_name == "fbeta_score":
        comparison_metric_name = f"f_{fbeta_score_beta_val}_score"

    # Import train and test sets to evaluate best model on test set
    # Note: it requires class labels to be encoded. An integration
    # test should be added to ensure the class labels are encoded.
    train_set = pd.read_parquet(
        data_dir / train_set_file_name,
    )

    valid_set = pd.read_parquet(
        data_dir / valid_set_file_name,
    )

    test_set = pd.read_parquet(
        data_dir / test_set_file_name,
    )

    logger.info("Imported train and test sets.")

    #############################################
    # Select the best model based on performance on validation set
    successful_exp_keys = pd.read_csv(
        f"{ARTIFACTS_DIR}/{exp_keys_file_name}.csv",
    )

    # Create champion model manager (tracker will be added later after selecting best model)
    champ_model_manager = ModelChampionManager(champ_model_name=champ_model_name)
    best_model_name = select_best_model(
        config_yaml_path=config_yaml_path,
        successful_exp_keys=successful_exp_keys,
        champ_model_manager=champ_model_manager,
        comparison_metric_name=comparison_metric_name,
        project_name=project_name,
        workspace_name=workspace_name,
    )

    logger.info("Best candidate model is %s.", best_model_name)

    # Create ExistingExperiment object to allow appending logging new metrics
    best_model_exp_key = successful_exp_keys.loc[
        successful_exp_keys["0"] == best_model_name, "1"
    ].iloc[0]
    best_model_exp_obj = ExistingExperiment(
        api_key=comet_api_key, experiment_key=best_model_exp_key
    )

    #############################################
    # Assess generalization capability of the best performer on test set
    # Note: test set was not exposed to any model during training or
    # evaluation to ensure all models are independent of the test set.
    best_model_pipeline = joblib.load(f"{ARTIFACTS_DIR}/{best_model_name}.pkl")
    test_scores = evaluate_best_model(
        best_model_exp_obj=best_model_exp_obj,
        best_model_pipeline=best_model_pipeline,
        train_set=train_set,
        test_set=test_set,
        class_col_name=class_col_name,
        fbeta_score_beta_val=fbeta_score_beta_val,
        best_model_name=best_model_name,
        ve_registered_model_name=ve_registered_model_name,
    )
    best_model_exp_obj.log_metrics(test_scores)

    # Calibrate champ model before deployment
    valid_features = valid_set.drop(class_col_name, axis=1)
    valid_class = np.array(valid_set[class_col_name])
    calib_pipeline = champ_model_manager.calibrate_pipeline(
        valid_features=valid_features,
        valid_class=valid_class,
        fitted_pipeline=best_model_pipeline,
        cv_folds=calib_cv_folds,
    )

    logger.info("Champion model %s was calibrated.", best_model_name)

    # Log and register champion model (in Comet, model must be logged first)
    # Note: the best model should not be deployed in production if its score
    # on the test set is below minimum score. Otherwise, prevent deploying
    # the model by raising error preventing build job.
    best_model_test_score = test_scores.get(f"test_{comparison_metric_name}")
    if best_model_test_score >= deployment_score_thresh:
        # Create tracker for champion model and update manager
        champ_tracker = CometExperimentTracker(experiment=best_model_exp_obj)
        champ_model_manager.tracker = champ_tracker

        champ_model_manager.log_and_register_champ_model(
            local_path=artifacts_dir,
            pipeline=calib_pipeline,
        )

        logger.info("Champion model was registered in %s workspace.", workspace_name)

        # Save the champion model in local direcotry to be packaged in docker container
        joblib.dump(best_model_pipeline, f"{ARTIFACTS_DIR}/{champ_model_name}.pkl")

    else:
        raise ValueError(
            f"""Best model score is {best_model_test_score}, which is lower than
                         deployment threshold {deployment_score_thresh}."""
        )


###########################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml_path",
        type=str,
        default="./config.yml",
        help="Path to the configuration yaml file.",
    )
    parser.add_argument(
        "--logger_path",
        type=str,
        default="./logger.conf",
        help="Path to the logger configuration file.",
    )

    args = parser.parse_args()

    # Get the logger objects by name
    module_name: str = PosixPath(__file__).stem
    console_logger = get_console_logger(module_name)
    console_logger.info("Models Evaluation Starts ...")

    main(
        config_yaml_path=args.config_yaml_path,
        comet_api_key=os.environ["COMET_API_KEY"],
        data_dir=DATA_DIR,
        artifacts_dir=ARTIFACTS_DIR,
        logger=console_logger,
    )
