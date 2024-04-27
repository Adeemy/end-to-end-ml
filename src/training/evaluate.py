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
from src.training.utils.model import ModelChampionManager, ModelEvaluator
from src.utils.logger import get_console_logger
from src.utils.path import ARTIFACTS_DIR, DATA_DIR

load_dotenv()


###########################################################
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

    Returns:
        None.

    Raises:
        ValueError: if no successful experiments are found.
        ValueError: if the best model score on the test set is lower than the deployment threshold.
    """

    logger.info(f"Directory of training config file: {config_yaml_path}")

    # Experiment settings
    config = Config(config_path=config_yaml_path)
    project_name = config.params["train"]["comet_project_name"]
    workspace_name = config.params["train"]["comet_workspace_name"]
    class_col_name = config.params["data"]["class_col_name"]
    fbeta_score_beta_val = config.params["train"]["fbeta_score_beta_val"]
    calib_cv_folds = config.params["train"]["cross_val_folds"]
    comparison_metric_name = config.params["train"]["comparison_metric"]
    exp_keys_file_name = config.params["files"]["experiments_keys_file_name"]
    train_set_file_name = config.params["files"]["train_set_file_name"]
    test_set_file_name = config.params["files"]["test_set_file_name"]
    ve_registered_model_name = config.params["modelregistry"][
        "voting_ensemble_registered_model_name"
    ]
    champ_model_name = config.params["modelregistry"]["champion_model_name"]
    deployment_score_thresh = config.params["train"]["deployment_score_thresh"]

    # Import train and test sets to evaluate best model on test set
    # Note: it requires class labels to be encoded. An integration
    # test should be added to ensure the class labels are encoded.
    train_set = pd.read_parquet(
        data_dir / train_set_file_name,
    )

    test_set = pd.read_parquet(
        data_dir / test_set_file_name,
    )

    logger.info("Imported train and test sets.")

    #############################################
    # Rename comparison metric if it's fbeta_score to include beta value
    if comparison_metric_name == "fbeta_score":
        comparison_metric_name = f"f_{fbeta_score_beta_val}_score"

    # Import experiment keys from artifacts folder
    successful_exp_keys = pd.read_csv(
        f"{ARTIFACTS_DIR}/{exp_keys_file_name}.csv",
    )

    if successful_exp_keys.shape[0] == 0:
        raise ValueError(
            "No successful experiments found. Please check the experiment logs."
        )

    # Select the best performing model
    champ_model_manager = ModelChampionManager(champ_model_name=champ_model_name)
    best_model_name = champ_model_manager.select_best_performer(
        comet_project_name=project_name,
        comet_workspace_name=workspace_name,
        comparison_metric=f"valid_{comparison_metric_name}",
        comet_exp_keys=successful_exp_keys,
    )

    logger.info(f"Best candidate model is {best_model_name}.")

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

    best_model_evaluator = ModelEvaluator(
        comet_exp=best_model_exp_obj,
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

    best_model_exp_obj.log_metrics(test_scores)

    # Calibrate champ model before deployment
    training_features = train_set.drop(class_col_name, axis=1)
    training_class = np.array(train_set[class_col_name])
    calib_pipeline = champ_model_manager.calibrate_pipeline(
        train_features=training_features,
        train_class=training_class,
        preprocessor_step=best_model_pipeline.named_steps["preprocessor"],
        selector_step=best_model_pipeline.named_steps["selector"],
        model=best_model_pipeline.named_steps["classifier"],
        cv_folds=calib_cv_folds,
    )

    logger.info(f"Champion model {best_model_name} was calibrated.")

    # Log and register champion model (in Comet, model must be logged first)
    # Note: the best model should not be deployed in production if its score
    # on the test set is below minimum score. Otherwise, prevent deploying
    # the model by raising error preventing build job.
    best_model_test_score = test_scores.get(f"test_{comparison_metric_name}")
    if best_model_test_score >= deployment_score_thresh:
        champ_model_manager.log_and_register_champ_model(
            local_path=artifacts_dir,
            pipeline=calib_pipeline,
            exp_obj=best_model_exp_obj,
        )

        logger.info(f"Champion model was registered in {workspace_name} workspace.")

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
    console_logger = get_console_logger("evaluate_logger")

    console_logger.info("Models Evaluation Starts ...")

    main(
        config_yaml_path=args.config_yaml_path,
        comet_api_key=os.environ["COMET_API_KEY"],
        data_dir=DATA_DIR,
        artifacts_dir=ARTIFACTS_DIR,
        logger=console_logger,
    )
