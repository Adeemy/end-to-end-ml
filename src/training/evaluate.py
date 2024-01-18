"""
This script evaluates trained models and selects 
the best model based on performance on validation 
set. The best model is then evaluated on testing 
set to assess its generalization capability and it
will be registered as champion model only if its score
on the test set is better than a required threshold value.
"""

import os
import sys
from pathlib import PosixPath

import joblib
import numpy as np
import pandas as pd
from comet_ml import ExistingExperiment
from dotenv import load_dotenv
from utils.config import Config
from utils.model import ModelEvaluator, PrepChampModel
from utils.path import ARTIFACTS_DIR, DATA_DIR

load_dotenv()


###########################################################
def main(
    config_yaml_abs_path: str,
    comet_api_key: str,
    data_dir: PosixPath,
    artifacts_dir: PosixPath,
):
    # Experiment settings
    config = Config(config_path=config_yaml_abs_path)
    COMET_API_KEY = comet_api_key
    COMET_PROJECT_NAME = config.params["train"]["params"]["comet_project_name"]
    COMET_WORKSPACE_NAME = config.params["train"]["params"]["comet_workspace_name"]
    CLASS_COL_NAME = config.params["data"]["params"]["class_col_name"]
    F_BETA_SCORE_BETA_VAL = config.params["train"]["params"]["fbeta_score_beta_val"]
    CALIB_CV_FOLDS = config.params["train"]["params"]["cross_val_folds"]
    COMPARISON_METRIC = config.params["train"]["params"]["comparison_metric"]
    EXP_KEY_FILE_NAME = config.params["files"]["params"]["experiments_keys_file_name"]
    TRAIN_FILE_NAME = config.params["files"]["params"]["train_set_file_name"]
    TEST_FILE_NAME = config.params["files"]["params"]["test_set_file_name"]
    VOTING_ENSEMBLE_REGISTERED_MODEL_NAME = config.params["modelregistry"]["params"][
        "voting_ensemble_registered_model_name"
    ]
    CHAMPION_MODEL_NAME = config.params["modelregistry"]["params"][
        "champion_model_name"
    ]
    DEPLOYMENT_SCORE_THRESH = config.params["train"]["params"][
        "deployment_score_thresh"
    ]

    # Import train and test sets to evaluate best model on test set
    # Note: it requires class labels to be encoded.
    train_set = pd.read_parquet(
        data_dir / TRAIN_FILE_NAME,
    )

    test_set = pd.read_parquet(
        data_dir / TEST_FILE_NAME,
    )

    #############################################
    # Rename comparison metric if it's fbeta_score to include beta value
    if COMPARISON_METRIC == "fbeta_score":
        COMPARISON_METRIC = f"f_{F_BETA_SCORE_BETA_VAL}_score"

    # Import experiment keys from artifacts folder
    successful_exp_keys = pd.read_csv(
        f"{ARTIFACTS_DIR}/{EXP_KEY_FILE_NAME}.csv",
    )

    # Select the best performing model
    prep_champ_model = PrepChampModel()
    best_model_name = prep_champ_model.select_best_performer(
        comet_project_name=COMET_PROJECT_NAME,
        comet_workspace_name=COMET_WORKSPACE_NAME,
        comparison_metric=f"valid_{COMPARISON_METRIC}",
        comet_exp_keys=successful_exp_keys,
    )

    # Create ExistingExperiment object to allow appending logging new metrics
    best_model_exp_key = successful_exp_keys.loc[
        successful_exp_keys["0"] == best_model_name, "1"
    ].iloc[0]
    best_model_exp_obj = ExistingExperiment(
        api_key=COMET_API_KEY, experiment_key=best_model_exp_key
    )

    #############################################
    # Assess generalization capability of the best performer on test set
    # Note: test set was not exposed to any model during training or
    # evaluation to ensure all models are independent of the test set.
    best_model_pipeline = joblib.load(f"{ARTIFACTS_DIR}/{best_model_name}.pkl")

    best_model_evaluator = ModelEvaluator(
        comet_exp=best_model_exp_obj,
        pipeline=best_model_pipeline,
        train_features=train_set.drop(CLASS_COL_NAME, axis=1),
        train_class=np.array(train_set[CLASS_COL_NAME]),
        valid_features=test_set.drop(CLASS_COL_NAME, axis=1),
        valid_class=np.array(test_set[CLASS_COL_NAME]),
        fbeta_score_beta=F_BETA_SCORE_BETA_VAL,
        is_voting_ensemble=True
        if best_model_name == VOTING_ENSEMBLE_REGISTERED_MODEL_NAME
        else False,
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

    # Calibrate champ model
    training_features = train_set.drop(CLASS_COL_NAME, axis=1)
    training_class = np.array(train_set[CLASS_COL_NAME])
    calib_pipeline = prep_champ_model.calibrate_pipeline(
        train_features=training_features,
        train_class=training_class,
        preprocessor_step=best_model_pipeline.named_steps["preprocessor"],
        selector_step=best_model_pipeline.named_steps["selector"],
        model=best_model_pipeline.named_steps["classifier"],
        cv_folds=CALIB_CV_FOLDS,
    )

    # Log and register champion model (in Comet, model must be logged first)
    # Note: the best model should not be deployed in production if its score
    # on the test set is below minimum score. Otherwise, prevent deploying
    # the model by raising error preventing build job.
    BEST_MODEL_TEST_SCORE = test_scores.get(f"test_{COMPARISON_METRIC}")
    if BEST_MODEL_TEST_SCORE >= DEPLOYMENT_SCORE_THRESH:
        prep_champ_model.log_and_register_champ_model(
            local_path=artifacts_dir,
            champ_model_name=CHAMPION_MODEL_NAME,
            pipeline=calib_pipeline,
            exp_obj=best_model_exp_obj,
        )

        # Save the champion model in local direcotry to be packaged in docker container
        joblib.dump(best_model_pipeline, f"{ARTIFACTS_DIR}/{CHAMPION_MODEL_NAME}.pkl")

    else:
        raise ValueError(
            f"Best model score is {BEST_MODEL_TEST_SCORE}, which is lower than deployment threshold {DEPLOYMENT_SCORE_THRESH}."
        )


###########################################################
if __name__ == "__main__":
    main(
        config_yaml_abs_path=sys.argv[1],
        comet_api_key=os.environ["COMET_API_KEY"],
        data_dir=DATA_DIR,
        artifacts_dir=ARTIFACTS_DIR,
    )
