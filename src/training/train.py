"""
This script submits experiments to perform 
hyperparameters optimization for multiple models.
"""

import os
import sys
from datetime import datetime
from pathlib import PosixPath

import comet_ml
import pandas as pd
from dotenv import load_dotenv
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from utils.config import Config
from utils.data import PrepTrainingData
from utils.job import create_voting_ensemble, submit_train_exp
from utils.path import ARTIFACTS_DIR, DATA_DIR
from xgboost import XGBClassifier

load_dotenv()


###########################################################
def main(
    config_yaml_abs_path: str,
    comet_api_key: str,
    data_dir: PosixPath,
    artifacts_dir: PosixPath,
):
    print(
        """\n
    ---------------------------------------------------------------------
    --- Hyperparameters Optimization Experiments Starts ...
    ---------------------------------------------------------------------\n"""
    )

    # Experiment settings
    config = Config(config_path=config_yaml_abs_path)
    INITIATE_COMET_PROJECT = bool(
        config.params["train"]["params"]["initiate_comet_project"]
    )
    COMET_API_KEY = comet_api_key
    COMET_PROJECT_NAME = config.params["train"]["params"]["comet_project_name"]
    COMET_WORKSPACE_NAME = config.params["train"]["params"]["comet_workspace_name"]
    PRIMARY_KEY = config.params["data"]["params"]["pk_col_name"]
    CLASS_COL_NAME = config.params["data"]["params"]["class_col_name"]
    num_col_names = config.params["data"]["params"]["num_col_names"]
    cat_col_names = config.params["data"]["params"]["cat_col_names"]
    MAX_SEARCH_ITERS = config.params["train"]["params"]["search_max_iters"]
    PARALLEL_JOBS_COUNT = config.params["train"]["params"]["parallel_jobs_count"]
    EXP_TIMEOUT_SECS = config.params["train"]["params"]["exp_timout_secs"]
    CROSS_VAL_FOLDS = config.params["train"]["params"]["cross_val_folds"]
    F_BETA_SCORE_BETA_VAL = config.params["train"]["params"]["fbeta_score_beta_val"]
    VOTING_RULE = config.params["train"]["params"]["voting_rule"]
    TRAIN_FILE_NAME = config.params["files"]["params"]["train_set_file_name"]
    VALID_FILE_NAME = config.params["files"]["params"]["valid_set_file_name"]
    TEST_FILE_NAME = config.params["files"]["params"]["test_set_file_name"]
    EXP_KEY_FILE_NAME = config.params["files"]["params"]["experiments_keys_file_name"]
    LR_REGISTERED_MODEL_NAME = config.params["modelregistry"]["params"][
        "lr_registered_model_name"
    ]
    RF_REGISTERED_MODEL_NAME = config.params["modelregistry"]["params"][
        "rf_registered_model_name"
    ]
    LGBM_REGISTERED_MODEL_NAME = config.params["modelregistry"]["params"][
        "lgbm_registered_model_name"
    ]
    XGB_REGISTERED_MODEL_NAME = config.params["modelregistry"]["params"][
        "xgb_registered_model_name"
    ]
    VOTING_ENSEMBLE_REGISTERED_MODEL_NAME = config.params["modelregistry"]["params"][
        "voting_ensemble_registered_model_name"
    ]
    POS_CLASS_LABEL = config.params["data"]["params"]["pos_class"]
    VAR_THRESH_VAL = config.params["data"]["params"]["variance_threshold_val"]

    # Import data splits
    training_set = pd.read_parquet(
        data_dir / TRAIN_FILE_NAME,
    )

    validation_set = pd.read_parquet(
        data_dir / VALID_FILE_NAME,
    )

    testing_set = pd.read_parquet(
        data_dir / TEST_FILE_NAME,
    )

    # Ensure that columns provided in config files exists in training data
    num_col_names = [col for col in num_col_names if col in training_set.columns]
    cat_col_names = [col for col in cat_col_names if col in training_set.columns]

    # Prepare data for training
    data_prep = PrepTrainingData(
        train_set=training_set,
        test_set=testing_set,
        primary_key=PRIMARY_KEY,
        class_col_name=CLASS_COL_NAME,
        numerical_feature_names=num_col_names,
        categorical_feature_names=cat_col_names,
    )
    data_prep.extract_features(valid_set=validation_set)
    data_prep.enforce_data_types()

    # Encode class labels
    # Note: class encoder is fitted on train class labels and will be used
    # to transform validation and test class labels.
    (
        train_class,
        valid_class,
        test_class,
        encoded_positive_class_label,
        class_encoder,
    ) = data_prep.encode_class_labels(
        pos_class_label=POS_CLASS_LABEL,
    )

    # Return features
    train_features = data_prep.get_training_features()
    valid_features = data_prep.get_validation_features()
    test_features = data_prep.get_testing_features()

    # Create data transformation pipeline
    data_transformation_pipeline = data_prep.create_data_transformation_pipeline(
        var_thresh_val=VAR_THRESH_VAL
    )
    data_prep.clean_up_feature_names()
    num_feature_names, cat_feature_names = data_prep.get_feature_names()

    # Return preprocessed train and validation features, which are needed during
    # hyperparams optimization to avoid applying data transformation in each iteration.
    train_features_preprocessed = data_prep.get_train_features_preprocessed()
    valid_features_preprocessed = data_prep.get_valid_features_preprocessed()

    # Save data splits with encoded class
    train_set = train_features
    train_set[CLASS_COL_NAME] = train_class
    train_set.to_parquet(
        data_dir / TRAIN_FILE_NAME,
        index=False,
    )

    valid_set = valid_features
    valid_set[CLASS_COL_NAME] = valid_class
    valid_set.to_parquet(
        data_dir / VALID_FILE_NAME,
        index=False,
    )

    test_set = test_features
    test_set[CLASS_COL_NAME] = test_class
    test_set.to_parquet(
        data_dir / TEST_FILE_NAME,
        index=False,
    )

    # Initiate a comet project if needed
    if INITIATE_COMET_PROJECT:
        comet_ml.init(
            project_name=COMET_PROJECT_NAME,
            workspace=COMET_WORKSPACE_NAME,
            api_key=COMET_API_KEY,
        )

    #############################################
    # Train Logistic Regression model
    if config.params["includedmodels"]["params"]["include_logistic_regression"]:
        lr_calibrated_pipeline, lr_experiment = submit_train_exp(
            comet_api_key=COMET_API_KEY,
            comet_project_name=COMET_PROJECT_NAME,
            comet_exp_name=f"logistic_regression_{datetime.now()}",
            train_features_preprocessed=train_features_preprocessed,
            train_class=train_class,
            valid_features_preprocessed=valid_features_preprocessed,
            valid_class=valid_class,
            train_features=train_features,
            valid_features=valid_features,
            n_features=train_features_preprocessed.shape[1],
            class_encoder=class_encoder,
            preprocessor_step=data_transformation_pipeline.named_steps["preprocessor"],
            selector_step=data_transformation_pipeline.named_steps["selector"],
            model=LogisticRegression(**config.params["logisticregression"]["params"]),
            artifacts_path=artifacts_dir,
            num_feature_names=num_feature_names,
            cat_feature_names=cat_feature_names,
            fbeta_score_beta=F_BETA_SCORE_BETA_VAL,
            encoded_pos_class_label=encoded_positive_class_label,
            max_search_iters=MAX_SEARCH_ITERS,
            optimize_in_parallel=True if PARALLEL_JOBS_COUNT > 1 else False,
            n_parallel_jobs=PARALLEL_JOBS_COUNT,
            model_opt_timeout_secs=EXP_TIMEOUT_SECS,
            registered_model_name=LR_REGISTERED_MODEL_NAME,
        )
        lr_experiment.end()
    else:
        lr_calibrated_pipeline = None
        lr_experiment = None

    #############################################
    # Train Random Forest model
    if config.params["includedmodels"]["params"]["include_random_forest"]:
        rf_calibrated_pipeline, rf_experiment = submit_train_exp(
            comet_api_key=COMET_API_KEY,
            comet_project_name=COMET_PROJECT_NAME,
            comet_exp_name=f"random_forest_{datetime.now()}",
            train_features_preprocessed=train_features_preprocessed,
            train_class=train_class,
            valid_features_preprocessed=valid_features_preprocessed,
            valid_class=valid_class,
            train_features=train_features,
            valid_features=valid_features,
            n_features=train_features_preprocessed.shape[1],
            class_encoder=class_encoder,
            preprocessor_step=data_transformation_pipeline.named_steps["preprocessor"],
            selector_step=data_transformation_pipeline.named_steps["selector"],
            model=RandomForestClassifier(**config.params["randomforest"]["params"]),
            artifacts_path=artifacts_dir,
            num_feature_names=num_feature_names,
            cat_feature_names=cat_feature_names,
            fbeta_score_beta=F_BETA_SCORE_BETA_VAL,
            encoded_pos_class_label=encoded_positive_class_label,
            max_search_iters=MAX_SEARCH_ITERS,
            optimize_in_parallel=True if PARALLEL_JOBS_COUNT > 1 else False,
            n_parallel_jobs=PARALLEL_JOBS_COUNT,
            model_opt_timeout_secs=EXP_TIMEOUT_SECS,
            registered_model_name=RF_REGISTERED_MODEL_NAME,
        )
        rf_experiment.end()
    else:
        rf_calibrated_pipeline = None
        rf_experiment = None

    #############################################
    # Train LightGBM model
    if config.params["includedmodels"]["params"]["include_lightgbm"]:
        lgbm_calibrated_pipeline, lgbm_experiment = submit_train_exp(
            comet_api_key=COMET_API_KEY,
            comet_project_name=COMET_PROJECT_NAME,
            comet_exp_name=f"lightgbm_{datetime.now()}",
            train_features_preprocessed=train_features_preprocessed,
            train_class=train_class,
            valid_features_preprocessed=valid_features_preprocessed,
            valid_class=valid_class,
            train_features=train_features,
            valid_features=valid_features,
            n_features=train_features_preprocessed.shape[1],
            class_encoder=class_encoder,
            preprocessor_step=data_transformation_pipeline.named_steps["preprocessor"],
            selector_step=data_transformation_pipeline.named_steps["selector"],
            model=LGBMClassifier(**config.params["lgbm"]["params"]),
            artifacts_path=artifacts_dir,
            num_feature_names=num_feature_names,
            cat_feature_names=cat_feature_names,
            fbeta_score_beta=F_BETA_SCORE_BETA_VAL,
            encoded_pos_class_label=encoded_positive_class_label,
            max_search_iters=MAX_SEARCH_ITERS,
            optimize_in_parallel=True if PARALLEL_JOBS_COUNT > 1 else False,
            n_parallel_jobs=PARALLEL_JOBS_COUNT,
            model_opt_timeout_secs=EXP_TIMEOUT_SECS,
            registered_model_name=LGBM_REGISTERED_MODEL_NAME,
        )
        lgbm_experiment.end()
    else:
        lgbm_calibrated_pipeline = None
        lgbm_experiment = None

    #############################################
    # Train XGBoost model
    if config.params["includedmodels"]["params"]["include_xgboost"]:
        xgb_calibrated_pipeline, xgb_experiment = submit_train_exp(
            comet_api_key=COMET_API_KEY,
            comet_project_name=COMET_PROJECT_NAME,
            comet_exp_name=f"xgboost_{datetime.now()}",
            train_features_preprocessed=train_features_preprocessed,
            train_class=train_class,
            valid_features_preprocessed=valid_features_preprocessed,
            valid_class=valid_class,
            train_features=train_features,
            valid_features=valid_features,
            n_features=train_features_preprocessed.shape[1],
            class_encoder=class_encoder,
            preprocessor_step=data_transformation_pipeline.named_steps["preprocessor"],
            selector_step=data_transformation_pipeline.named_steps["selector"],
            model=XGBClassifier(
                scale_pos_weight=sum(train_class == 0) / sum(train_class == 1),
                **config.params["xgboost"]["params"],
            ),
            artifacts_path=artifacts_dir,
            num_feature_names=num_feature_names,
            cat_feature_names=cat_feature_names,
            fbeta_score_beta=F_BETA_SCORE_BETA_VAL,
            encoded_pos_class_label=encoded_positive_class_label,
            max_search_iters=MAX_SEARCH_ITERS,
            optimize_in_parallel=True if PARALLEL_JOBS_COUNT > 1 else False,
            n_parallel_jobs=PARALLEL_JOBS_COUNT,
            model_opt_timeout_secs=EXP_TIMEOUT_SECS,
            registered_model_name=XGB_REGISTERED_MODEL_NAME,
        )
        xgb_experiment.end()
    else:
        xgb_calibrated_pipeline = None
        xgb_experiment = None

    #############################################
    # Create a voting ensmble model with LR, RF, LightGBM, and XGBoost as base estimators
    if config.params["includedmodels"]["params"]["include_voting_ensemble"]:
        ve_experiment = create_voting_ensemble(
            comet_api_key=COMET_API_KEY,
            comet_project_name=COMET_PROJECT_NAME,
            comet_exp_name=f"voting_ensemble_{datetime.now()}",
            lr_calib_pipeline=lr_calibrated_pipeline,
            rf_calib_pipeline=rf_calibrated_pipeline,
            lgbm_calib_pipeline=lgbm_calibrated_pipeline,
            xgb_calib_pipeline=xgb_calibrated_pipeline,
            train_features=train_features,
            valid_features=valid_features,
            valid_features_preprocessed=valid_features_preprocessed,
            train_class=train_class,
            valid_class=valid_class,
            class_encoder=class_encoder,
            artifacts_path=artifacts_dir,
            voting_rule=VOTING_RULE,
            encoded_pos_class_label=encoded_positive_class_label,
            cv_folds=CROSS_VAL_FOLDS,
            fbeta_score_beta=F_BETA_SCORE_BETA_VAL,
            registered_model_name=VOTING_ENSEMBLE_REGISTERED_MODEL_NAME,
        )
        ve_experiment.end()
    else:
        ve_experiment = None

    #############################################
    # Select the best performer
    exp_objects = {
        LR_REGISTERED_MODEL_NAME: lr_experiment,
        RF_REGISTERED_MODEL_NAME: rf_experiment,
        LGBM_REGISTERED_MODEL_NAME: lgbm_experiment,
        XGB_REGISTERED_MODEL_NAME: xgb_experiment,
        VOTING_ENSEMBLE_REGISTERED_MODEL_NAME: ve_experiment,
    }
    exp_objects = {
        key: value for key, value in exp_objects.items() if value is not None
    }

    if len(exp_objects) == 0:
        raise ValueError(
            "No model was selected for training or all training experiments failed."
        )

    # Save names of successful experiments names so that logged training
    # metrics can imported in evaluate.py from workspace
    exp_names_keys = {}
    for i in range(len(exp_objects)):
        exp_key = list(exp_objects.keys())[i]
        exp_value = list(exp_objects.values())[i]
        exp_names_keys.update(**{f"{exp_key}": exp_value.get_key()})

    successful_exp = pd.DataFrame(exp_names_keys.items())
    successful_exp.to_csv(f"{ARTIFACTS_DIR}/{EXP_KEY_FILE_NAME}.csv", index=False)


###########################################################
if __name__ == "__main__":
    main(
        config_yaml_abs_path=sys.argv[1],
        comet_api_key=os.environ["COMET_API_KEY"],
        data_dir=DATA_DIR,
        artifacts_dir=ARTIFACTS_DIR,
    )
