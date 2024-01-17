"""
This script performs hyperparameters optimization 
using Optuna Python package.
"""

import os
import sys
from datetime import datetime
from pathlib import PosixPath

import comet_ml
import joblib
from comet_ml import ExistingExperiment
from dotenv import load_dotenv
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from utils.config import Config
from utils.data import PrepTrainingData
from utils.job import create_voting_ensemble, submit_train_exp
from utils.model import (
    ModelEvaluator,
    log_and_register_champ_model,
    select_best_performer,
)
from utils.path import ARTIFACTS_DIR
from xgboost import XGBClassifier

load_dotenv()


###########################################################
def main(config_yaml_abs_path: str, comet_api_key: str, artifacts_dir: PosixPath):
    # Experiment settings
    config = Config(config_path=config_yaml_abs_path)
    INITIATE_COMET_PROJECT = bool(
        config.params["train"]["params"]["initiate_comet_project"]
    )
    COMET_API_KEY = comet_api_key
    COMET_PROJECT_NAME = config.params["train"]["params"]["comet_project_name"]
    COMET_WORKSPACE_NAME = config.params["train"]["params"]["comet_workspace_name"]
    MAX_SEARCH_ITERS = config.params["train"]["params"]["search_max_iters"]
    PARALLEL_JOBS_COUNT = config.params["train"]["params"]["parallel_jobs_count"]
    EXP_TIMEOUT_SECS = config.params["train"]["params"]["exp_timout_secs"]
    CROSS_VAL_FOLDS = config.params["train"]["params"]["cross_val_folds"]
    F_BETA_SCORE_BETA_VAL = config.params["train"]["params"]["fbeta_score_beta_val"]
    COMPARISON_METRIC = config.params["train"]["params"]["comparison_metric"]
    VOTING_RULE = config.params["train"]["params"]["voting_rule"]
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
    CHAMPION_MODEL_NAME = config.params["modelregistry"]["params"][
        "champion_model_name"
    ]
    DEPLOYMENT_SCORE_THRESH = config.params["train"]["params"][
        "deployment_score_thresh"
    ]

    # Dataset split configuration, feature data types, and positive class label
    HUGGINGFACE_SOURCE = config.params["data"]["params"]["raw_dataset_source"]
    PRIMARY_KEY = config.params["data"]["params"]["pk_col_name"]
    CLASS_COL_NAME = config.params["data"]["params"]["class_col_name"]
    POS_CLASS_LABEL = config.params["data"]["params"]["pos_class"]
    num_col_names = config.params["data"]["params"]["num_col_names"]
    cat_col_names = config.params["data"]["params"]["cat_col_names"]
    DATASET_SPLIT_TYPE = config.params["data"]["params"]["split_type"]
    DATASET_SPLIT_SEED = config.params["data"]["params"]["split_rand_seed"]
    SPLIT_DATE_COL_NAME = config.params["data"]["params"]["split_date_col_name"]
    SPLIT_CUTOFF_DATE = config.params["data"]["params"]["train_valid_split_curoff_date"]
    SPLIT_DATE_FORMAT = config.params["data"]["params"]["split_date_col_format"]
    CAT_FEAT_NAN_REPLACEMENT = config.params["data"]["params"][
        "cat_features_nan_replacement"
    ]
    TRAIN_SIZE = config.params["data"]["params"]["train_set_size"]
    VAR_THRESH_VAL = config.params["data"]["params"]["variance_threshold_val"]

    # Import dataset and prepare it for training
    data_prep = PrepTrainingData(
        primary_key=PRIMARY_KEY,
        class_col_name=CLASS_COL_NAME,
        numerical_feature_names=num_col_names,
        categorical_feature_names=cat_col_names,
    )

    # Note: train and test sets are imported from Hugging Face dataset repo
    # to enable running training pipeline in GitHub Actions. Thus, train and
    # test splits performed in previous step has not impact in this setting.
    data_prep.import_datasets(
        hf_data_source=HUGGINGFACE_SOURCE,
        is_local_source=False,
    )

    # Preprocess train and test sets by enforcing data types of numerical and categorical features
    data_prep.select_relevant_columns()
    data_prep.enforce_data_types()
    data_prep.replace_nans_in_cat_features(nan_replacement=CAT_FEAT_NAN_REPLACEMENT)
    data_prep.create_validation_set(
        split_type=DATASET_SPLIT_TYPE,
        train_set_size=TRAIN_SIZE,
        split_random_seed=DATASET_SPLIT_SEED,
        split_date_col_name=SPLIT_DATE_COL_NAME,
        split_cutoff_date=SPLIT_CUTOFF_DATE,
        split_date_col_format=SPLIT_DATE_FORMAT,
    )
    data_prep.drop_primary_key()
    data_prep.extract_features()

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

    # Create data transformation pipeline
    data_transformation_pipeline = data_prep.create_data_transformation_pipeline(
        var_thresh_val=VAR_THRESH_VAL
    )
    data_prep.clean_up_feature_names()
    num_feature_names, cat_feature_names = data_prep.get_feature_names()

    # Return datasets
    # Note: preprocessed train and validation features are needed during hyperparams
    # optimization procedure to avoid data transformation in each iteration.
    train_features = data_prep.get_training_features()
    valid_features = data_prep.get_validation_features()
    test_features = data_prep.get_testing_features()
    train_features_preprocessed = data_prep.get_train_features_preprocessed()
    valid_features_preprocessed = data_prep.get_valid_features_preprocessed()

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
    else:
        xgb_calibrated_pipeline = None
        xgb_experiment = None

    #############################################
    # Create a voting ensmble model with LR, RF, LightGBM, and XGBoost as base estimators
    if config.params["includedmodels"]["params"]["include_voting_ensemble"]:
        ve_calibrated_pipeline, ve_experiment = create_voting_ensemble(
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
    else:
        ve_calibrated_pipeline = None
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
            "No model was selected to be trained. Select at least one model!"
        )

    # Rename comparison metric if it's fbeta_score to include beta value
    if COMPARISON_METRIC == "fbeta_score":
        COMPARISON_METRIC = f"f_{F_BETA_SCORE_BETA_VAL}_score"

    best_model_name = select_best_performer(
        comparison_metric="valid_" + COMPARISON_METRIC, models_with_exp=exp_objects
    )

    # Register the best performer as champion model for scoring
    best_model_exp = exp_objects.get(best_model_name)
    best_model_exp_key = best_model_exp.get_key()

    #############################################
    # Assess generalization capability of the best performer on test set
    # Note: test set was not exposed to any model during training or
    # evaluation to ensure all models are independent of the test set.
    compared_pipelines = {
        LR_REGISTERED_MODEL_NAME: lr_calibrated_pipeline,
        RF_REGISTERED_MODEL_NAME: rf_calibrated_pipeline,
        LGBM_REGISTERED_MODEL_NAME: lgbm_calibrated_pipeline,
        XGB_REGISTERED_MODEL_NAME: xgb_calibrated_pipeline,
        VOTING_ENSEMBLE_REGISTERED_MODEL_NAME: ve_calibrated_pipeline,
    }
    compared_pipelines = {
        key: value for key, value in compared_pipelines.items() if value is not None
    }
    best_model_pipeline = compared_pipelines.get(best_model_name)

    best_model_evaluator = ModelEvaluator(
        comet_exp=best_model_exp,
        pipeline=best_model_pipeline,
        train_features=train_features,
        train_class=train_class,
        valid_features=test_features,
        valid_class=test_class,
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
        class_encoder=class_encoder,
    )

    test_scores = best_model_evaluator.convert_metrics_from_df_to_dict(
        scores=test_scores, prefix="test_"
    )

    # Create ExistingExperiment object to allow appending logging new metrics
    best_model_exp_obj = ExistingExperiment(
        api_key=COMET_API_KEY, experiment_key=best_model_exp_key
    )
    best_model_exp_obj.log_metrics(test_scores)

    # Log and register champion model (in Comet, model must be logged first)
    # Note: the best model should not be deployed in production if its score
    # on the test set is below minimum score. Otherwise, prevent deploying
    # the model by raising error preventing build job.
    BEST_MODEL_TEST_SCORE = test_scores.get(f"test_{COMPARISON_METRIC}")
    if BEST_MODEL_TEST_SCORE >= DEPLOYMENT_SCORE_THRESH:
        log_and_register_champ_model(
            local_path=artifacts_dir,
            champ_model_name=CHAMPION_MODEL_NAME,
            pipeline=best_model_pipeline,
            exp_obj=best_model_exp_obj,
        )

        # Save the champion model in local direcotry to be uploaded as an artifact
        joblib.dump(best_model_pipeline, f"{ARTIFACTS_DIR}/{CHAMPION_MODEL_NAME}.pkl")

    else:
        raise ValueError(
            f"Best model score is {BEST_MODEL_TEST_SCORE}, which is lower than deployment threshold {DEPLOYMENT_SCORE_THRESH}."
        )


###########################################################
if __name__ == "__main__":
    # Submit training experiment
    main(
        config_yaml_abs_path=sys.argv[1],
        comet_api_key=os.environ["COMET_API_KEY"],
        artifacts_dir=ARTIFACTS_DIR,
    )
