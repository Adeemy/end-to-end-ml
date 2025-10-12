"""
Runs training experiments to perform hyperparameters optimization
for multiple models. It tracks the experiments using Comet.ml.
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import PosixPath
from typing import List, Tuple

import comet_ml
import pandas as pd
from dotenv import load_dotenv
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from xgboost import XGBClassifier

from src.training.utils.config import Config
from src.training.utils.data import TrainingDataPrep
from src.training.utils.job import ModelTrainer, VotingEnsembleCreator
from src.utils.logger import get_console_logger
from src.utils.path import ARTIFACTS_DIR, DATA_DIR

load_dotenv()


def prepare_data(
    config_yaml_path: str,
    training_set: pd.DataFrame,
    validation_set: pd.DataFrame,
    testing_set: pd.DataFrame,
) -> Tuple[
    TrainingDataPrep,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    List[str],
    List[str],
    LabelEncoder,
    int,
]:
    """Prepare data for training by preprocessing training and validation sets.

    Args:
        config_yaml_path (str): path to config yaml file.
        training_set (pd.DataFrame): training set.
        validation_set (pd.DataFrame): validation set.
        testing_set (pd.DataFrame): testing set.

    Returns:
        data_prep (TrainingDataPrep): data preparation object.
        data_transformation_pipeline (Pipeline): data transformation pipeline.
        train_features (pd.DataFrame): preprocessed training features.
        valid_features (pd.DataFrame): preprocessed validation features.
        test_features (pd.DataFrame): preprocessed testing features.
        train_class (pd.Series): training class labels.
        valid_class (pd.Series): validation class labels.
        test_class (pd.Series): testing class labels.
        num_feature_names (List[str]): numerical feature names.
        cat_feature_names (List[str]): categorical feature names.
        class_encoder (LabelEncoder): class label encoder.
        encoded_positive_class_label (int): encoded positive class label.
    """

    config = Config(config_path=config_yaml_path)
    pk_col_name = config.params["data"]["pk_col_name"]
    class_column_name = config.params["data"]["class_col_name"]
    num_col_names = config.params["data"]["num_col_names"]
    cat_col_names = config.params["data"]["cat_col_names"]
    pos_class = config.params["data"]["pos_class"]

    num_features_imputer = config.params["preprocessing"]["num_features_imputer"]
    num_features_scaler = config.params["preprocessing"]["num_features_scaler"]
    scaler_params = config.params["preprocessing"].get("scaler_params", {})
    cat_features_imputer = config.params["preprocessing"]["cat_features_imputer"]
    cat_features_ohe_handle_unknown = config.params["preprocessing"][
        "cat_features_ohe_handle_unknown"
    ]
    cat_features_nans_replacement = config.params["preprocessing"][
        "cat_features_nans_replacement"
    ]
    var_thresh_val = config.params["preprocessing"]["var_thresh_val"]

    # Prepare data for training
    data_prep = TrainingDataPrep(
        train_set=training_set,
        test_set=testing_set,
        primary_key=pk_col_name,
        class_col_name=class_column_name,
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
        pos_class_label=pos_class,
    )

    # Return features
    train_features = data_prep.training_features
    valid_features = data_prep.validation_features
    test_features = data_prep.testing_features

    # Define the mapping from strings to scaler classes
    scaler_mapping = {
        "robust": RobustScaler,
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "none": None,
    }
    scaler_class = scaler_mapping[num_features_scaler]
    scaler_params = {k: v for d in scaler_params for k, v in d.items()}

    # Create data transformation pipeline
    data_transformation_pipeline = data_prep.create_data_transformation_pipeline(
        num_features_imputer=num_features_imputer,
        num_features_scaler=scaler_class(**scaler_params),
        cat_features_imputer=cat_features_imputer,
        cat_features_ohe_handle_unknown=cat_features_ohe_handle_unknown,
        cat_features_nans_replacement=cat_features_nans_replacement,
        var_thresh_val=var_thresh_val,
    )
    data_prep.clean_up_feature_names()
    num_feature_names, cat_feature_names = data_prep.get_feature_names()

    return (
        data_prep,
        data_transformation_pipeline,
        train_features,
        valid_features,
        test_features,
        train_class,
        valid_class,
        test_class,
        num_feature_names,
        cat_feature_names,
        class_encoder,
        encoded_positive_class_label,
    )


def main(
    config_yaml_path: str,
    api_key: str,
    data_dir: PosixPath,
    artifacts_dir: PosixPath,
    logger: logging.Logger,
) -> None:
    """
    Takes a config file as input and submits experiments to perform
    hyperparameters optimization for multiple models.

    Args:
        config_yaml_path (str): path to config yaml file.
        api_key (str): Comet API key.
        data_dir (PosixPath): path to data directory.
        artifacts_dir (PosixPath): path to artifacts directory.
        logger (logging.Logger): logger object.

    Raises:
        ValueError: if no model specified in config file.
    """

    logger.info("Directory of training config file: %s", config_yaml_path)

    # Get configuration parameters
    config = Config(config_path=config_yaml_path)
    initiate_comet_project = bool(config.params["train"]["initiate_comet_project"])
    project_name = config.params["train"]["project_name"]
    workspace_name = config.params["train"]["workspace_name"]
    class_column_name = config.params["data"]["class_col_name"]
    num_col_names = config.params["data"]["num_col_names"]
    cat_col_names = config.params["data"]["cat_col_names"]

    search_max_iters = config.params["train"]["search_max_iters"]
    parallel_jobs_count = config.params["train"]["parallel_jobs_count"]
    exp_timeout_in_secs = config.params["train"]["exp_timout_secs"]
    f_beta_score_beta_val = config.params["train"]["fbeta_score_beta_val"]
    ve_voting_rule = config.params["train"]["voting_rule"]
    train_file_name = config.params["files"]["train_set_file_name"]
    valid_set_file_name = config.params["files"]["valid_set_file_name"]
    test_set_file_name = config.params["files"]["test_set_file_name"]
    exp_key_file_name = config.params["files"]["experiments_keys_file_name"]
    lr_registered_model_name = config.params["modelregistry"][
        "lr_registered_model_name"
    ]
    rf_registered_model_name = config.params["modelregistry"][
        "rf_registered_model_name"
    ]
    lgbm_registered_model_name = config.params["modelregistry"][
        "lgbm_registered_model_name"
    ]
    xgb_registered_model_name = config.params["modelregistry"][
        "xgb_registered_model_name"
    ]
    ve_registered_model_name = config.params["modelregistry"][
        "voting_ensemble_registered_model_name"
    ]

    lr_params = config.params["logisticregression"]["params"]
    rf_params = config.params["randomforest"]["params"]
    lgbm_params = config.params["lgbm"]["params"]
    xgb_params = config.params["xgboost"]["params"]

    lr_search_space_params = config.params["logisticregression"]["search_space_params"]
    rf_search_space_params = config.params["randomforest"]["search_space_params"]
    lgbm_search_space_params = config.params["lgbm"]["search_space_params"]
    xgb_search_space_params = config.params["xgboost"]["search_space_params"]

    # Import data splits
    training_set = pd.read_parquet(
        data_dir / train_file_name,
    )

    validation_set = pd.read_parquet(
        data_dir / valid_set_file_name,
    )

    testing_set = pd.read_parquet(
        data_dir / test_set_file_name,
    )

    # Ensure that columns provided in config files exists in training data
    num_col_names = [col for col in num_col_names if col in training_set.columns]
    cat_col_names = [col for col in cat_col_names if col in training_set.columns]

    # Prepare data for training
    (
        data_prep,
        data_transformation_pipeline,
        train_features,
        valid_features,
        test_features,
        train_class,
        valid_class,
        test_class,
        num_feature_names,
        cat_feature_names,
        class_encoder,
        encoded_positive_class_label,
    ) = prepare_data(
        config_yaml_path=config_yaml_path,
        training_set=training_set,
        validation_set=validation_set,
        testing_set=testing_set,
    )

    # Preprocessed train and validation features are needed during hyperparams
    # optimization to avoid applying data transformation in each iteration.
    train_features_preprocessed = data_prep.train_features_preprocessed
    valid_features_preprocessed = data_prep.valid_features_preprocessed

    # Save data splits with encoded class to be used in models evaluation
    # TODO: an integration test should be added to check if the saved files.
    train_set = train_features
    train_set[class_column_name] = train_class
    train_set.to_parquet(
        data_dir / train_file_name,
        index=False,
    )

    valid_set = valid_features
    valid_set[class_column_name] = valid_class
    valid_set.to_parquet(
        data_dir / valid_set_file_name,
        index=False,
    )

    test_set = test_features
    test_set[class_column_name] = test_class
    test_set.to_parquet(
        data_dir / test_set_file_name,
        index=False,
    )

    # Initiate a comet project if needed
    if initiate_comet_project:
        comet_ml.init(
            project_name=project_name,
            workspace=workspace_name,
            api_key=api_key,
        )

    # Initialize training class
    model_trainer = ModelTrainer(
        train_features=train_features,
        train_class=train_class,
        valid_features=valid_features,
        valid_class=valid_class,
        train_features_preprocessed=train_features_preprocessed,
        valid_features_preprocessed=valid_features_preprocessed,
        n_features=train_features_preprocessed.shape[1],
        class_encoder=class_encoder,
        preprocessor_step=data_transformation_pipeline.named_steps["preprocessor"],
        selector_step=data_transformation_pipeline.named_steps["selector"],
        artifacts_path=artifacts_dir,
        num_feature_names=num_feature_names,
        cat_feature_names=cat_feature_names,
        fbeta_score_beta=f_beta_score_beta_val,
        encoded_pos_class_label=encoded_positive_class_label,
    )

    #############################################
    # Train Logistic Regression model
    if config.params["includedmodels"]["include_logistic_regression"]:
        lr_calibrated_pipeline, lr_experiment = model_trainer.submit_train_exp(
            comet_api_key=api_key,
            project_name=project_name,
            comet_exp_name=f"logistic_regression_{datetime.now()}",
            model=LogisticRegression(**lr_params),
            search_space_params=lr_search_space_params,
            max_search_iters=search_max_iters,
            optimize_in_parallel=True if parallel_jobs_count > 1 else False,
            n_parallel_jobs=parallel_jobs_count,
            model_opt_timeout_secs=exp_timeout_in_secs,
            registered_model_name=lr_registered_model_name,
        )
    else:
        lr_calibrated_pipeline = None
        lr_experiment = None

    #############################################
    # Train Random Forest model
    if config.params["includedmodels"]["include_random_forest"]:
        rf_calibrated_pipeline, rf_experiment = model_trainer.submit_train_exp(
            comet_api_key=api_key,
            project_name=project_name,
            comet_exp_name=f"random_forest_{datetime.now()}",
            model=RandomForestClassifier(**rf_params),
            search_space_params=rf_search_space_params,
            max_search_iters=search_max_iters,
            optimize_in_parallel=True if parallel_jobs_count > 1 else False,
            n_parallel_jobs=parallel_jobs_count,
            model_opt_timeout_secs=exp_timeout_in_secs,
            registered_model_name=rf_registered_model_name,
        )
    else:
        rf_calibrated_pipeline = None
        rf_experiment = None

    #############################################
    # Train LightGBM model
    if config.params["includedmodels"]["include_lightgbm"]:
        lgbm_calibrated_pipeline, lgbm_experiment = model_trainer.submit_train_exp(
            comet_api_key=api_key,
            project_name=project_name,
            comet_exp_name=f"random_forest_{datetime.now()}",
            model=LGBMClassifier(**lgbm_params),
            search_space_params=lgbm_search_space_params,
            max_search_iters=search_max_iters,
            optimize_in_parallel=True if parallel_jobs_count > 1 else False,
            n_parallel_jobs=parallel_jobs_count,
            model_opt_timeout_secs=exp_timeout_in_secs,
            registered_model_name=lgbm_registered_model_name,
        )
    else:
        lgbm_calibrated_pipeline = None
        lgbm_experiment = None

    #############################################
    # Train XGBoost model
    if config.params["includedmodels"]["include_xgboost"]:
        xgb_calibrated_pipeline, xgb_experiment = model_trainer.submit_train_exp(
            comet_api_key=api_key,
            project_name=project_name,
            comet_exp_name=f"random_forest_{datetime.now()}",
            model=XGBClassifier(
                scale_pos_weight=sum(train_class == 0) / sum(train_class == 1),
                **xgb_params,
            ),
            search_space_params=xgb_search_space_params,
            max_search_iters=search_max_iters,
            optimize_in_parallel=True if parallel_jobs_count > 1 else False,
            n_parallel_jobs=parallel_jobs_count,
            model_opt_timeout_secs=exp_timeout_in_secs,
            registered_model_name=xgb_registered_model_name,
        )
    else:
        xgb_calibrated_pipeline = None
        xgb_experiment = None

    #############################################
    # Create a voting ensmble model with LR, RF, LightGBM, and XGBoost as base estimators
    if config.params["includedmodels"]["include_voting_ensemble"]:
        ve_creator = VotingEnsembleCreator(
            comet_api_key=api_key,
            project_name=project_name,
            comet_exp_name=f"voting_ensemble_{datetime.now()}",
            train_features=train_features,
            valid_features=valid_features,
            train_class=train_class,
            valid_class=valid_class,
            class_encoder=class_encoder,
            artifacts_path=artifacts_dir,
            lr_calib_pipeline=lr_calibrated_pipeline,
            rf_calib_pipeline=rf_calibrated_pipeline,
            lgbm_calib_pipeline=lgbm_calibrated_pipeline,
            xgb_calib_pipeline=xgb_calibrated_pipeline,
            voting_rule=ve_voting_rule,
            encoded_pos_class_label=encoded_positive_class_label,
            fbeta_score_beta=f_beta_score_beta_val,
            registered_model_name=ve_registered_model_name,
        )

        _, ve_experiment = ve_creator.create_voting_ensemble()

    else:
        ve_experiment = None

    #############################################
    # Select the best performer
    exp_objects = {
        lr_registered_model_name: lr_experiment,
        rf_registered_model_name: rf_experiment,
        lgbm_registered_model_name: lgbm_experiment,
        xgb_registered_model_name: xgb_experiment,
        ve_registered_model_name: ve_experiment,
    }
    exp_objects = {
        key: value for key, value in exp_objects.items() if value is not None
    }

    if len(exp_objects) == 0:
        raise ValueError(
            "No model was selected in config for training or all training experiments failed."
        )

    # Save names of successful experiments names so that logged training
    # metrics can imported in evaluate.py from workspace
    exp_names_keys = {}
    for i in range(len(exp_objects)):
        exp_key = list(exp_objects.keys())[i]
        exp_value = list(exp_objects.values())[i]
        exp_names_keys.update(**{f"{exp_key}": exp_value.get_key()})

    successful_exp = pd.DataFrame(exp_names_keys.items())
    successful_exp.to_csv(f"{ARTIFACTS_DIR}/{exp_key_file_name}.csv", index=False)

    logger.info("Model Training Experiments Finished ...")


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

    module_name: str = PosixPath(__file__).stem
    console_logger = get_console_logger(module_name)
    console_logger.info("Hyperparameters Optimization Experiments Starts ...")

    main(
        config_yaml_path=args.config_yaml_path,
        api_key=os.environ["COMET_API_KEY"],
        data_dir=DATA_DIR,
        artifacts_dir=ARTIFACTS_DIR,
        logger=console_logger,
    )
