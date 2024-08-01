"""
Runs training experiments to perform hyperparameters optimization
for multiple models. It tracks the experiments using mlflow.
"""

import argparse
import logging
import logging.config
import os
import sys
from datetime import datetime
from pathlib import Path, PosixPath

import mlflow
from azureml.core import Dataset, Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from dotenv import load_dotenv
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# To import modules from the parent directory in Azure compute cluster
root_dir = Path(__name__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.training.utils.config import Config
from src.training.utils.helpers import prepare_training_data
from src.training.utils.job import ModelTrainer, VotingEnsembleCreator
from src.utils.logger import get_console_logger
from src.utils.path import ARTIFACTS_DIR

load_dotenv()


def main(
    config_yaml_path: str,
    artifacts_dir: PosixPath,
    logger: logging.Logger,
) -> None:
    """Takes a config file as input and submits experiments to perform
    hyperparameters optimization for multiple models.

    Args:
        config_yaml_path (str): path to config yaml file.
        artifacts_dir (PosixPath): path to artifacts directory.
        logger (logging.Logger): logger object.

    Raises:
        ValueError: if no model specified in config file.
    """

    logger.info("Directory of training config file: %s", config_yaml_path)

    # Get configuration parameters
    config = Config(config_path=config_yaml_path)
    project_name = config.params["logger"]["project"]
    class_column_name = config.params["data"]["class_col_name"]
    num_col_names = config.params["data"]["num_col_names"]
    cat_col_names = config.params["data"]["cat_col_names"]

    search_max_iters = config.params["train"]["search_max_iters"]
    parallel_jobs_no = config.params["train"]["parallel_jobs_no"]
    exp_timeout_in_secs = config.params["train"]["exp_timout_secs"]
    f_beta_score_beta_val = float(config.params["train"]["fbeta_score_beta_val"])
    cross_val_folds = config.params["train"]["cross_val_folds"]
    ve_voting_rule = config.params["train"]["voting_rule"]

    datastore_name = config.params["datasets"]["datastore_name"]
    train_set_name = config.params["datasets"]["train_set_name"]
    valid_set_name = config.params["datasets"]["valid_set_file_name"]
    test_set_name = config.params["datasets"]["test_set_file_name"]
    train_set_version = config.params["datasets"]["train_set_version"]
    valid_set_version = config.params["datasets"]["valid_set_version"]
    test_set_version = config.params["datasets"]["test_set_version"]
    train_set_desc = config.params["datasets"]["train_set_desc"]
    valid_set_desc = config.params["datasets"]["valid_set_desc"]

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

    # Connect to the training workspace
    sp_authentication = ServicePrincipalAuthentication(
        tenant_id=os.environ["TENANT_ID"],
        service_principal_id=os.environ["APP_REGISTRATION_ID"],
        service_principal_password=os.environ["SP_PWD"],
    )
    ws = Workspace(
        os.environ["SUBSCRIPTION_ID"],
        os.environ["RESOURCE_GROUP_NAME"],
        os.environ["AML_WORKSPACE_NAME"],
        auth=sp_authentication,
    )

    #################################
    # Import data splits
    registered_train_set = Dataset.get_by_name(
        ws, name=train_set_name, version=train_set_version
    )
    training_set = registered_train_set.to_pandas_dataframe()

    registered_valid_set = Dataset.get_by_name(
        ws, name=valid_set_name, version=valid_set_version
    )
    validation_set = registered_valid_set.to_pandas_dataframe()

    registered_test_set = Dataset.get_by_name(
        ws, name=test_set_name, version=test_set_version
    )
    testing_set = registered_test_set.to_pandas_dataframe()

    # Ensure that columns provided in config files exists in training data
    num_col_names = [col for col in num_col_names if col in training_set.columns]
    cat_col_names = [col for col in cat_col_names if col in training_set.columns]

    # Prepare data for training
    (
        data_prep,
        data_transformation_pipeline,
        train_features,
        valid_features,
        train_class,
        valid_class,
        num_feature_names,
        cat_feature_names,
        class_encoder,
        encoded_positive_class_label,
    ) = prepare_training_data(
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
    # Register the train set
    train_set = train_features
    train_set[class_column_name] = train_class
    _ = Dataset.Tabular.register_pandas_dataframe(
        dataframe=train_set,
        target=ws.get_default_datastore(),
        name=f"transformed_{train_set_name}",
        description=f"transformed {train_set_desc}",
        show_progress=True,
    )

    # Register the validation set
    valid_set = valid_features
    valid_set[class_column_name] = valid_class
    _ = Dataset.Tabular.register_pandas_dataframe(
        dataframe=valid_set,
        target=ws.get_default_datastore(),
        name=f"transformed_{valid_set_name}",
        description=f"transformed {valid_set_desc}",
        show_progress=True,
    )

    # Create training input parameters
    training_input_params = {
        "Max Search Iters": search_max_iters,
        "Parallel Jobs Count": parallel_jobs_no,
        "CPUs Count": os.cpu_count(),
        "CV Folds": cross_val_folds,
        "Class Column Name": class_column_name,
        "Encoded Positive Class Label": encoded_positive_class_label,
        "Datastore Name": datastore_name,
        "Training Set Name": registered_train_set.name,
        "Testing Set Name": registered_test_set.name,
        "Training Set Version": registered_train_set.version,
        "Testing Set Version": registered_test_set.version,
        "No. of Original Features": data_prep.training_features.shape[1],
        "No. of Transformed Features": data_prep.train_features_preprocessed.shape[1],
    }
    logger.info("Training input parameters: %s", training_input_params)

    # Create model trainer
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
        registered_train_set=registered_train_set,
        registered_test_set=registered_test_set,
        artifacts_path=artifacts_dir,
        num_feature_names=num_feature_names,
        cat_feature_names=cat_feature_names,
        fbeta_score_beta=f_beta_score_beta_val,
        encoded_pos_class_label=encoded_positive_class_label,
        optimize_in_parallel=True if parallel_jobs_no > 1 else False,
        n_parallel_jobs=parallel_jobs_no,
        model_opt_timeout_secs=exp_timeout_in_secs,
        cv_folds=cross_val_folds,
    )

    # Create an experiment
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    mlflow.set_experiment(f"{project_name}_training_exp")

    # Ensure that any active runs within this session are terminated
    mlflow.end_run()

    with mlflow.start_run(
        run_name=f"model_training_{datetime.now()}",
    ) as parent_run:
        mlflow.set_tags(training_input_params)

        # Train Logistic Regression model
        if config.params["includedmodels"]["include_logistic_regression"]:
            with mlflow.start_run(
                nested=True, run_name="logistic_regression"
            ) as child_run:
                mlflow.set_tags(training_input_params)

            #############################################
            # Train Logistic Regression model
            if config.params["includedmodels"]["include_logistic_regression"]:
                lr_pipeline = model_trainer.submit_train_exp(
                    parent_run_id=parent_run.info.run_id,
                    child_run_id=child_run.info.run_id,
                    model=LogisticRegression(**lr_params),
                    search_space_params=lr_search_space_params,
                    registered_model_name=lr_registered_model_name,
                )
            else:
                lr_pipeline = None

            #############################################
            # Train Random Forest model
            if config.params["includedmodels"]["include_random_forest"]:
                rf_pipeline = model_trainer.submit_train_exp(
                    parent_run_id=parent_run.info.run_id,
                    child_run_id=child_run.info.run_id,
                    model=RandomForestClassifier(**rf_params),
                    search_space_params=rf_search_space_params,
                    registered_model_name=rf_registered_model_name,
                )
            else:
                rf_pipeline = None

            #############################################
            # Train LightGBM model
            if config.params["includedmodels"]["include_lightgbm"]:
                lgbm_pipeline = model_trainer.submit_train_exp(
                    parent_run_id=parent_run.info.run_id,
                    child_run_id=child_run.info.run_id,
                    model=LGBMClassifier(**lgbm_params),
                    search_space_params=lgbm_search_space_params,
                    registered_model_name=lgbm_registered_model_name,
                )
            else:
                lgbm_pipeline = None

            #############################################
            # Train XGBoost model
            if config.params["includedmodels"]["include_xgboost"]:
                xgb_pipeline = model_trainer.submit_train_exp(
                    parent_run_id=parent_run.info.run_id,
                    child_run_id=child_run.info.run_id,
                    model=XGBClassifier(
                        scale_pos_weight=sum(train_class == 0) / sum(train_class == 1),
                        **xgb_params,
                    ),
                    search_space_params=xgb_search_space_params,
                    registered_model_name=xgb_registered_model_name,
                )
            else:
                xgb_pipeline = None

            #############################################
            # Create a voting ensmble model with LR, RF, LightGBM, and XGBoost as base estimators
            if config.params["includedmodels"]["include_voting_ensemble"]:
                ve_creator = VotingEnsembleCreator(
                    parent_run_id=parent_run.info.run_id,
                    child_run_id=child_run.info.run_id,
                    train_features=data_prep.training_features,
                    valid_features=data_prep.validation_features,
                    train_class=train_class,
                    valid_class=valid_class,
                    class_encoder=model_trainer.class_encoder,
                    artifacts_path=model_trainer.artifacts_path,
                    registered_train_set=model_trainer.registered_train_set,
                    registered_test_set=model_trainer.registered_test_set,
                    lr_pipeline=lr_pipeline,
                    rf_pipeline=rf_pipeline,
                    lgbm_pipeline=lgbm_pipeline,
                    xgb_pipeline=xgb_pipeline,
                    voting_rule=ve_voting_rule,
                    encoded_pos_class_label=model_trainer.encoded_pos_class_label,
                    fbeta_score_beta=model_trainer.fbeta_score_beta,
                    registered_model_name=ve_registered_model_name,
                    conf_score_threshold_val=model_trainer.conf_score_threshold_val,
                    cv_folds=cross_val_folds,
                )

                _ = ve_creator.create_voting_ensemble()

    # Terminate current active run
    mlflow.end_run()

    logger.info("Model training experiments finished ...")

    # Evaluate models on testing set
    # Note: evaluate.py must be run using the activated env
    # os.system(
    #     f"{sys.prefix}/bin/python evaluate.py --run_id {parent_run.info.run_id}"
    # )


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
    console_logger = get_console_logger("train_logger")

    console_logger.info("Hyperparameters Optimization Experiments Starts ...")

    main(
        config_yaml_path=args.config_yaml_path,
        artifacts_dir=ARTIFACTS_DIR,
        logger=console_logger,
    )
