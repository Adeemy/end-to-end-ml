"""
This utility module includes functions to submit
training job.
"""

import re
from copy import deepcopy
from typing import Callable, Literal

import joblib
import numpy as np
import pandas as pd
from comet_ml import Experiment
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from utils.model import ModelEvaluator, ModelOptimizer

###########################################################


def submit_train_exp(
    comet_api_key: str,
    comet_project_name: str,
    comet_exp_name: str,
    train_features_preprocessed: pd.DataFrame,
    train_class: np.ndarray,
    valid_features_preprocessed: pd.DataFrame,
    valid_class: np.ndarray,
    train_features: pd.DataFrame,
    valid_features: pd.DataFrame,
    n_features: int,
    class_encoder: LabelEncoder,
    preprocessor_step: ColumnTransformer,
    selector_step: VarianceThreshold,
    model: Callable,
    artifacts_path: str,
    num_feature_names: list = None,
    cat_feature_names: list = None,
    fbeta_score_beta: float = 1.0,
    encoded_pos_class_label: int = 1,
    max_search_iters: int = 100,
    optimize_in_parallel: bool = False,
    n_parallel_jobs: int = 4,
    model_opt_timeout_secs: int = 600,
    registered_model_name: str = None,
    is_voting_ensemble: bool = False,
) -> Pipeline:
    """Submits an experiment to train a model using hyperparameter optimization.
    Args:
        comet_api_key (str): Comet API key,
        comet_project_name (str): Comet project name,
        comet_exp_name (str)L Comet experiment name,
        train_features_preprocessed (pd.DataFrame): transformed training set for ModelOptimizer class.
        train_class (np.ndarray): The target labels for the training set.
        valid_features_preprocessed (pd.DataFrame): transformed validation set for ModelOptimizer class.
        valid_class (np.ndarray): The target labels for the validation set.
        train_features (pd.DataFrame): features of training set before encoding and
            feature selection for ModelEvaluator class.
        valid_features (pd.DataFrame): features of validation set before encoding and
            feature selection for ModelEvaluator class.
        class_encoder (LabelEncoder): encoder object that maps the class labels to integers.
        n_features (int): count of all features after encoding and feature selection,
        preprocessor_step (ColumnTransformer): preprocessing pipeline step that
            transforms features.
        selector_step (VarianceThreshold): feature selection pipeline step that removes
            low variance features.
        model (Callable): model object that implements the fit and predict methods.
        artifacts_path (str): path to save training artificats, e.g., .pkl and .png files.
        num_feature_names (list): list of original numerical feature names.
        cat_feature_names (list, optional): list of original categorical feature names
        (before encoding and feature selection). Defaults to None.
        fbeta_score_beta (float, optional): beta value (weight of recall) in fbeta_score().
            Default to 1 (same as F1).
        encoded_pos_class_label (int, optional): encoded label of positive class using LabelEncoder().
            Default to 1.
        max_search_iters (int, optional): maximum number of iterations for the hyperparameter
            optimization algorithm. Defaults to 100.
        optimize_in_parallel (bool, optional): should optimization be run in parallel. Default
            to False.
        n_parallel_jobs (int, optional): number of parallel jobs to run during the
            hyperparameters optimization. Defaults to 4.
        model_opt_timeout_secs (int, optional): timeout in seconds for each trial of the
            hyperparameters optimization. Defaults to 600.
        registered_model_name (str, optional): name used for the registered model.
        is_voting_ensemble (bool, optional): is it a voting ensemble classifier? This is needed
            for extracting model name in ModelOptimizer class.

    Returns:
        Pipeline: calibrated pipeline object that contains the model transformation pipeline
            and calibrated classifier.
        comet_exp (Experiment): Comet experiment object to be used to access returned model metrics.
    """

    # Extract classifier name
    classifier_name = model.__class__.__name__

    if registered_model_name is None:
        registered_model_name = classifier_name

    # Create a comet experiment
    comet_exp = Experiment(api_key=comet_api_key, project_name=comet_project_name)
    comet_exp.log_code(folder=".")
    comet_exp.set_name(comet_exp_name)

    try:
        #############################################
        # Create optimizer class
        optimizer = ModelOptimizer(
            comet_exp=comet_exp,
            train_features_preprocessed=train_features_preprocessed,
            train_class=train_class,
            valid_features_preprocessed=valid_features_preprocessed,
            valid_class=valid_class,
            n_features=n_features,
            model=model,
            fbeta_score_beta=fbeta_score_beta,
            encoded_pos_class_label=encoded_pos_class_label,
        )

        # Perform hyperparameters optimization procedure
        if optimize_in_parallel:
            study = optimizer.tune_model_in_parallel(
                max_search_iters=max_search_iters,
                n_parallel_jobs=n_parallel_jobs,
                model_opt_timeout_secs=model_opt_timeout_secs,
            )

        else:
            study = optimizer.tune_model(
                max_search_iters=max_search_iters,
                model_opt_timeout_secs=model_opt_timeout_secs,
            )

        # Log trials
        study_results = study.trials_dataframe()
        study_results.rename(
            columns={"value": f"f_{fbeta_score_beta}_score"}, inplace=True
        )
        study_results.rename(columns=lambda x: re.sub("params_", "", x), inplace=True)
        study_results.to_csv(
            f"{artifacts_path}/study_{classifier_name}.csv", index=False
        )
        comet_exp.log_asset(
            file_data=f"{artifacts_path}/study_{classifier_name}.csv",
            file_name=f"study_{classifier_name}",
        )

        #############################################
        # Fit best model with and without calibration (the latter is for feature importance)
        model = model.set_params(**study.best_params)
        fitted_pipeline = optimizer.fit_pipeline(
            train_features=train_features,
            preprocessor_step=preprocessor_step,
            selector_step=selector_step,
            model=model,
        )

        # Log model params
        model_params = {
            k: v
            for k, v in fitted_pipeline.get_params().items()
            if k.startswith("classifier__")
        }
        comet_exp.log_parameters(model_params)

        # Get feature importance for best model

        evaluator = ModelEvaluator(
            comet_exp=comet_exp,
            pipeline=fitted_pipeline,
            train_features=train_features,
            train_class=train_class,
            valid_features=valid_features,
            valid_class=valid_class,
            fbeta_score_beta=fbeta_score_beta,
            is_voting_ensemble=is_voting_ensemble,
        )

        evaluator.extract_feature_importance(
            pipeline=fitted_pipeline,
            num_feature_names=num_feature_names,
            cat_feature_names=cat_feature_names,
            figure_size=(24, 36),
            font_size=10,
        )

        # Evaluate best model on training and validation sets
        (
            train_scores,
            valid_scores,
        ) = evaluator.evaluate_model_perf(
            class_encoder=class_encoder,
        )

        #############################################
        # Log model scores on training and validation sets
        metrics_to_log = {}
        train_metric_values = evaluator.convert_metrics_from_df_to_dict(
            scores=train_scores, prefix="train_"
        )
        metrics_to_log.update(train_metric_values)

        valid_metric_values = evaluator.convert_metrics_from_df_to_dict(
            scores=valid_scores, prefix="valid_"
        )
        metrics_to_log.update(valid_metric_values)

        # Add Expected Calibration Error (ECE) to model metrics
        pred_probs = fitted_pipeline.predict_proba(valid_features)
        pos_class_index = list(fitted_pipeline.classes_).index(encoded_pos_class_label)
        model_ece = evaluator.calc_expected_calibration_error(
            pred_probs=pred_probs[:, pos_class_index],
            true_labels=valid_class,
            nbins=5,
        )
        metrics_to_log.update({"model_ece": model_ece})
        comet_exp.log_metrics(metrics_to_log)

        # Log model
        joblib.dump(fitted_pipeline, f"{artifacts_path}/{registered_model_name}.pkl")
        comet_exp.log_model(
            name=registered_model_name,
            file_or_folder=f"{artifacts_path}/{registered_model_name}.pkl",
            overwrite=False,
        )
        comet_exp.register_model(registered_model_name)

    except Exception as e:  # pylint: disable=W0718
        print(f"\n\nModel training error --> {e}\n\n")
        fitted_pipeline = None

    # End experiment to upload metrics and artifacts to Comet project
    comet_exp.end()

    return fitted_pipeline, comet_exp


def create_voting_ensemble(
    comet_api_key: str,
    comet_project_name: str,
    comet_exp_name: str,
    lr_calib_pipeline: Pipeline,
    rf_calib_pipeline: Pipeline,
    lgbm_calib_pipeline: Pipeline,
    xgb_calib_pipeline: Pipeline,
    train_features: pd.DataFrame,
    valid_features: pd.DataFrame,
    valid_features_preprocessed: pd.DataFrame,
    train_class: np.ndarray,
    valid_class: np.ndarray,
    class_encoder: LabelEncoder,
    artifacts_path: str,
    voting_rule: Literal["hard", "soft"] = "soft",
    encoded_pos_class_label: int = 1,
    cv_folds: int = 5,
    fbeta_score_beta: float = 1.0,
    registered_model_name: str = None,
) -> Pipeline:
    """Creates a voting ensemble model with data transformation pipeline
    given three models (Random Forest, LightGBM, and XGBoost). The created
    pipeline is registred if the number of base estimators are at least three.

    Args:
        comet_api_key (str): Comet API key,
        comet_project_name (str): Comet project name,
        comet_exp_name (str)L Comet experiment name,
        lr_calib_pipeline (Pipeline): pipeline of calibrated Logistic Regression model,
        rf_calib_pipeline (Pipeline): pipeline of calibrated Random Forest model,
        lgbm_calib_pipeline (Pipeline): pipeline of calibrated LightGBM model,
        xgb_calib_pipeline (Pipeline): pipeline of calibrated XGBoost model,
        train_features (pd.DataFrame): original features (not preprocessed) of training set
            for ModelEvaluator class.
        train_class (np.ndarray): The target labels for the training set.
        valid_features_preprocessed (pd.DataFrame): transformed features of validation set.
        valid_features (pd.DataFrame): original features (not preprocessed) of validation set
            for ModelEvaluator class.
        valid_class (np.ndarray): The target labels for the validation set.
        class_encoder (LabelEncoder): encoder object that maps the class labels
        to integers.
        voting_rule (str): voting ensemble startegy. Default to "soft".
        encoded_pos_class_label (int): encoded label of positive class using
            LabelEncoder(). Default to 1.
        artifacts_path (str): path to save training artificats, e.g., .pkl and .png files.
        cv_folds (int, optional): number of folds for cross-validation.
        fbeta_score_beta (float): beta value (weight of recall) in fbeta_score().
            Default to 1 (same as F1).
        registered_model_name (str): name used for the registered model.

    Returns:
        Pipeline: calibrated pipeline object that contains the model data transformation
            pipeline and calibrated classifier.
        comet_exp (Experiment): Comet experiment object to be used to access returned model metrics.
    """

    if registered_model_name is None:
        registered_model_name = "VotingEnsemble"

    # Create a comet experiment
    comet_exp = Experiment(api_key=comet_api_key, project_name=comet_project_name)
    comet_exp.log_code(folder=".")
    comet_exp.set_name(comet_exp_name)

    #############################################
    # Conditionally add each base model to the list
    # Note: some base models may not exist if all its losses are zero.
    base_models = []
    try:
        model_lr = CalibratedClassifierCV(
            estimator=lr_calib_pipeline.named_steps["classifier"],
            method="isotonic" if len(valid_class) > 1000 else "sigmoid",
            cv=cv_folds,
        )
        base_models.append(("LR", model_lr))
    except Exception:  # pylint: disable=W0718
        print("RF model does not exist or not in required type!")

    try:
        model_rf = CalibratedClassifierCV(
            estimator=rf_calib_pipeline.named_steps["classifier"],
            method="isotonic" if len(valid_class) > 1000 else "sigmoid",
            cv=cv_folds,
        )
        base_models.append(("RF", model_rf))
    except Exception:  # pylint: disable=W0718
        print("RF model does not exist or not in required type!")

    try:
        model_lgbm = CalibratedClassifierCV(
            estimator=lgbm_calib_pipeline.named_steps["classifier"],
            method="isotonic" if len(valid_class) > 1000 else "sigmoid",
            cv=cv_folds,
        )
        base_models.append(("LightGBM", model_lgbm))
    except Exception:  # pylint: disable=W0718
        print("LightGBM model does not exist or not in required type!")

    try:
        model_xgb = CalibratedClassifierCV(
            estimator=xgb_calib_pipeline.named_steps["classifier"],
            method="isotonic" if len(valid_class) > 1000 else "sigmoid",
            cv=cv_folds,
        )
        base_models.append(("XGBoost", model_xgb))
    except Exception:  # pylint: disable=W0718
        print("XGBoost model does not exist or not in required type!")

    # Create the ensemble model using the list of base models
    ve_model = VotingClassifier(estimators=base_models, voting=voting_rule)

    #############################################
    # Evaluate voting ensemble model and register it only if count of
    # base models > 1
    if len(base_models) > 1:
        # Copy fitted data transformation steps from any base pipeline
        if "lr_calib_pipeline" in locals():
            ve_pipeline = deepcopy(lr_calib_pipeline)
        elif "rf_calib_pipeline" in locals():
            ve_pipeline = deepcopy(rf_calib_pipeline)
        elif "lgbm_calib_pipeline" in locals():
            ve_pipeline = deepcopy(lgbm_calib_pipeline)
        elif "xgb_calib_pipeline" in locals():
            ve_pipeline = deepcopy(xgb_calib_pipeline)

        # Drop base classifier and recreate pipeline by adding voing ensemble
        # with fitted base classfiers
        _ = ve_pipeline.steps.pop(len(ve_pipeline) - 1)
        ve_pipeline.steps.insert(
            len(ve_pipeline) + 1,
            ["classifier", ve_model],
        )

        # Fit pipeline and calibrate base classifiers
        ve_pipeline.fit(train_features, train_class)
        ve_pipeline.named_steps["classifier"].fit(
            valid_features_preprocessed, valid_class
        )

        # Evaluate voting ensemble on training and validation sets
        evaluator = ModelEvaluator(
            comet_exp=comet_exp,
            pipeline=ve_pipeline,
            train_features=train_features,
            train_class=train_class,
            valid_features=valid_features,
            valid_class=valid_class,
            fbeta_score_beta=fbeta_score_beta,
            is_voting_ensemble=True,
        )

        (
            train_scores,
            valid_scores,
        ) = evaluator.evaluate_model_perf(
            class_encoder=class_encoder,
        )

        #############################################
        # Log training scores metrics
        metrics_to_log = {}
        train_metric_values = evaluator.convert_metrics_from_df_to_dict(
            scores=train_scores, prefix="train_"
        )
        metrics_to_log.update(train_metric_values)

        valid_metric_values = evaluator.convert_metrics_from_df_to_dict(
            scores=valid_scores, prefix="valid_"
        )
        metrics_to_log.update(valid_metric_values)

        # Add model expected calibration error to model metrics
        pred_probs = ve_pipeline.predict_proba(valid_features)
        pos_class_index = list(ve_pipeline.classes_).index(encoded_pos_class_label)
        model_ece = evaluator.calc_expected_calibration_error(
            pred_probs=pred_probs[:, pos_class_index],
            true_labels=valid_class,
            nbins=5,
        )
        metrics_to_log.update({"model_ece": model_ece})
        comet_exp.log_metrics(metrics_to_log)

        # Log model
        joblib.dump(ve_pipeline, f"{artifacts_path}/{registered_model_name}.pkl")
        comet_exp.log_model(
            name=registered_model_name,
            file_or_folder=f"{artifacts_path}/{registered_model_name}.pkl",
            overwrite=False,
        )
        comet_exp.register_model(registered_model_name, tags=metrics_to_log)

    else:
        ve_pipeline = None
        print("Voting Ensemble model couldn't be created.")

    # End experiment to upload metrics and artifacts to Comet project
    comet_exp.end()

    return ve_pipeline, comet_exp
