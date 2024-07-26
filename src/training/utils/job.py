"""
Includes functions to submit training job, where the model is trained and evaluated
using ModelOptimizer and ModelEvaluator classes. It also includes a class to create
a voting ensemble classifier using the base models and evaluate the model using
ModelEvaluator.
"""

import os
import re
import subprocess
from copy import deepcopy
from datetime import datetime
from typing import Callable, Literal, Optional, Union

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import sklearn
from azureml.core import Environment, Workspace
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.data.tabular_dataset import TabularDataset
from mlflow.models import infer_signature
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.training.utils.model import ModelEvaluator, ModelOptimizer
from src.utils.logger import get_console_logger

###########################################################
# Get console logger
logger = get_console_logger("job_logger")


class ModelTrainer:
    """Trains an sklearn classifier using ModelOptimizer and ModelEvaluator classes
    and evaluates the model and logs its metrics to experiment.

    Attributes:
        train_features (pd.DataFrame): features of training set before encoding and
            feature selection for ModelEvaluator class.
        train_class (np.ndarray): The target labels for the training set.
        valid_features (pd.DataFrame): features of validation set before encoding and
            feature selection for ModelEvaluator class.
        valid_class (np.ndarray): The target labels for the validation set.
        train_features_preprocessed (pd.DataFrame): features of training set after encoding and
            feature selection for ModelOptimizer class.
        valid_features_preprocessed (pd.DataFrame): features of validation set after encoding and
            feature selection for ModelOptimizer class.
        n_features (int): number of features in the training set.
        class_encoder (LabelEncoder): encoder object that maps the class labels to integers.
        preprocessor_step (ColumnTransformer): preprocessor step in the pipeline.
        selector_step (VarianceThreshold): selector step in the pipeline.
        registered_train_set (TabularDataset): registered training set in the workspace.
        registered_test_set (TabularDataset): registered test set in the workspace.
        artifacts_path (str): path to save training artificats, e.g., .pkl and .png files.
        num_feature_names (Optional[list]): list of numerical feature names.
        cat_feature_names (Optional[list]): list of categorical feature names.
        fbeta_score_beta (float): beta value (weight of recall) in fbeta_score().
        encoded_pos_class_label (int): encoded label of positive class using LabelEncoder().
        conda_env (str): path to conda environment file.
        max_search_iters (int): maximum number of iterations for the hyperparameter optimization
            algorithm.
        optimize_in_parallel (bool): should optimization be run in parallel.
        n_parallel_jobs (int): number of parallel jobs to run during the hyperparameters optimization.
        model_opt_timeout_secs (int): timeout in seconds for each trial of the hyperparameters
            optimization.
        conf_score_threshold_val (float): decision threshold value.
        cv_folds (int): number of cross-validation folds for calibration.
    """

    def __init__(
        self,
        train_features: pd.DataFrame,
        train_class: np.ndarray,
        valid_features: pd.DataFrame,
        valid_class: np.ndarray,
        train_features_preprocessed: pd.DataFrame,
        valid_features_preprocessed: pd.DataFrame,
        n_features: int,
        class_encoder: LabelEncoder,
        preprocessor_step: ColumnTransformer,
        selector_step: VarianceThreshold,
        registered_train_set: TabularDataset,
        registered_test_set: TabularDataset,
        artifacts_path: str = "tmp",
        num_feature_names: Optional[list] = None,
        cat_feature_names: Optional[list] = None,
        fbeta_score_beta: float = 1.0,
        encoded_pos_class_label: int = 1,
        conda_env: str = "train_env.yml",
        max_search_iters: int = 50,
        optimize_in_parallel: bool = False,
        n_parallel_jobs: int = 1,
        model_opt_timeout_secs: int = 600,
        conf_score_threshold_val: float = 0.5,
        cv_folds: int = 5,
    ):
        self.train_features = train_features
        self.train_class = train_class
        self.valid_features = valid_features
        self.valid_class = valid_class
        self.train_features_preprocessed = train_features_preprocessed
        self.valid_features_preprocessed = valid_features_preprocessed
        self.n_features = n_features
        self.class_encoder = class_encoder
        self.preprocessor_step = preprocessor_step
        self.selector_step = selector_step
        self.registered_train_set = registered_train_set
        self.registered_test_set = registered_test_set
        self.artifacts_path = artifacts_path
        self.num_feature_names = num_feature_names
        self.cat_feature_names = cat_feature_names
        self.fbeta_score_beta = fbeta_score_beta
        self.encoded_pos_class_label = encoded_pos_class_label
        self.conda_env = conda_env
        self.max_search_iters = max_search_iters
        self.optimize_in_parallel = optimize_in_parallel
        self.n_parallel_jobs = n_parallel_jobs
        self.model_opt_timeout_secs = model_opt_timeout_secs
        self.conf_score_threshold_val = conf_score_threshold_val
        self.cv_folds = cv_folds

    def _optimize_model(
        self,
        model: Callable,
        search_space_params: dict,
        max_search_iters: int = 100,
        optimize_in_parallel: bool = False,
        n_parallel_jobs: int = 4,
        model_opt_timeout_secs: int = 600,
        is_voting_ensemble: bool = False,
    ) -> Union[optuna.study.Study, ModelOptimizer]:
        """Optimizes the model using ModelOptimizer class.

        Args:
            model (Callable): model object that implements the fit and predict methods,
            search_space_params (dict): hyperparameter search space for the model,
            max_search_iters (int, optional): maximum number of iterations for the hyperparameter
                optimization algorithm. Default to 100,
            optimize_in_parallel (bool, optional): should optimization be run in parallel. Defaults
                to False,
            n_parallel_jobs (int, optional): number of parallel jobs to run during the
                hyperparameters optimization. Default to 4,
            model_opt_timeout_secs (int, optional): timeout in seconds for each trial of the
                hyperparameters optimization. Default to 600.
            is_voting_ensemble (bool): is it a voting ensemble classifier? This is needed
                for extracting model name in ModelOptimizer class. Default to False.

        Returns:
            study (optuna.study.Study): Optuna study object and ModelOptimizer object.
            optimizer (ModelOptimizer): ModelOptimizer object.
        """

        optimizer = ModelOptimizer(
            train_features_preprocessed=self.train_features_preprocessed,
            train_class=self.train_class,
            valid_features_preprocessed=self.valid_features_preprocessed,
            valid_class=self.valid_class,
            n_features=self.n_features,
            model=model,
            search_space_params=search_space_params,
            fbeta_score_beta=self.fbeta_score_beta,
            encoded_pos_class_label=self.encoded_pos_class_label,
            is_voting_ensemble=is_voting_ensemble,
        )

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

        return study, optimizer

    def _log_study_trials(
        self,
        study: optuna.study.Study,
        classifier_name: str,
    ) -> None:
        """Logs Optuna study results to experiment.

        Args:
            study (optuna.study.Study): Optuna study object,
            classifier_name (str): name of the classifier.
        """

        study_results = study.trials_dataframe()
        study_results.rename(
            columns={"value": f"f_{self.fbeta_score_beta}_score"}, inplace=True
        )
        study_results.rename(columns=lambda x: re.sub("params_", "", x), inplace=True)
        study_results.to_csv(
            f"/{self.artifacts_path}/study_{classifier_name}.csv", index=False
        )
        mlflow.log_artifact(
            f"/{self.artifacts_path}/study_{classifier_name}.csv",
        )

    def _fit_best_model(
        self,
        study: optuna.study.Study,
        optimizer: ModelOptimizer,
        model: Callable,
    ) -> Pipeline:
        """Fits the best model found by the optimizer.

        Args:
            study (optuna.study.Study): Optuna study object,
            optimizer (ModelOptimizer): ModelOptimizer object,
            model (Callable): model object.

        Returns:
            Pipeline: fitted pipeline object.
        """

        model = model.set_params(**study.best_params)
        fitted_pipeline = optimizer.fit_pipeline(
            train_features=self.train_features,
            preprocessor_step=self.preprocessor_step,
            selector_step=self.selector_step,
            model=model,
        )
        return fitted_pipeline

    def _evaluate_model(
        self,
        fitted_pipeline: Pipeline,
        is_voting_ensemble: bool = False,
        ece_nbins: int = 5,
    ) -> Union[dict, dict, float]:
        """Evaluates the model using ModelEvaluator class.

        Args:
            fitted_pipeline (Pipeline): fitted pipeline object,
            is_voting_ensemble (bool, optional): is it a voting ensemble classifier? Default to False,
            ece_nbins (int, optional): number of bins for expected calibration error. Default to 5.

        Returns:
            train_metric_values (dict): training scores,
            valid_metric_values (dict): validation scores,
            model_ece (float): expected calibration error.
        """

        # Evaluate model performance on training and validation sets
        evaluator = ModelEvaluator(
            pipeline=fitted_pipeline,
            train_features=self.train_features,
            train_class=self.train_class,
            valid_features=self.valid_features,
            valid_class=self.valid_class,
            encoded_pos_class_label=self.encoded_pos_class_label,
            fbeta_score_beta=self.fbeta_score_beta,
            is_voting_ensemble=is_voting_ensemble,
        )

        train_scores, valid_scores = evaluator.evaluate_model_perf(
            class_encoder=self.class_encoder
        )

        # Plot feature importance and log it to experiment
        evaluator.extract_feature_importance(
            pipeline=fitted_pipeline,
            num_feature_names=self.num_feature_names,
            cat_feature_names=self.cat_feature_names,
            figure_size=(24, 36),
            font_size=10,
        )

        # Convert metrics from dataframes to dictionaries
        train_metric_values = evaluator.convert_metrics_from_df_to_dict(
            scores=train_scores, prefix="train_"
        )
        valid_metric_values = evaluator.convert_metrics_from_df_to_dict(
            scores=valid_scores, prefix="valid_"
        )

        # Add ECE to logged metrics
        pred_probs = fitted_pipeline.predict_proba(self.valid_features)
        pos_class_index = list(fitted_pipeline.classes_).index(
            self.encoded_pos_class_label
        )
        model_ece = evaluator.calc_expected_calibration_error(
            pred_probs=pred_probs[:, pos_class_index],
            true_labels=self.valid_class,
            nbins=ece_nbins,
        )

        return train_metric_values, valid_metric_values, model_ece

    def _log_model_metrics(
        self,
        train_metric_values: dict,
        valid_metric_values: dict,
        model_ece: float,
    ) -> None:
        """Logs model metrics to experiment.

        Args:
            evaluator (ModelEvaluator): ModelEvaluator object,
            train_metric_values (dict): training scores,
            valid_metric_values (dict): validation scores,
            model_ece (float): expected calibration error.
        """

        metrics_to_log = {}
        metrics_to_log.update(train_metric_values)
        metrics_to_log.update(valid_metric_values)
        metrics_to_log.update({"model_ece": model_ece})

        mlflow.log_metrics(metrics_to_log)

    @staticmethod
    def create_model_tags(
        model_name: str,
        pipeline: Pipeline,
        decoded_class_labels: list,
        registered_train_set: TabularDataset,
        registered_test_set: TabularDataset,
        train_scores: pd.DataFrame,
        valid_scores: pd.DataFrame,
        conf_score_threshold_val: float,
        parent_run_id: str,
        child_run_id: str,
    ) -> dict:
        """Creates model tags as dictionary to be added to model info when
        it's registered. Note that 'Description' is included in tags because
        mlflow.register_model (mlflow version: 2.8.1) doesn't have an argument
        to add description.

        Args:
            model_name (str): name of the classifier,
            pipeline (Pipeline): fitted pipeline object,
            decoded_class_labels (list): list of class labels,
            registered_train_set (TabularDataset): registered training set in the workspace,
            registered_test_set (TabularDataset): registered test set in the workspace,
            train_scores (pd.DataFrame): training scores,
            valid_scores (pd.DataFrame): validation scores,
            conf_score_threshold_val (float): decision threshold value,
            parent_run_id (str): parent run ID for the experiment,
            child_run_id (str): child run ID for the experiment,

        Returns:
            model_tags (dict): model tags.
        """

        # Create model tags
        model_tags = {
            "Name": model_name,
            "Registration Timestamp": datetime.now().strftime("%Y-%d-%m %H:%M:%S"),
            "scikit-learn Version": sklearn.__version__,
            "Classes": decoded_class_labels,
            "Training Set Name": registered_train_set.name,
            "Training Set Version": registered_train_set.version,
            "Testing Set Name": registered_test_set.name,
            "Testing Set Version": registered_test_set.version,
            "Training Scores": train_scores,
            "Testing Scores": valid_scores,
            "Decision Threshold Value": conf_score_threshold_val,
            "Parent Exp. Run ID": parent_run_id,
            "Child Exp. Run ID": child_run_id,
        }

        # Check if classifier name is VotingClassifier to split Model Parameters
        # Note: model registration could fail for Voting Ensemble becuase Model
        # Parameters is too long. So it has to be split into each base classifier.
        model_tags["Model Parameters"] = str(
            {
                k: v
                for k, v in pipeline.get_params().items()
                if k.startswith("classifier__")
            }
        )

        return model_tags

    def _register_model(
        self,
        pipeline: Pipeline,
        classifier_name: str,
        registered_model_name: str,
        model_uri: str,
        tags: dict,
    ) -> None:
        """Registers the model in the experiment using MLflow.

        Args:
            pipeline (Pipeline): fitted pipeline object,
            classifier_name (str): name of the classifier,
            registered_model_name (str): name used for the registered model,
            model_uri (str): URI of the model,
            tags (dict): additional tags for the model,
        """

        joblib.dump(pipeline, f"/{self.artifacts_path}/{classifier_name}.pkl")
        signature = infer_signature(
            self.train_features, pipeline.predict(self.train_features.iloc[0:10, :])
        )
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",  # Model folder in the experiment UI
            signature=signature,
            input_example=self.train_features.iloc[0:10, :],
            conda_env=self.conda_env,
        )
        mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name,
            tags=tags,
        )

    @staticmethod
    def calibrate_pipeline(
        fitted_pipeline: Pipeline,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        cv_folds: int = 5,
    ) -> Pipeline:
        """Takes a fitted pipeline and returns a calibrated pipeline.

        Args:
            fitted_pipeline (Pipeline): Fitted pipeline on the training set.
            X_valid (np.ndarray): Validation features.
            y_valid (np.ndarray): Validation class labels.
            cv_folds (int): Number of cross-validation folds for calibration.

        Returns:
            calib_pipeline (Pipeline): Calibrated pipeline.

        Raises:
            ValueError: if fitted_pipeline is not fitted.
        """

        # Extract preprocessor, selector, and classifier from the fitted pipeline
        preprocessor = fitted_pipeline.named_steps.get("preprocessor")
        selector = fitted_pipeline.named_steps.get("selector")
        model = fitted_pipeline.named_steps.get("classifier")

        if not hasattr(model, "classes_"):
            raise ValueError("The classifier in the fitted pipeline is not fitted.")

        # Calibrate the newly fitted model using the validation set
        calibrator = CalibratedClassifierCV(
            base_estimator=model,
            method=("isotonic" if len(y_valid) > 1000 else "sigmoid"),
            cv=cv_folds,  # Indicate that the model is already fitted
        )

        # Fit the calibrator on the validation set
        calibrator.fit(X_valid, y_valid)

        # Create a new pipeline with the calibrated classifier
        calib_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("selector", selector),
                ("classifier", calibrator),
            ]
        )

        return calib_pipeline

    def submit_train_exp(
        self,
        parent_run_id: str,
        child_run_id: str,
        model: Callable,
        search_space_params: dict,
        registered_model_name: Optional[str] = None,
        is_voting_ensemble: bool = False,
        ece_nbins: int = 5,
    ) -> Pipeline:
        """Submits a training experiment to workspace. This is a wrapper function
        that uses ModelOptimizer and ModelEvaluator classes to tune and evaluate the model,
        and logs the model and its training metrics to experiment. While the class instance
        is used to store the training data and model transformation pipeline, this function
        can be used to submit the training job for different models that use the same training
        and validation sets and data tranformation pipeline.

        Args:
            parent_run_id (str): parent run ID for the experiment,
            child_run_id (str): child run ID for the experiment,
            model (Callable): model object that implements the fit and predict methods.
            search_space_params (dict): hyperparameter search space for the model.
            registered_model_name (str, optional): name used for the registered model.
            is_voting_ensemble (bool, optional): is it a voting ensemble classifier? This is needed
                for extracting model name in ModelOptimizer class.
            ece_nbins (int, optional): number of bins for expected calibration error. Default to 5.

        Returns:
            Pipeline: calibrated pipeline object that contains the model transformation pipeline
                and calibrated classifier.

        Raises:
            Exception: if model training fails.
        """

        # Extract the classifier name from the model object if not provided
        classifier_name = model.__class__.__name__
        if registered_model_name is None:
            registered_model_name = classifier_name

        try:
            # Tune model
            study, optimizer = self._optimize_model(
                model=model,
                search_space_params=search_space_params,
                max_search_iters=self.max_search_iters,
                optimize_in_parallel=self.optimize_in_parallel,
                n_parallel_jobs=self.n_parallel_jobs,
                model_opt_timeout_secs=self.model_opt_timeout_secs,
                is_voting_ensemble=is_voting_ensemble,
            )

            # Log study trials
            self._log_study_trials(
                study=study,
                classifier_name=classifier_name,
            )

            # Fit best model pipeline
            fitted_pipeline = self._fit_best_model(
                study=study,
                optimizer=optimizer,
                model=model,
            )

            # Log model parameters
            model_params = {
                k: v
                for k, v in fitted_pipeline.get_params().items()
                if k.startswith("classifier__")
            }
            mlflow.log_params(model_params)

            # Evaluate best model
            train_metric_values, valid_metric_values, model_ece = self._evaluate_model(
                fitted_pipeline=fitted_pipeline,
                is_voting_ensemble=is_voting_ensemble,
                ece_nbins=ece_nbins,
            )

            # Log model metrics
            self._log_model_metrics(
                train_metric_values=train_metric_values,
                valid_metric_values=valid_metric_values,
                model_ece=model_ece,
            )

            # Calibrate model
            pipeline = self.calibrate_pipeline(
                X_valid=self.valid_features_preprocessed,
                y_valid=self.valid_class,
                fitted_pipeline=fitted_pipeline,
                cv_folds=self.cv_folds,
            )

            # Save and register model
            model_tags = self.create_model_tags(
                model_name=classifier_name,
                pipeline=pipeline,
                decoded_class_labels=self.class_encoder.inverse_transform(
                    range(len(self.class_encoder.classes_))
                ),
                registered_train_set=self.registered_train_set,
                registered_test_set=self.registered_test_set,
                train_scores=train_metric_values,
                valid_scores=valid_metric_values,
                conf_score_threshold_val=self.conf_score_threshold_val,
                parent_run_id=parent_run_id,
                child_run_id=child_run_id,
            )
            self._register_model(
                pipeline=pipeline,
                classifier_name=classifier_name,
                registered_model_name=registered_model_name,
                model_uri=f"runs:/{child_run_id}/model",
                tags=model_tags,
            )

            logger.info(f"{classifier_name} model registered successfully!")

        except Exception as e:  # pylint: disable=W0718
            logger.info(f"\n\nModel training error --> {e}\n\n")
            pipeline = None

        return pipeline


class VotingEnsembleCreator(ModelTrainer):
    """Creates a voting ensemble classifier. It uses ModelEvaluator class to
    evaluate the model and logs the model and its metrics to experiment.
    It is a subclass of ModelTrainer class and utilizes some of its methods like
    create_experiment, evaluate_model, log_model_metrics,

    Attributes:
        parent_run_id (str): parent run ID for the experiment.
        child_run_id (str): child run ID for the experiment.
        train_features (pd.DataFrame): features of training set before encoding and
            feature selection for ModelEvaluator class.
        valid_features (pd.DataFrame): features of validation set before encoding and
            feature selection for ModelEvaluator class.
        train_class (np.ndarray): The target labels for the training set.
        valid_class (np.ndarray): The target labels for the validation set.
        class_encoder (LabelEncoder): encoder object that maps the class labels to integers.
        artifacts_path (str): path to save training artificats, e.g., .pkl and .png files.
        registered_train_set (TabularDataset): registered training set in the workspace.
        registered_test_set (TabularDataset): registered test set in the workspace.
        lr_pipeline (Pipeline): calibrated pipeline for logistic regression model,
        rf_pipeline (Pipeline): calibrated pipeline for random forest model,
        lgbm_pipeline (Pipeline): calibrated pipeline for LightGBM model,
        xgb_pipeline (Pipeline): calibrated pipeline for XGBoost model,
        voting_rule (Literal["hard", "soft"], optional): voting rule for the ensemble classifier.
            Default to "soft".
        encoded_pos_class_label (int, optional): encoded label of positive class using LabelEncoder().
            Default to 1.
        fbeta_score_beta (float, optional): beta value (weight of recall) in fbeta_score().
            Default to 1 (same as F1).
        registered_model_name (str, optional): name used for the registered model.
        ece_nbins (int, optional): number of bins for expected calibration error. Default to 5.
        conf_score_threshold_val (float, optional): decision threshold value. Default to 0.5.
    """

    def __init__(
        self,
        parent_run_id: str,
        child_run_id: str,
        train_features: pd.DataFrame,
        valid_features: pd.DataFrame,
        train_class: np.ndarray,
        valid_class: np.ndarray,
        class_encoder: LabelEncoder,
        artifacts_path: str,
        registered_train_set: TabularDataset,
        registered_test_set: TabularDataset,
        lr_pipeline: Optional[Pipeline] = None,
        rf_pipeline: Optional[Pipeline] = None,
        lgbm_pipeline: Optional[Pipeline] = None,
        xgb_pipeline: Optional[Pipeline] = None,
        voting_rule: Literal["hard", "soft"] = "soft",
        encoded_pos_class_label: int = 1,
        conda_env: str = "train_env.yml",
        fbeta_score_beta: float = 1.0,
        registered_model_name: Optional[str] = None,
        ece_nbins: int = 5,
        conf_score_threshold_val: float = 0.5,
        cv_folds: int = 5,
    ):
        super().__init__(
            train_features=train_features,
            train_class=train_class,
            valid_features=valid_features,
            valid_class=valid_class,
            train_features_preprocessed=None,
            valid_features_preprocessed=None,
            n_features=None,
            class_encoder=class_encoder,
            preprocessor_step=None,
            selector_step=None,
            registered_train_set=registered_train_set,
            registered_test_set=registered_test_set,
            artifacts_path=artifacts_path,
            num_feature_names=None,
            cat_feature_names=None,
            fbeta_score_beta=fbeta_score_beta,
            encoded_pos_class_label=encoded_pos_class_label,
            conda_env=conda_env,
            cv_folds=cv_folds,
        )

        self.parent_run_id = parent_run_id
        self.child_run_id = child_run_id
        self.lr_pipeline = lr_pipeline
        self.rf_pipeline = rf_pipeline
        self.lgbm_pipeline = lgbm_pipeline
        self.xgb_pipeline = xgb_pipeline
        self.voting_rule = voting_rule
        self.registered_model_name = registered_model_name or "VotingEnsemble"
        self.conf_score_threshold_val = conf_score_threshold_val
        self.ece_nbins = ece_nbins

    def _get_base_models(
        self,
    ) -> list:
        """Creates a list of base models for the voting ensemble.

        Returns:
            base_models (list): list of base models.

        Raises:
            ValueError: if less than two base models are provided.
        """

        # Conditionally add each base model to the list
        # Note: some base models may not exist if all its losses are zero.
        base_models = []
        try:
            model_lr = self.lr_pipeline.named_steps["classifier"]
            base_models.append(("LR", model_lr))
        except Exception:  # pylint: disable=W0718
            logger.info("RF model does not exist or not in required type!")

        try:
            model_rf = self.rf_pipeline.named_steps["classifier"]
            base_models.append(("RF", model_rf))
        except Exception:  # pylint: disable=W0718
            logger.info("RF model does not exist or not in required type!")

        try:
            model_lgbm = self.lgbm_pipeline.named_steps["classifier"]
            base_models.append(("LightGBM", model_lgbm))
        except Exception:  # pylint: disable=W0718
            logger.info("LightGBM model does not exist or not in required type!")

        try:
            model_xgb = self.xgb_pipeline.named_steps["classifier"]
            base_models.append(("XGBoost", model_xgb))
        except Exception:  # pylint: disable=W0718
            logger.info("XGBoost model does not exist or not in required type!")

        if len(base_models) < 2:
            raise ValueError(
                "At least two base models are needed for a voting ensemble."
            )

        return base_models

    def _copy_data_transform_pipeline(
        self,
    ) -> Pipeline:
        """Copies (deep copy) the data transformation pipeline from the first base
        model. It assumes all base models have the same data transformation pipeline.

        Returns:
            data_pipeline (Pipeline): data transformation pipeline object.

        Raises:
            ValueError: if no base model pipelines are found.
        """

        # Copy fitted data transformation steps from any base pipeline
        if hasattr(self, "lr_pipeline") and self.lr_pipeline is not None:
            data_pipeline = deepcopy(self.lr_pipeline)
        elif hasattr(self, "rf_pipeline") and self.lr_pipeline is not None:
            data_pipeline = deepcopy(self.rf_pipeline)
        elif hasattr(self, "lgbm_pipeline") and self.lr_pipeline is not None:
            data_pipeline = deepcopy(self.lgbm_pipeline)
        elif hasattr(self, "xgb_pipeline") and self.lr_pipeline is not None:
            data_pipeline = deepcopy(self.xgb_pipeline)
        else:
            raise ValueError("No base model pipelines found!")

        return data_pipeline

    def _create_fitted_ensemble_pipeline(
        self,
        base_models: list,
    ) -> Pipeline:
        """Creates a voting ensemble pipeline (data transformation pipeline and
        base models) and fits the pipeline to the training set.

        Args:
            base_models (list): list of base models,

        Returns:
            ve_pipeline (Pipeline): fitted voting ensemble pipeline.

        Raises:
            ValueError: if less than two base models are provided.
        """

        ve_model = VotingClassifier(estimators=base_models, voting=self.voting_rule)
        if len(base_models) > 1:
            # Copy data transformation pipeline from a base model (all must have same data pipeline)
            ve_pipeline = self._copy_data_transform_pipeline()

            # Drop base classifier and recreate pipeline by adding voing ensemble
            # with fitted base classfiers
            _ = ve_pipeline.steps.pop(len(ve_pipeline) - 1)
            ve_pipeline.steps.insert(
                len(ve_pipeline) + 1,
                ["classifier", ve_model],
            )
            ve_pipeline.fit(self.train_features, self.train_class)

        else:
            raise ValueError(
                "At least two base models are needed for a voting ensemble."
            )

        return ve_pipeline

    def create_voting_ensemble(
        self,
    ) -> Pipeline:
        """Creates a voting ensemble classifier using the base models and evaluates the model
        using ModelEvaluator class. It logs the model metrics to experiment.

        Returns:
            Pipeline: calibrated pipeline object that contains the model transformation pipeline
                and calibrated classifier.

        Raises:
            Exception: if model training fails.
        """

        try:
            # Create voting ensemble pipeline (data transformation pipeline and base models)
            base_models = self._get_base_models()
            ve_pipeline = self._create_fitted_ensemble_pipeline(base_models)
            logger.info("Voting ensemble model created successfully!")

            # Evaluate voting ensemble classifier
            train_metric_values, valid_metric_values, model_ece = self._evaluate_model(
                fitted_pipeline=ve_pipeline,
                is_voting_ensemble=True,
                ece_nbins=self.ece_nbins,
            )

            self._log_model_metrics(
                train_metric_values=train_metric_values,
                valid_metric_values=valid_metric_values,
                model_ece=model_ece,
            )
            logger.info("Voting ensemble model metrics logged successfully!")

            model_tags = self.create_model_tags(
                model_name=ve_pipeline.named_steps["classifier"].__class__.__name__,
                pipeline=ve_pipeline,
                decoded_class_labels=self.class_encoder.inverse_transform(
                    range(len(self.class_encoder.classes_))
                ),
                registered_train_set=self.registered_train_set,
                registered_test_set=self.registered_test_set,
                train_scores=train_metric_values,
                valid_scores=valid_metric_values,
                parent_run_id=self.parent_run_id,
                child_run_id=self.child_run_id,
                conf_score_threshold_val=self.conf_score_threshold_val,
            )

            ve_pipeline = self.calibrate_pipeline(
                X_valid=self.valid_features,
                y_valid=self.valid_class,
                fitted_pipeline=ve_pipeline,
                cv_folds=self.cv_folds,
            )

            self._register_model(
                pipeline=ve_pipeline,
                classifier_name={
                    ve_pipeline.named_steps["classifier"].__class__.__name__
                },
                registered_model_name=self.registered_model_name,
                model_uri=f"runs:/{self.child_run_id}/model",
                tags=model_tags,
            )
            logger.info("Voting ensemble model metrics registered successfully!")

        except Exception as e:  # pylint: disable=W0718
            logger.info(f"\nVoting ensemble error --> {e}\n\n")
            ve_pipeline = None

        return ve_pipeline


class EnvCreator:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def create_compute_cluster(
        self,
        aml_clust_name: str,
        aml_clust_vm_type: str = "STANDARD_D16S_V3",
        aml_clust_vm_priority: str = "lowpriority",
        aml_clust_max_node_no: int = 1,
        aml_clust_min_node_no: int = 0,
        aml_clust_idle_secs_scaledown: int = 120,
        aml_clust_identity_type: str = "SystemAssigned",
    ) -> ComputeTarget:
        """Creates Azure ML compute cluster if they don't exist.

        Args:
            aml_clust_name (str): The name of the compute cluster.
            aml_clust_vm_type (str): The VM type of the cluster.
            aml_clust_vm_priority (str): The VM priority of the cluster.
            aml_clust_max_node_no (int): The maximum number of nodes in the cluster.
            aml_clust_min_node_no (int): The minimum number of nodes in the cluster.
            aml_clust_idle_secs_scaledown (int): The number of idle seconds before the cluster scales down.
            aml_clust_identity_type (str): The identity type of the cluster.

        Returns:
            aml_compute_clust (ComputeTarget): The created or existing compute target.
        """

        try:
            aml_compute_clust = ComputeTarget(
                workspace=self.workspace, name=aml_clust_name
            )
            logger.info(f"Compute cluster {aml_clust_name} exists and it will be used.")

        except ComputeTargetException:
            compute_config = AmlCompute.provisioning_configuration(
                vm_size=aml_clust_vm_type,
                vm_priority=aml_clust_vm_priority,
                max_nodes=aml_clust_max_node_no,
                min_nodes=aml_clust_min_node_no,
                idle_seconds_before_scaledown=aml_clust_idle_secs_scaledown,
                identity_type=aml_clust_identity_type,
            )
            aml_compute_clust = ComputeTarget.create(
                self.workspace, aml_clust_name, compute_config
            )
            aml_compute_clust.wait_for_completion(show_output=True)

        return aml_compute_clust

    def is_env_updated(
        self,
        env_name: str,
        conda_yml_path: str,
    ) -> bool:
        """Checks if registered env in Azure workspace has identical dependencies to that
        of a local conda file. It returns a True/False flag indicating whether to recreate
        env or not.

        Args:
            env_name (str): name of the registered environment in Azure ML workspace,
            conda_yml_path (str): path to the local conda yaml file.

        Returns:
            local_and_remote_env_mismatch (bool): flag indicating whether to recreate env or not.
        """

        try:
            # Import training environment object from a Conda specification file
            # Note: Environment.from_conda_specification returns broad Exception
            # rather than specific exception. Thus, pylint must ignore it.
            local_env_yml = Environment.from_conda_specification(
                name=env_name, file_path=conda_yml_path
            )

        except Exception as e:  # pylint: disable=W0718
            logger.error(f"Local conda file {conda_yml_path} does not exist! --> {e}")

        try:
            # Get latest version of training environment conda file
            registered_env = Environment.get(
                workspace=self.workspace, name=env_name, version=None
            )

        except UnboundLocalError:
            logger.error(f"Registered env {env_name} does not exist!")

        # Extract dependencies from imported yaml files
        local_env_depends = (
            local_env_yml.python.conda_dependencies.serialize_to_string()
        )
        registered_env_depends = (
            registered_env.python.conda_dependencies.serialize_to_string()
        )

        # Split serialized dependencies on newline characters
        local_env_depends_lines = local_env_depends.split("\n")
        registered_env_depends_lines = registered_env_depends.split("\n")

        # Extract only the lines that represent dependencies
        local_env_depends = [
            line
            for line in local_env_depends_lines
            if not line.startswith("#") and line.strip()
        ]
        registered_env_depends = [
            line
            for line in registered_env_depends_lines
            if not line.startswith("#") and line.strip()
        ]

        # Remove whitespace and leading hyphens from each dependency
        local_env_depends = [
            dep.replace(" ", "").replace("- ", "") for dep in local_env_depends
        ]
        registered_env_depends = [
            dep.replace(" ", "").replace("- ", "") for dep in registered_env_depends
        ]

        logger.info(f"Required dependencies for model training: {local_env_depends}\n")
        logger.info(
            f"Dependencies in latest version of registered training environment: {registered_env_depends}\n"
        )

        # Compare dependencies
        common_depends_in_both_envs = set(local_env_depends) & set(
            registered_env_depends
        )
        only_in_local_env = set(local_env_depends) - set(registered_env_depends)
        only_in_registered_env = set(registered_env_depends) - set(local_env_depends)
        logger.info(f"Common dependencies: {common_depends_in_both_envs}")
        logger.info(f"Dependencies only in data update env: {only_in_local_env}")
        logger.info(
            f"Dependencies only in the latest version of registered data update env: {only_in_registered_env}"
        )

        local_and_remote_env_mismatch = False
        if len(only_in_local_env) > 0 or len(only_in_registered_env) > 0:
            local_and_remote_env_mismatch = True
            logger.warning("Training environment needs to be updated!")

        return local_and_remote_env_mismatch

    def create_or_update_existing_env(
        self,
        env_name: str,
        conda_yml_path: str,
        env_base_image_path: str,
        wait_until_build_completed: bool = False,
    ) -> None:
        """Creates or updates an environment from conda specifications (yaml file "env_name"
        in local path "conda_yml_path") and register it in Azure ML workspace. It
        gives the option to wait until docker image build is completed (default: False)
        if you need to make sure the image build is finished before submitting script
        to compute cluster. Container registry image path "env_base_image_path" can be
        found when editing Parent Image path in registered env. This method can be used
        when training script is updated during dev and training env needs to be updated
        in prod during CI instead of updating environment manually.

        Args:
            env_name (str): name of the registered environment in Azure ML workspace,
            conda_yml_path (str): path to the local conda yaml file,
            env_base_image_path (str): container registry image path of the registered env,
            wait_until_build_completed (bool, optional): wait until docker image build is completed.
        """

        # Create local environment object from a Conda specification file
        env_obj = Environment.from_conda_specification(
            name=env_name, file_path=conda_yml_path
        )

        # Update existing environment
        # docker_config = DockerConfiguration(use_docker=True)
        env_obj.docker.base_image = env_base_image_path
        env_obj.register(self.workspace)
        env_docker_image_build = env_obj.build(self.workspace)
        if wait_until_build_completed == True:
            env_docker_image_build.wait_for_completion(show_output=True)

    @staticmethod
    def build_and_push_docker_image(
        acr_name: str,
        acr_username: str,
        acr_password: str,
        image_name: str,
        dockerfile_local_path: str,
    ) -> None:
        """
        Build and push Docker image to Azure Container Registry to be used by in job submission in Azure DevOps.

        Args:
            acr_name (str): The name of the Azure Container Registry.
            acr_username (str): The username for the Azure Container Registry, which is usually dev service principal.
            acr_password (str): The password for the Azure Container Registry, which is usually dev service principal.
            image_name (str): The name of the Docker image.
            dockerfile_local_path (str): The path to the Dockerfile.
        """

        # Upgrade Azure CLI
        os.system("az upgrade --yes")

        # Upgrade some packages to avoid issues with Azure CLI
        subprocess.run(
            [
                "pip",
                "install",
                "--upgrade",
                "azure-cli",
                "azure-mgmt-resource",
                "azure-core",
            ]
        )

        # Login to Azure Container Registry and build image and push to ACR
        # os.system(f"az acr login --name {acr_name}")
        os.system(
            f"az acr login --name {acr_name} --username {acr_username} --password {acr_password}"
        )
        os.system(
            f"docker build -t {acr_name}.azurecr.io/{image_name} -f {dockerfile_local_path} ."
        )
        os.system(f"docker push {acr_name}.azurecr.io/{image_name}")
