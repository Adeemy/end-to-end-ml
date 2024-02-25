"""
This utility module includes functions to submit
training job.
"""

import re
from copy import deepcopy
from typing import Callable, Literal, Optional, Union

import joblib
import numpy as np
import optuna
import pandas as pd
from comet_ml import Experiment
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.training.utils.model import ModelEvaluator, ModelOptimizer

###########################################################


class ModelTrainer:
    """Trains an sklearn classifier using ModelOptimizer and ModelEvaluator classes
    and evaluates the model and logs its metrics to Comet experiment.

    Attributes:
        train_features (pd.DataFrame): training features,
        train_class (np.ndarray): training class labels,
        valid_features (pd.DataFrame): validation features,
        valid_class (np.ndarray): validation class labels,
        train_features_preprocessed (pd.DataFrame): transformed training set for ModelOptimizer class,
        valid_features_preprocessed (pd.DataFrame): transformed validation set for ModelOptimizer class,
        n_features (int): count of all features after encoding and feature selection,
        class_encoder (LabelEncoder): encoder object that maps the class labels to integers,
        preprocessor_step (ColumnTransformer): preprocessing pipeline step that
            transforms features,
        selector_step (VarianceThreshold): feature selection pipeline step that removes
            low variance features,
        artifacts_path (str): path to save training artificats, e.g., .pkl and .png files,
        num_feature_names (list, optional): list of original numerical feature names,
        cat_feature_names (list, optional): list of original categorical feature names
            (before encoding and feature selection). Default to None,
        fbeta_score_beta (float): beta value (weight of recall) in fbeta_score(),
        encoded_pos_class_label (int): encoded label of positive class using LabelEncoder().
                Default to 1.
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
        artifacts_path: str,
        num_feature_names: Optional[list] = None,
        cat_feature_names: Optional[list] = None,
        fbeta_score_beta: float = 1.0,
        encoded_pos_class_label: int = 1,
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
        self.artifacts_path = artifacts_path
        self.num_feature_names = num_feature_names
        self.cat_feature_names = cat_feature_names
        self.fbeta_score_beta = fbeta_score_beta
        self.encoded_pos_class_label = encoded_pos_class_label

    @staticmethod
    def _create_comet_experiment(
        comet_api_key: str, comet_project_name: str, comet_exp_name: str
    ) -> Experiment:
        """Creates a Comet experiment object.

        Args:
            comet_api_key (str): Comet API key,
            comet_project_name (str): Comet project name,
            comet_exp_name (str)L Comet experiment name,

        Returns:
            Experiment: Comet experiment object.

        Raises:
            ValueError: if Comet experiment creation fails.
        """

        try:
            comet_exp = Experiment(
                api_key=comet_api_key, project_name=comet_project_name
            )
            comet_exp.log_code(folder=".")
            comet_exp.set_name(comet_exp_name)
        except ValueError as e:
            raise ValueError(f"Comet experiment creation error --> {e}") from e
        return comet_exp

    def _optimize_model(
        self,
        comet_exp: Experiment,
        model: Callable,
        max_search_iters: int = 100,
        optimize_in_parallel: bool = False,
        n_parallel_jobs: int = 4,
        model_opt_timeout_secs: int = 600,
        is_voting_ensemble: bool = False,
    ) -> Union[optuna.study.Study, tuple]:
        """Optimizes the model using ModelOptimizer class.

        Args:
            comet_exp (Experiment): Comet experiment object,
            model (Callable): model object that implements the fit and predict methods,
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
            comet_exp=comet_exp,
            train_features_preprocessed=self.train_features_preprocessed,
            train_class=self.train_class,
            valid_features_preprocessed=self.valid_features_preprocessed,
            valid_class=self.valid_class,
            n_features=self.n_features,
            model=model,
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
        comet_exp: Experiment,
        study: optuna.study.Study,
        classifier_name: str,
    ) -> None:
        """Logs Optuna study results to Comet experiment.

        Args:
            comet_exp (Experiment): Comet experiment object,
            study (optuna.study.Study): Optuna study object,
            classifier_name (str): name of the classifier.

        Returns:
            None
        """

        study_results = study.trials_dataframe()
        study_results.rename(
            columns={"value": f"f_{self.fbeta_score_beta}_score"}, inplace=True
        )
        study_results.rename(columns=lambda x: re.sub("params_", "", x), inplace=True)
        study_results.to_csv(
            f"{self.artifacts_path}/study_{classifier_name}.csv", index=False
        )
        comet_exp.log_asset(
            file_data=f"{self.artifacts_path}/study_{classifier_name}.csv",
            file_name=f"study_{classifier_name}",
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
        evaluator: ModelEvaluator,
        fitted_pipeline: Pipeline,
    ) -> Union[pd.DataFrame, pd.DataFrame]:
        """Evaluates the model using ModelEvaluator class.

        Args:
            evaluator (ModelEvaluator): ModelEvaluator object,
            fitted_pipeline (Pipeline): fitted pipeline object.

        Returns:
            train_scores (pd.DataFrame): training scores,
            valid_scores (pd.DataFrame): validation scores.
        """

        evaluator.extract_feature_importance(
            pipeline=fitted_pipeline,
            num_feature_names=self.num_feature_names,
            cat_feature_names=self.cat_feature_names,
            figure_size=(24, 36),
            font_size=10,
        )

        train_scores, valid_scores = evaluator.evaluate_model_perf(
            class_encoder=self.class_encoder
        )

        return train_scores, valid_scores

    def _log_model_metrics(
        self,
        comet_exp: Experiment,
        evaluator: ModelEvaluator,
        fitted_pipeline: Pipeline,
        metrics_to_log: dict,
    ) -> None:
        """Logs model metrics to Comet experiment.

        Args:
            comet_exp (Experiment): Comet experiment object,
            evaluator (ModelEvaluator): ModelEvaluator object,
            fitted_pipeline (Pipeline): fitted pipeline object,
            valid_features (pd.DataFrame): validation features,
            encoded_pos_class_label (int): encoded label of positive class,
            metrics_to_log (dict): dictionary of model metrics.

        Returns:
            None
        """

        pred_probs = fitted_pipeline.predict_proba(self.valid_features)
        pos_class_index = list(fitted_pipeline.classes_).index(
            self.encoded_pos_class_label
        )
        model_ece = evaluator.calc_expected_calibration_error(
            pred_probs=pred_probs[:, pos_class_index],
            true_labels=self.valid_class,
            nbins=5,
        )
        metrics_to_log.update({"model_ece": model_ece})
        comet_exp.log_metrics(metrics_to_log)

    def _register_model(
        self,
        comet_exp: Experiment,
        fitted_pipeline: Pipeline,
        registered_model_name: str,
        metrics_to_log: dict,
    ) -> None:
        """Saves and registers the model to Comet experiment.

        Args:
            comet_exp (Experiment): Comet experiment object,
            fitted_pipeline (Pipeline): fitted pipeline object,
            registered_model_name (str): name of the registered model.
            metrics_to_log (dict): dictionary of model metrics.

        Returns:
            None
        """

        joblib.dump(
            fitted_pipeline, f"{self.artifacts_path}/{registered_model_name}.pkl"
        )
        comet_exp.log_model(
            name=registered_model_name,
            file_or_folder=f"{self.artifacts_path}/{registered_model_name}.pkl",
            overwrite=False,
        )
        comet_exp.register_model(model_name=registered_model_name, tags=metrics_to_log)

    def submit_train_exp(
        self,
        comet_api_key: str,
        comet_project_name: str,
        comet_exp_name: str,
        model: Callable,
        max_search_iters: int = 100,
        optimize_in_parallel: bool = False,
        n_parallel_jobs: int = 4,
        model_opt_timeout_secs: int = 600,
        registered_model_name: Optional[str] = None,
        is_voting_ensemble: bool = False,
    ) -> Union[Pipeline, Experiment]:
        """Submits a training experiment to Comet project. This is a wrapper function
        that uses ModelOptimizer and ModelEvaluator classes to tune and evaluate the model,
        and logs the model and its training metrics to Comet experiment. While the class instance
        is used to store the training data and model transformation pipeline, this function can be
        used to submit the training job for different models that use the same training and validation
        sets and data tranformation pipeline.

        Args:
            comet_api_key (str): Comet API key,
            comet_project_name (str): Comet project name,
            comet_exp_name (str)L Comet experiment name,
            model (Callable): model object that implements the fit and predict methods.
            artifacts_path (str): path to save training artificats, e.g., .pkl and .png files.
            max_search_iters (int, optional): maximum number of iterations for the hyperparameter
                optimization algorithm. Default to 100.
            optimize_in_parallel (bool, optional): should optimization be run in parallel. Default
                to False.
            n_parallel_jobs (int, optional): number of parallel jobs to run during the
                hyperparameters optimization. Default to 4.
            model_opt_timeout_secs (int, optional): timeout in seconds for each trial of the
                hyperparameters optimization. Default to 600.
            registered_model_name (str, optional): name used for the registered model.
            is_voting_ensemble (bool, optional): is it a voting ensemble classifier? This is needed
                for extracting model name in ModelOptimizer class.

        Returns:
            Pipeline: calibrated pipeline object that contains the model transformation pipeline
                and calibrated classifier.
            comet_exp (Experiment): Comet experiment object to be used to access returned model metrics.

        Raises:
            Exception: if model training fails.
        """

        # Extract the classifier name from the model object if not provided
        classifier_name = model.__class__.__name__
        if registered_model_name is None:
            registered_model_name = classifier_name

        # Create Comet experiment
        comet_exp = self._create_comet_experiment(
            comet_api_key=comet_api_key,
            comet_project_name=comet_project_name,
            comet_exp_name=comet_exp_name,
        )

        try:
            # Tune model
            study, optimizer = self._optimize_model(
                comet_exp=comet_exp,
                model=model,
                max_search_iters=max_search_iters,
                optimize_in_parallel=optimize_in_parallel,
                n_parallel_jobs=n_parallel_jobs,
                model_opt_timeout_secs=model_opt_timeout_secs,
                is_voting_ensemble=is_voting_ensemble,
            )

            # Log study trials
            self._log_study_trials(
                comet_exp=comet_exp,
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
            comet_exp.log_parameters(model_params)

            # Evaluate best model
            evaluator = ModelEvaluator(
                comet_exp=comet_exp,
                pipeline=fitted_pipeline,
                train_features=self.train_features,
                train_class=self.train_class,
                valid_features=self.valid_features,
                valid_class=self.valid_class,
                fbeta_score_beta=self.fbeta_score_beta,
                is_voting_ensemble=is_voting_ensemble,
            )

            train_scores, valid_scores = self._evaluate_model(
                evaluator=evaluator,
                fitted_pipeline=fitted_pipeline,
            )

            # Log model metrics
            metrics_to_log = {}
            train_metric_values = evaluator.convert_metrics_from_df_to_dict(
                scores=train_scores, prefix="train_"
            )
            metrics_to_log.update(train_metric_values)

            valid_metric_values = evaluator.convert_metrics_from_df_to_dict(
                scores=valid_scores, prefix="valid_"
            )
            metrics_to_log.update(valid_metric_values)

            self._log_model_metrics(
                comet_exp=comet_exp,
                evaluator=evaluator,
                fitted_pipeline=fitted_pipeline,
                metrics_to_log=metrics_to_log,
            )

            # Save and register model
            self._register_model(
                comet_exp=comet_exp,
                fitted_pipeline=fitted_pipeline,
                registered_model_name=registered_model_name,
                metrics_to_log=metrics_to_log,
            )

        except Exception as e:  # pylint: disable=W0718
            print(f"\n\nModel training error --> {e}\n\n")
            fitted_pipeline = None

        comet_exp.end()

        return fitted_pipeline, comet_exp


class VotingEnsembleCreator:
    """Creates a voting ensemble classifier. It uses ModelEvaluator class to
    evaluate the model and logs the model and its metrics to Comet experiment.

    Attributes:
        comet_api_key (str): Comet API key,
        comet_project_name (str): Comet project name,
        comet_exp_name (str)L Comet experiment name,
        train_features (pd.DataFrame): features of training set before encoding and
        feature selection for ModelEvaluator class.
        valid_features (pd.DataFrame): features of validation set before encoding and
        feature selection for ModelEvaluator class.
        train_class (np.ndarray): The target labels for the training set.
        valid_class (np.ndarray): The target labels for the validation set.
        class_encoder (LabelEncoder): encoder object that maps the class labels to integers.
        artifacts_path (str): path to save training artificats, e.g., .pkl and .png files.
        voting_rule (Literal["hard", "soft"], optional): voting rule for the ensemble classifier.
            Default to "soft".
        encoded_pos_class_label (int, optional): encoded label of positive class using LabelEncoder().
            Default to 1.
        fbeta_score_beta (float, optional): beta value (weight of recall) in fbeta_score().
            Default to 1 (same as F1).
        registered_model_name (str, optional): name used for the registered model.
    """

    def __init__(
        self,
        comet_api_key: str,
        comet_project_name: str,
        comet_exp_name: str,
        train_features: pd.DataFrame,
        valid_features: pd.DataFrame,
        train_class: np.ndarray,
        valid_class: np.ndarray,
        class_encoder: LabelEncoder,
        artifacts_path: str,
        voting_rule: Literal["hard", "soft"] = "soft",
        encoded_pos_class_label: int = 1,
        fbeta_score_beta: float = 1.0,
        registered_model_name: Optional[str] = None,
    ):
        self.comet_api_key = comet_api_key
        self.comet_project_name = comet_project_name
        self.comet_exp_name = comet_exp_name
        self.train_features = train_features
        self.valid_features = valid_features
        self.train_class = train_class
        self.valid_class = valid_class
        self.class_encoder = class_encoder
        self.artifacts_path = artifacts_path
        self.voting_rule = voting_rule
        self.encoded_pos_class_label = encoded_pos_class_label
        self.fbeta_score_beta = fbeta_score_beta
        self.registered_model_name = registered_model_name or "VotingEnsemble"

    @staticmethod
    def _create_comet_experiment(
        comet_api_key: str, comet_project_name: str, comet_exp_name: str
    ) -> Experiment:
        """Creates a Comet experiment object.

        Args:
            comet_api_key (str): Comet API key,
            comet_project_name (str): Comet project name,
            comet_exp_name (str)L Comet experiment name,

        Returns:
            Experiment: Comet experiment object.

        Raises:
            ValueError: if Comet experiment creation fails.
        """

        try:
            comet_exp = Experiment(
                api_key=comet_api_key, project_name=comet_project_name
            )
            comet_exp.log_code(folder=".")
            comet_exp.set_name(comet_exp_name)
        except ValueError as e:
            raise ValueError(f"Comet experiment creation error --> {e}") from e
        return comet_exp

    def _get_base_models(
        self,
        lr_calib_pipeline: Pipeline,
        rf_calib_pipeline: Pipeline,
        lgbm_calib_pipeline: Pipeline,
        xgb_calib_pipeline: Pipeline,
    ) -> list:
        """Creates a list of tuples of base models for the voting ensemble.

        Args:
            lr_calib_pipeline (Pipeline): calibrated pipeline for logistic regression model,
            rf_calib_pipeline (Pipeline): calibrated pipeline for random forest model,
            lgbm_calib_pipeline (Pipeline): calibrated pipeline for LightGBM model,
            xgb_calib_pipeline (Pipeline): calibrated pipeline for XGBoost model,

        Returns:
            base_models (list): list of tuples of base models.

        Raises:
            ValueError: if less than two base models are provided.
        """

        base_models = []
        if lr_calib_pipeline:
            model_lr = lr_calib_pipeline.named_steps["classifier"]
            base_models.append(("LR", model_lr))
        if rf_calib_pipeline:
            model_rf = rf_calib_pipeline.named_steps["classifier"]
            base_models.append(("RF", model_rf))
        if lgbm_calib_pipeline:
            model_lgbm = lgbm_calib_pipeline.named_steps["classifier"]
            base_models.append(("LightGBM", model_lgbm))
        if xgb_calib_pipeline:
            model_xgb = xgb_calib_pipeline.named_steps["classifier"]
            base_models.append(("XGBoost", model_xgb))

        if len(base_models) < 2:
            raise ValueError(
                "At least two base models are needed for a voting ensemble."
            )

        return base_models

    def _copy_data_transform_pipeline(self, base_models: list) -> Pipeline:
        """Copies (deep copy) the data transformation pipeline from the first base
        model. It assumes all base models have the same data transformation pipeline.

        Args:
            base_models (list): list of tuples of base models.

        Returns:
            data_pipeline (Pipeline): data transformation pipeline object.
        """

        data_pipeline = base_models[0][1].steps[:-1]
        data_pipeline = deepcopy(data_pipeline)
        return data_pipeline

    def _create_fitted_ensemble_pipeline(self, base_models: list) -> Pipeline:
        """Creates a voting ensemble pipeline (data transformation pipeline and
        base models) and fits the pipeline to the training set.

        Args:
            base_models (list): list of tuples of base models.

        Returns:
            ve_model (VotingClassifier): voting ensemble classifier object,
            ve_pipeline (Pipeline): voting ensemble pipeline object.
        """

        ve_model = VotingClassifier(estimators=base_models, voting=self.voting_rule)
        ve_pipeline = None
        if len(base_models) > 1:
            ve_pipeline = self._copy_data_transform_pipeline(base_models)
            ve_pipeline.steps.append(["classifier", ve_model])
            ve_pipeline.fit(self.train_features, self.train_class)
        return ve_pipeline

    def _evaluate_model(
        self,
        comet_exp: Experiment,
        pipeline: Pipeline,
    ) -> ModelEvaluator:
        """Evaluates the model using ModelEvaluator class.

        Args:
            comet_exp (Experiment): Comet experiment object,
            pipeline (Pipeline): voting ensemble pipeline object.

        Returns:
            evaluator (ModelEvaluator): ModelEvaluator object.
        """

        evaluator = ModelEvaluator(
            comet_exp=comet_exp,
            pipeline=pipeline,
            train_features=self.train_features,
            train_class=self.train_class,
            valid_features=self.valid_features,
            valid_class=self.valid_class,
            fbeta_score_beta=self.fbeta_score_beta,
            is_voting_ensemble=True,
        )
        evaluator.evaluate_model_perf(class_encoder=self.class_encoder)
        return evaluator

    def _log_model_metrics(
        self,
        comet_exp: Experiment,
        evaluator: ModelEvaluator,
        pipeline: Pipeline,
    ) -> dict:
        """Logs model metrics to Comet experiment.

        Args:
            comet_exp (Experiment): Comet experiment object,
            evaluator (ModelEvaluator): ModelEvaluator object,
            pipeline (Pipeline): voting ensemble pipeline object.

        Returns:
            metrics_to_log (dict): dictionary of model metrics.
        """

        metrics_to_log = {}
        train_scores, valid_scores = evaluator.get_model_scores()
        train_metric_values = evaluator.convert_metrics_from_df_to_dict(
            scores=train_scores, prefix="train_"
        )
        metrics_to_log.update(train_metric_values)
        valid_metric_values = evaluator.convert_metrics_from_df_to_dict(
            scores=valid_scores, prefix="valid_"
        )
        metrics_to_log.update(valid_metric_values)
        pred_probs = pipeline.predict_proba(self.valid_features)
        pos_class_index = list(pipeline.classes_).index(self.encoded_pos_class_label)
        model_ece = evaluator.calc_expected_calibration_error(
            pred_probs=pred_probs[:, pos_class_index],
            true_labels=self.valid_class,
            nbins=5,
        )
        metrics_to_log.update({"model_ece": model_ece})
        comet_exp.log_metrics(metrics_to_log)
        return metrics_to_log

    def _register_model(
        self, comet_exp: Experiment, pipeline: Pipeline, metrics_to_log: dict
    ) -> None:
        """Saves and registers the model to Comet experiment.

        Args:
            comet_exp (Experiment): Comet experiment object,
            pipeline (Pipeline): voting ensemble pipeline object,
            metrics_to_log (dict): dictionary of model metrics.

        Returns:
            None
        """

        joblib.dump(
            pipeline,
            f"{self.artifacts_path}/{self.registered_model_name}.pkl",
        )
        comet_exp.log_model(
            name=self.registered_model_name,
            file_or_folder=f"{self.artifacts_path}/{self.registered_model_name}.pkl",
            overwrite=False,
        )

        comet_exp.register_model(
            model_name=self.registered_model_name,
            tags=metrics_to_log,
        )

    def create_voting_ensemble(
        self,
        lr_calib_pipeline: Pipeline = None,
        rf_calib_pipeline: Pipeline = None,
        lgbm_calib_pipeline: Pipeline = None,
        xgb_calib_pipeline: Pipeline = None,
    ) -> Pipeline:
        """Creates a voting ensemble classifier using the base models and evaluates the model
        using ModelEvaluator class. It logs the model metrics to Comet experiment.

        Args:
            lr_calib_pipeline (Pipeline): calibrated pipeline for logistic regression model,
            rf_calib_pipeline (Pipeline): calibrated pipeline for random forest model,
            lgbm_calib_pipeline (Pipeline): calibrated pipeline for LightGBM model,
            xgb_calib_pipeline (Pipeline): calibrated pipeline for XGBoost model,

        Returns:
            Pipeline: calibrated pipeline object that contains the model transformation pipeline
                and calibrated classifier.

        Raises:
            Exception: if model training fails.
        """

        # Create Comet experiment
        comet_exp = self._create_comet_experiment(
            comet_api_key=self.comet_api_key,
            comet_project_name=self.comet_project_name,
            comet_exp_name=self.comet_exp_name,
        )

        try:
            # Create voting ensemble pipeline (data transformation pipeline and base models)
            base_models = self._get_base_models(
                lr_calib_pipeline=lr_calib_pipeline,
                rf_calib_pipeline=rf_calib_pipeline,
                lgbm_calib_pipeline=lgbm_calib_pipeline,
                xgb_calib_pipeline=xgb_calib_pipeline,
            )
            ve_pipeline = self._create_fitted_ensemble_pipeline(base_models)
            evaluator = self._evaluate_model(
                comet_exp=comet_exp,
                pipeline=ve_pipeline,
            )
            metrics_to_log = self._log_model_metrics(
                comet_exp=comet_exp,
                evaluator=evaluator,
                pipeline=ve_pipeline,
            )
            self._register_model(
                comet_exp=comet_exp, pipeline=ve_pipeline, metrics_to_log=metrics_to_log
            )

        except Exception as e:  # pylint: disable=W0718
            print(f"\nVoting ensemble error --> {e}\n\n")
            ve_pipeline = None

        comet_exp.end()

        return ve_pipeline, comet_exp
