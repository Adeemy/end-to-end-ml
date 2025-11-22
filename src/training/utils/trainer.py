"""
Model training orchestration - coordinates the full training workflow.
"""

from pathlib import PosixPath
from typing import Callable, Optional

import numpy as np
import optuna
import pandas as pd
from comet_ml import Experiment
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.training.utils.config import SupportedModelsConfig
from src.training.utils.evaluator import ModelEvaluator
from src.training.utils.experiment import CometExperimentManager
from src.training.utils.experiment_tracker import CometExperimentTracker
from src.training.utils.optimizer import ModelOptimizer
from src.training.utils.study_logger import StudyLogger
from src.utils.logger import get_console_logger

module_name: str = PosixPath(__file__).stem
logger = get_console_logger(module_name)


class TrainingOrchestrator:
    """Orchestrates the full model training workflow.

    Single Responsibility: Coordinate training, evaluation, and model registration.
    Open/Closed Principle: Extensible without modification.
    Dependency Inversion: Depends on abstractions (ExperimentManager, ModelOptimizer, etc.)
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
        supported_models: SupportedModelsConfig,
        num_feature_names: Optional[list] = None,
        cat_feature_names: Optional[list] = None,
        fbeta_score_beta: float = 1.0,
        encoded_pos_class_label: int = 1,
    ):
        """Initializes the TrainingOrchestrator.

        Args:
            train_features: Training features.
            train_class: Training class labels.
            valid_features: Validation features.
            valid_class: Validation class labels.
            train_features_preprocessed: Preprocessed training features.
            valid_features_preprocessed: Preprocessed validation features.
            n_features: Number of features after preprocessing.
            class_encoder: Label encoder for class labels.
            preprocessor_step: Preprocessing pipeline step.
            selector_step: Feature selection pipeline step.
            artifacts_path: Path to save artifacts.
            supported_models: Supported models configuration.
            num_feature_names: List of numerical feature names.
            cat_feature_names: List of categorical feature names.
            fbeta_score_beta: Beta value for fbeta score.
            encoded_pos_class_label: Encoded positive class label.
        """
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
        self.supported_models = supported_models
        self.num_feature_names = num_feature_names
        self.cat_feature_names = cat_feature_names
        self.fbeta_score_beta = fbeta_score_beta
        self.encoded_pos_class_label = encoded_pos_class_label

    def optimize_model(
        self,
        experiment: Experiment,
        model: Callable,
        search_space_params: dict,
        registered_model_name: str,
        max_search_iters: int = 100,
        optimize_in_parallel: bool = False,
        n_parallel_jobs: int = 4,
        model_opt_timeout_secs: int = 600,
        is_voting_ensemble: bool = False,
    ) -> tuple[optuna.study.Study, ModelOptimizer]:
        """Optimizes model hyperparameters.

        Args:
            experiment: Comet experiment object.
            model: Model object to optimize.
            search_space_params: Hyperparameter search space.
            registered_model_name: Registry name for this model.
            max_search_iters: Maximum optimization iterations.
            optimize_in_parallel: Whether to optimize in parallel.
            n_parallel_jobs: Number of parallel jobs.
            model_opt_timeout_secs: Optimization timeout in seconds.
            is_voting_ensemble: Whether this is a voting ensemble.

        Returns:
            Tuple of (study, optimizer).
        """
        tracker = CometExperimentTracker(experiment=experiment)

        optimizer = ModelOptimizer(
            tracker=tracker,
            train_features_preprocessed=self.train_features_preprocessed,
            train_class=self.train_class,
            valid_features_preprocessed=self.valid_features_preprocessed,
            valid_class=self.valid_class,
            n_features=self.n_features,
            model=model,
            search_space_params=search_space_params,
            supported_models=self.supported_models,
            registered_model_name=registered_model_name,
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

    def fit_best_model(
        self,
        study: optuna.study.Study,
        optimizer: ModelOptimizer,
        model: Callable,
    ) -> Pipeline:
        """Fits the best model from optimization study.

        Args:
            study: Optuna study object.
            optimizer: ModelOptimizer instance.
            model: Model object.

        Returns:
            Fitted pipeline.
        """
        model = model.set_params(**study.best_params)
        fitted_pipeline = optimizer.fit_pipeline(
            train_features=self.train_features,
            preprocessor_step=self.preprocessor_step,
            selector_step=self.selector_step,
            model=model,
        )
        return fitted_pipeline

    def evaluate_model(
        self,
        experiment: Experiment,
        fitted_pipeline: Pipeline,
        is_voting_ensemble: bool = False,
        ece_nbins: int = 5,
    ) -> tuple[dict, dict, float]:
        """Evaluates the fitted model.

        Args:
            experiment: Comet experiment object.
            fitted_pipeline: Fitted pipeline.
            is_voting_ensemble: Whether this is a voting ensemble.
            ece_nbins: Number of bins for ECE calculation.

        Returns:
            Tuple of (train_metrics, valid_metrics, model_ece).
        """
        tracker = CometExperimentTracker(experiment=experiment)

        evaluator = ModelEvaluator(
            tracker=tracker,
            pipeline=fitted_pipeline,
            train_features=self.train_features,
            train_class=self.train_class,
            valid_features=self.valid_features,
            valid_class=self.valid_class,
            fbeta_score_beta=self.fbeta_score_beta,
            is_voting_ensemble=is_voting_ensemble,
        )

        train_scores, valid_scores = evaluator.evaluate_model_perf(
            class_encoder=self.class_encoder
        )

        evaluator.extract_feature_importance(
            pipeline=fitted_pipeline,
            num_feature_names=self.num_feature_names,
            cat_feature_names=self.cat_feature_names,
            figure_size=(24, 36),
            font_size=10,
        )

        train_metric_values = evaluator.convert_metrics_from_df_to_dict(
            scores=train_scores, prefix="train_"
        )
        valid_metric_values = evaluator.convert_metrics_from_df_to_dict(
            scores=valid_scores, prefix="valid_"
        )

        # Calculate ECE
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

    def run_training_experiment(
        self,
        comet_api_key: str,
        project_name: str,
        experiment_name: str,
        model: Callable,
        search_space_params: dict,
        registered_model_name: str,
        max_search_iters: int = 100,
        optimize_in_parallel: bool = False,
        n_parallel_jobs: int = 4,
        model_opt_timeout_secs: int = 600,
        is_voting_ensemble: bool = False,
        ece_nbins: int = 5,
    ) -> tuple[Optional[Pipeline], Experiment]:
        """Runs the complete training experiment workflow.

        Args:
            comet_api_key: Comet API key.
            project_name: Comet project name.
            experiment_name: Experiment name.
            model: Model to train.
            search_space_params: Hyperparameter search space.
            registered_model_name: Model registry name.
            max_search_iters: Maximum optimization iterations.
            optimize_in_parallel: Whether to optimize in parallel.
            n_parallel_jobs: Number of parallel jobs.
            model_opt_timeout_secs: Optimization timeout.
            is_voting_ensemble: Whether this is a voting ensemble.
            ece_nbins: Number of bins for ECE.

        Returns:
            Tuple of (fitted_pipeline, experiment).
        """
        classifier_name = model.__class__.__name__

        # Create experiment manager
        experiment_manager = CometExperimentManager()

        # Create experiment
        experiment = experiment_manager.create_experiment(
            comet_api_key=comet_api_key,
            project_name=project_name,
            experiment_name=experiment_name,
        )

        try:
            # Optimize model
            study, optimizer = self.optimize_model(
                experiment=experiment,
                model=model,
                search_space_params=search_space_params,
                registered_model_name=registered_model_name,
                max_search_iters=max_search_iters,
                optimize_in_parallel=optimize_in_parallel,
                n_parallel_jobs=n_parallel_jobs,
                model_opt_timeout_secs=model_opt_timeout_secs,
                is_voting_ensemble=is_voting_ensemble,
            )

            # Log study trials
            StudyLogger.log_study_trials(
                experiment=experiment,
                study=study,
                classifier_name=classifier_name,
                artifacts_path=self.artifacts_path,
                fbeta_score_beta=self.fbeta_score_beta,
            )

            # Fit best model
            fitted_pipeline = self.fit_best_model(
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
            experiment_manager.log_parameters(experiment, model_params)

            # Evaluate model
            train_metrics, valid_metrics, model_ece = self.evaluate_model(
                experiment=experiment,
                fitted_pipeline=fitted_pipeline,
                is_voting_ensemble=is_voting_ensemble,
                ece_nbins=ece_nbins,
            )

            # Log metrics
            metrics_to_log = {**train_metrics, **valid_metrics, "model_ece": model_ece}
            experiment_manager.log_metrics(experiment, metrics_to_log)

            # Register model
            experiment_manager.register_model(
                experiment=experiment,
                pipeline=fitted_pipeline,
                registered_model_name=registered_model_name,
                artifacts_path=self.artifacts_path,
            )

        except Exception as e:  # pylint: disable=W0718
            logger.error("Model training error --> %s", e)
            fitted_pipeline = None

        experiment_manager.end_experiment(experiment)

        return fitted_pipeline, experiment
