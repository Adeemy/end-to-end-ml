"""
Model training orchestration - coordinates the full training workflow.
"""

from pathlib import PosixPath
from typing import Any, Callable, Optional

import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.training.utils.config.config import SupportedModelsConfig
from src.training.utils.core.optimizer import ModelOptimizer
from src.training.utils.evaluation.evaluator import create_model_evaluator
from src.training.utils.tracking.experiment import ExperimentManager
from src.training.utils.tracking.experiment_tracker import ExperimentTracker
from src.training.utils.tracking.study_logger import StudyLogger
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
        experiment_manager: ExperimentManager,
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
            experiment_manager: Experiment manager instance.
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
        self.experiment_manager = experiment_manager
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
        tracker: ExperimentTracker,
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
            tracker: Experiment tracker object.
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
        tracker: ExperimentTracker,
        fitted_pipeline: Pipeline,
        is_voting_ensemble: bool = False,
        ece_nbins: int = 5,
    ) -> tuple[dict, dict, Optional[float]]:
        """Evaluates the fitted model for both binary and multi-class models.

        Args:
            tracker: Experiment tracker object.
            fitted_pipeline: Fitted pipeline.
            is_voting_ensemble: Whether this is a voting ensemble.
            ece_nbins: Number of bins for ECE calculation.

        Returns:
            Tuple of (train_metrics, valid_metrics, model_ece).

        Note: model_ece is only calculated for binary classification models,
            returns None for multi-class models.
        """
        evaluator = create_model_evaluator(
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

        # Determine if this is a binary or multi-class model
        num_classes = len(fitted_pipeline.classes_)
        is_binary_classification = num_classes == 2

        # Calculate ECE only for binary classification models
        model_ece = None
        if is_binary_classification:
            pred_probs = fitted_pipeline.predict_proba(self.valid_features)
            pos_class_index = list(fitted_pipeline.classes_).index(
                self.encoded_pos_class_label
            )

            # Probability array handlers registry for binary classification
            prob_handlers = {
                1: lambda probs, pos_idx: probs,  # 1D array case (binary edge case)
                2: lambda probs, pos_idx: probs[
                    :, pos_idx
                ],  # 2D array case (normal binary)
            }

            # Handle both 1D and 2D probability arrays using registry
            handler = prob_handlers.get(pred_probs.ndim)
            if handler is None:
                available_dims = ", ".join(map(str, prob_handlers.keys()))
                raise ValueError(
                    f"Unexpected predict_proba shape for binary classification: {pred_probs.shape}. "
                    f"Supported dimensions: {available_dims}"
                )

            pos_probs = handler(pred_probs, pos_class_index)

            model_ece = evaluator.calc_expected_calibration_error(
                pred_probs=pos_probs,
                true_labels=self.valid_class,
                nbins=ece_nbins,
            )

            logger.info("Calculated ECE for binary model %.4f", model_ece)
        else:
            logger.info(
                "Skipping ECE calculation for multi-class model with %d classes",
                num_classes,
            )

        return train_metric_values, valid_metric_values, model_ece

    def run_training_experiment(
        self,
        project_name: str,
        experiment_name: str,
        model: Callable,
        search_space_params: dict,
        registered_model_name: str,
        api_key: Optional[str] = None,
        max_search_iters: int = 100,
        optimize_in_parallel: bool = False,
        n_parallel_jobs: int = 4,
        model_opt_timeout_secs: int = 600,
        is_voting_ensemble: bool = False,
        ece_nbins: int = 5,
    ) -> tuple[Optional[Pipeline], Any]:
        """Runs the complete training experiment workflow.

        Args:
            project_name: Project name.
            experiment_name: Experiment name.
            model: Model to train.
            search_space_params: Hyperparameter search space.
            registered_model_name: Model registry name.
            api_key: Optional API key.
            max_search_iters: Maximum optimization iterations. Defaults to 100.
            optimize_in_parallel: Whether to optimize in parallel. Defaults to False.
            n_parallel_jobs: Number of parallel jobs. Defaults to 4.
            model_opt_timeout_secs: Optimization timeout. Defaults to 600.
            is_voting_ensemble: Whether this is a voting ensemble. Defaults to False.
            ece_nbins: Number of bins for ECE. Defaults to 5.

        Returns:
            Tuple of (fitted_pipeline, experiment).
        """
        classifier_name = model.__class__.__name__

        # Create experiment
        experiment = self.experiment_manager.create_experiment(
            api_key=api_key,
            project_name=project_name,
            experiment_name=experiment_name,
        )

        try:
            # Get tracker
            tracker = self.experiment_manager.get_tracker(experiment)

            # Optimize model
            study, optimizer = self.optimize_model(
                tracker=tracker,
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
                tracker=tracker,
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
            tracker.log_parameters(model_params)

            # Evaluate model
            train_metrics, valid_metrics, model_ece = self.evaluate_model(
                tracker=tracker,
                fitted_pipeline=fitted_pipeline,
                is_voting_ensemble=is_voting_ensemble,
                ece_nbins=ece_nbins,
            )

            # Log metrics
            metrics_to_log = {**train_metrics, **valid_metrics, "model_ece": model_ece}
            tracker.log_metrics(metrics_to_log)

            # Register model
            self.experiment_manager.register_model(
                experiment=experiment,
                pipeline=fitted_pipeline,
                registered_model_name=registered_model_name,
                artifacts_path=self.artifacts_path,
            )

        except Exception as e:  # pylint: disable=W0718
            logger.error("Model training error --> %s", e)
            fitted_pipeline = None

        self.experiment_manager.end_experiment(experiment)

        return fitted_pipeline, experiment
