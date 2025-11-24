"""
Orchestrates voting classifier ensemble creation and evaluation.
"""

from copy import deepcopy
from pathlib import PosixPath
from typing import Any, List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.training.evaluation.evaluator import create_model_evaluator
from src.training.schemas import SupportedModelsConfig
from src.training.tracking.experiment import ExperimentManager
from src.training.tracking.experiment_tracker import ExperimentTracker
from src.utils.logger import get_console_logger

module_name: str = PosixPath(__file__).stem
logger = get_console_logger(module_name)


class ClassifierEnsembleOrchestrator:
    """Orchestrates voting ensemble creation and evaluation.

    Single Responsibility: Build and evaluate voting ensemble models.
    Open/Closed Principle: Extensible for different ensemble types.
    Dependency Inversion: Depends on abstractions (ExperimentManager, ModelEvaluator).
    """

    def __init__(
        self,
        experiment_manager: ExperimentManager,
        train_features: pd.DataFrame,
        valid_features: pd.DataFrame,
        train_class: np.ndarray,
        valid_class: np.ndarray,
        class_encoder: LabelEncoder,
        artifacts_path: str,
        supported_models: SupportedModelsConfig,
        base_pipelines: List[Pipeline],
        voting_rule: Literal["hard", "soft"] = "soft",
        encoded_pos_class_label: int = 1,
        fbeta_score_beta: float = 1.0,
    ):
        """Initializes the ClassifierEnsembleOrchestrator.

        Args:
            experiment_manager: Experiment manager instance.
            train_features: Training features.
            valid_features: Validation features.
            train_class: Training class labels.
            valid_class: Validation class labels.
            class_encoder: Label encoder for class labels.
            artifacts_path: Path to save artifacts.
            supported_models: Supported models configuration.
            base_pipelines: List of calibrated pipelines to include in the ensemble.
            voting_rule: Voting rule for ensemble ("hard" or "soft").
            encoded_pos_class_label: Encoded positive class label.
            fbeta_score_beta: Beta value for fbeta score.
        """
        self.experiment_manager = experiment_manager
        self.train_features = train_features
        self.valid_features = valid_features
        self.train_class = train_class
        self.valid_class = valid_class
        self.class_encoder = class_encoder
        self.artifacts_path = artifacts_path
        self.supported_models = supported_models
        self.base_pipelines = [p for p in base_pipelines if p is not None]
        self.voting_rule = voting_rule
        self.encoded_pos_class_label = encoded_pos_class_label
        self.fbeta_score_beta = fbeta_score_beta

    def get_base_models(self) -> list:
        """Creates a list of base models for the voting ensemble.

        Returns:
            List of (name, model) tuples.

        Raises:
            ValueError: If less than two base models are provided.
        """
        base_models = []

        # Extract classifiers from pipelines into (name, model) tuples
        for pipeline in self.base_pipelines:
            try:
                if "classifier" in pipeline.named_steps:
                    name, model = (
                        model.__class__.__name__,
                        pipeline.named_steps["classifier"],
                    )
                    base_models.append((name, model))
                else:
                    logger.warning(
                        "Pipeline provided to ensemble does not contain a 'classifier' step."
                    )
            except Exception as e:  # pylint: disable=W0718
                logger.warning("Failed to extract classifier from pipeline: %s", e)

        if len(base_models) < 2:
            raise ValueError(
                "At least two base models are needed for a voting ensemble."
            )

        return base_models

    def copy_data_transform_pipeline(self) -> Pipeline:
        """Copies the data transformation pipeline from the first available base model.
        The deepcopy is used to avoid modifying the original pipeline.

        Note:
            This method assumes all base models have the same data transformation pipeline.

        Returns:
            Data transformation pipeline object.

        Raises:
            ValueError: If no base model pipelines are found.
        """
        if not self.base_pipelines:
            raise ValueError("No base model pipelines found!")

        return deepcopy(self.base_pipelines[0])

    def create_fitted_ensemble_pipeline(self, base_models: list) -> Pipeline:
        """Creates and fits a voting ensemble pipeline.

        Args:
            base_models: List of (name, model) tuples.

        Returns:
            Fitted voting ensemble pipeline.

        Raises:
            ValueError: If less than two base models are provided.
        """
        if len(base_models) < 2:
            raise ValueError(
                "At least two base models are needed for a voting ensemble."
            )

        ve_model = VotingClassifier(estimators=base_models, voting=self.voting_rule)

        # Copy data transformation pipeline from a base model
        ve_pipeline = self.copy_data_transform_pipeline()

        # Drop base classifier and add voting ensemble with fitted base classifiers
        _ = ve_pipeline.steps.pop(len(ve_pipeline) - 1)
        ve_pipeline.steps.insert(
            len(ve_pipeline) + 1,
            ["classifier", ve_model],
        )
        ve_pipeline.fit(self.train_features, self.train_class)

        return ve_pipeline

    def evaluate_ensemble(
        self,
        tracker: ExperimentTracker,
        fitted_pipeline: Pipeline,
        ece_nbins: int = 5,
    ) -> tuple[dict, dict, Optional[float]]:
        """Evaluates the voting ensemble model for both binary and multi-class models.

        Args:
            tracker: Experiment tracker object.
            fitted_pipeline: Fitted ensemble pipeline.
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
            is_voting_ensemble=True,
        )

        train_scores, valid_scores = evaluator.evaluate_model_perf(
            class_encoder=self.class_encoder
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

            logger.info("Calculated ECE for binary ensemble %s: {}", f"{model_ece:.4f}")
        else:
            logger.info(
                "Skipping ECE calculation for multi-class ensemble with %d classes",
                num_classes,
            )

        return train_metric_values, valid_metric_values, model_ece

    def create_voting_ensemble(
        self,
        project_name: str,
        experiment_name: str,
        api_key: Optional[str] = None,
        registered_model_name: str = "VotingEnsemble",
        ece_nbins: int = 5,
    ) -> tuple[Optional[Pipeline], Any]:
        """Creates and evaluates a voting ensemble classifier.

        Args:
            project_name: Project name.
            experiment_name: Experiment name.
            api_key: Optional API key.
            registered_model_name: Model registry name. Defaults to "VotingEnsemble".
            ece_nbins: Number of bins for ECE. Defaults to 5.

        Returns:
            Tuple of (ensemble_pipeline, experiment).
        """
        # Create experiment
        experiment = self.experiment_manager.create_experiment(
            api_key=api_key,
            project_name=project_name,
            experiment_name=experiment_name,
        )

        try:
            # Create voting ensemble pipeline
            base_models = self.get_base_models()
            ve_pipeline = self.create_fitted_ensemble_pipeline(base_models)

            # Get tracker
            tracker = self.experiment_manager.get_tracker(experiment)

            # Evaluate voting ensemble classifier
            train_metrics, valid_metrics, model_ece = self.evaluate_ensemble(
                tracker=tracker,
                fitted_pipeline=ve_pipeline,
                ece_nbins=ece_nbins,
            )

            # Log metrics
            metrics_to_log = {**train_metrics, **valid_metrics, "model_ece": model_ece}
            tracker.log_metrics(metrics_to_log)

            # Register model
            self.experiment_manager.register_model(
                experiment=experiment,
                pipeline=ve_pipeline,
                registered_model_name=registered_model_name,
                artifacts_path=self.artifacts_path,
            )

        except Exception as e:  # pylint: disable=W0718
            logger.error("Voting ensemble error --> %s", e)
            ve_pipeline = None

        self.experiment_manager.end_experiment(experiment)

        return ve_pipeline, experiment
