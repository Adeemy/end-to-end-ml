"""
Voting ensemble creation - orchestrates building and evaluating ensemble models.
"""

from copy import deepcopy
from pathlib import PosixPath
from typing import Literal, Optional

import numpy as np
import pandas as pd
from comet_ml import Experiment
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.training.utils.config import SupportedModelsConfig
from src.training.utils.evaluator import ModelEvaluator
from src.training.utils.experiment import CometExperimentManager
from src.training.utils.experiment_tracker import CometExperimentTracker
from src.utils.logger import get_console_logger

module_name: str = PosixPath(__file__).stem
logger = get_console_logger(module_name)


class EnsembleOrchestrator:
    """Orchestrates voting ensemble creation and evaluation.

    Single Responsibility: Build and evaluate voting ensemble models.
    Open/Closed Principle: Extensible for different ensemble types.
    Dependency Inversion: Depends on abstractions (ExperimentManager, ModelEvaluator).
    """

    def __init__(
        self,
        train_features: pd.DataFrame,
        valid_features: pd.DataFrame,
        train_class: np.ndarray,
        valid_class: np.ndarray,
        class_encoder: LabelEncoder,
        artifacts_path: str,
        supported_models: SupportedModelsConfig,
        lr_calib_pipeline: Optional[Pipeline] = None,
        rf_calib_pipeline: Optional[Pipeline] = None,
        lgbm_calib_pipeline: Optional[Pipeline] = None,
        xgb_calib_pipeline: Optional[Pipeline] = None,
        voting_rule: Literal["hard", "soft"] = "soft",
        encoded_pos_class_label: int = 1,
        fbeta_score_beta: float = 1.0,
    ):
        """Initializes the EnsembleOrchestrator.

        Args:
            train_features: Training features.
            valid_features: Validation features.
            train_class: Training class labels.
            valid_class: Validation class labels.
            class_encoder: Label encoder for class labels.
            artifacts_path: Path to save artifacts.
            supported_models: Supported models configuration.
            lr_calib_pipeline: Calibrated logistic regression pipeline.
            rf_calib_pipeline: Calibrated random forest pipeline.
            lgbm_calib_pipeline: Calibrated LightGBM pipeline.
            xgb_calib_pipeline: Calibrated XGBoost pipeline.
            voting_rule: Voting rule for ensemble ("hard" or "soft").
            encoded_pos_class_label: Encoded positive class label.
            fbeta_score_beta: Beta value for fbeta score.
        """
        self.train_features = train_features
        self.valid_features = valid_features
        self.train_class = train_class
        self.valid_class = valid_class
        self.class_encoder = class_encoder
        self.artifacts_path = artifacts_path
        self.supported_models = supported_models
        self.lr_calib_pipeline = lr_calib_pipeline
        self.rf_calib_pipeline = rf_calib_pipeline
        self.lgbm_calib_pipeline = lgbm_calib_pipeline
        self.xgb_calib_pipeline = xgb_calib_pipeline
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

        try:
            model_lr = self.lr_calib_pipeline.named_steps["classifier"]
            base_models.append(("LR", model_lr))
        except Exception:  # pylint: disable=W0718
            logger.info("LR model does not exist or not in required type!")

        try:
            model_rf = self.rf_calib_pipeline.named_steps["classifier"]
            base_models.append(("RF", model_rf))
        except Exception:  # pylint: disable=W0718
            logger.info("RF model does not exist or not in required type!")

        try:
            model_lgbm = self.lgbm_calib_pipeline.named_steps["classifier"]
            base_models.append(("LightGBM", model_lgbm))
        except Exception:  # pylint: disable=W0718
            logger.info("LightGBM model does not exist or not in required type!")

        try:
            model_xgb = self.xgb_calib_pipeline.named_steps["classifier"]
            base_models.append(("XGBoost", model_xgb))
        except Exception:  # pylint: disable=W0718
            logger.info("XGBoost model does not exist or not in required type!")

        if len(base_models) < 2:
            raise ValueError(
                "At least two base models are needed for a voting ensemble."
            )

        return base_models

    def copy_data_transform_pipeline(self) -> Pipeline:
        """Copies the data transformation pipeline from the first available base model.

        Assumes all base models have the same data transformation pipeline.

        Returns:
            Data transformation pipeline object.

        Raises:
            ValueError: If no base model pipelines are found.
        """
        if hasattr(self, "lr_calib_pipeline") and self.lr_calib_pipeline is not None:
            data_pipeline = deepcopy(self.lr_calib_pipeline)
        elif hasattr(self, "rf_calib_pipeline") and self.rf_calib_pipeline is not None:
            data_pipeline = deepcopy(self.rf_calib_pipeline)
        elif (
            hasattr(self, "lgbm_calib_pipeline")
            and self.lgbm_calib_pipeline is not None
        ):
            data_pipeline = deepcopy(self.lgbm_calib_pipeline)
        elif (
            hasattr(self, "xgb_calib_pipeline") and self.xgb_calib_pipeline is not None
        ):
            data_pipeline = deepcopy(self.xgb_calib_pipeline)
        else:
            raise ValueError("No base model pipelines found!")

        return data_pipeline

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
        experiment: Experiment,
        fitted_pipeline: Pipeline,
        ece_nbins: int = 5,
    ) -> tuple[dict, dict, float]:
        """Evaluates the voting ensemble model.

        Args:
            experiment: Comet experiment object.
            fitted_pipeline: Fitted ensemble pipeline.
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

    def create_voting_ensemble(
        self,
        comet_api_key: str,
        project_name: str,
        experiment_name: str,
        registered_model_name: str = "VotingEnsemble",
        ece_nbins: int = 5,
    ) -> tuple[Optional[Pipeline], Experiment]:
        """Creates and evaluates a voting ensemble classifier.

        Args:
            comet_api_key: Comet API key.
            project_name: Comet project name.
            experiment_name: Experiment name.
            registered_model_name: Model registry name.
            ece_nbins: Number of bins for ECE.

        Returns:
            Tuple of (ensemble_pipeline, experiment).
        """
        # Create experiment manager
        experiment_manager = CometExperimentManager()

        # Create experiment
        experiment = experiment_manager.create_experiment(
            comet_api_key=comet_api_key,
            project_name=project_name,
            experiment_name=experiment_name,
        )

        try:
            # Create voting ensemble pipeline
            base_models = self.get_base_models()
            ve_pipeline = self.create_fitted_ensemble_pipeline(base_models)

            # Evaluate voting ensemble classifier
            train_metrics, valid_metrics, model_ece = self.evaluate_ensemble(
                experiment=experiment,
                fitted_pipeline=ve_pipeline,
                ece_nbins=ece_nbins,
            )

            # Log metrics
            metrics_to_log = {**train_metrics, **valid_metrics, "model_ece": model_ece}
            experiment_manager.log_metrics(experiment, metrics_to_log)

            # Register model
            experiment_manager.register_model(
                experiment=experiment,
                pipeline=ve_pipeline,
                registered_model_name=registered_model_name,
                artifacts_path=self.artifacts_path,
            )

        except Exception as e:  # pylint: disable=W0718
            logger.error("Voting ensemble error --> %s", e)
            ve_pipeline = None

        experiment_manager.end_experiment(experiment)

        return ve_pipeline, experiment
