"""
Test set evaluation orchestration - evaluates models on held-out test data.
"""

from pathlib import PosixPath
from typing import TYPE_CHECKING, Optional

import joblib
import numpy as np
import pandas as pd
from comet_ml import ExistingExperiment

from src.training.utils.evaluation.champion import ModelChampionManager
from src.training.utils.evaluation.evaluator import create_model_evaluator
from src.training.utils.evaluation.selector import ModelSelector
from src.training.utils.tracking.experiment_tracker import CometExperimentTracker
from src.utils.logger import get_console_logger

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

module_name: str = PosixPath(__file__).stem
logger = get_console_logger(module_name)


class TestSetEvaluationOrchestrator:
    """Orchestrates model evaluation on test set and champion model registration.

    Single Responsibility: Coordinate test evaluation and champion selection.
    Dependency Inversion: Depends on abstractions (ModelSelector, ModelEvaluator).
    """

    def __init__(
        self,
        train_features: pd.DataFrame,
        train_class: np.ndarray,
        test_features: pd.DataFrame,
        test_class: np.ndarray,
        artifacts_path: str,
        fbeta_score_beta: float = 1.0,
        voting_ensemble_name: Optional[str] = None,
    ):
        """Initializes the TestSetEvaluationOrchestrator.

        Args:
            train_features: Training features.
            train_class: Training class labels.
            test_features: Test features.
            test_class: Test class labels.
            artifacts_path: Path to model artifacts.
            fbeta_score_beta: Beta value for fbeta score.
            voting_ensemble_name: Name of voting ensemble model (if exists).
        """
        self.train_features = train_features
        self.train_class = train_class
        self.test_features = test_features
        self.test_class = test_class
        self.artifacts_path = artifacts_path
        self.fbeta_score_beta = fbeta_score_beta
        self.voting_ensemble_name = voting_ensemble_name

    def evaluate_on_test_set(
        self,
        experiment: ExistingExperiment,
        model_pipeline: "Pipeline",
        model_name: str,
    ) -> dict:
        """Evaluates model on test set.

        Args:
            experiment: Comet ExistingExperiment object.
            model_pipeline: Fitted model pipeline (Pipeline).
            model_name: Name of the model being evaluated.

        Returns:
            Dictionary of test metrics.
        """
        tracker = CometExperimentTracker(experiment=experiment)

        is_voting_ensemble = (
            model_name == self.voting_ensemble_name
            if self.voting_ensemble_name
            else False
        )

        evaluator = create_model_evaluator(
            tracker=tracker,
            pipeline=model_pipeline,
            train_features=self.train_features,
            train_class=self.train_class,
            valid_features=self.test_features,
            valid_class=self.test_class,
            fbeta_score_beta=self.fbeta_score_beta,
            is_voting_ensemble=is_voting_ensemble,
        )

        _, test_scores = evaluator.evaluate_model_perf(class_encoder=None)

        test_metrics = evaluator.convert_metrics_from_df_to_dict(
            scores=test_scores, prefix="test_"
        )

        logger.info("Evaluated %s on test set", model_name)

        return test_metrics

    def calibrate_and_register_champion(
        self,
        experiment: ExistingExperiment,
        model_pipeline: "Pipeline",
        model_name: str,
        valid_features: pd.DataFrame,
        valid_class: np.ndarray,
        champion_manager: ModelChampionManager,
        cv_folds: int = 5,
    ) -> None:
        """Calibrates and registers champion model.

        Args:
            experiment: Comet ExistingExperiment object.
            model_pipeline: Fitted model pipeline (Pipeline).
            model_name: Name of the champion model.
            valid_features: Validation features for calibration.
            valid_class: Validation class labels for calibration.
            champion_manager: ModelChampionManager instance.
            cv_folds: Number of cross-validation folds for calibration.
        """
        # Calibrate the champion model
        calibrated_pipeline = champion_manager.calibrate_pipeline(
            valid_features=valid_features,
            valid_class=valid_class,
            fitted_pipeline=model_pipeline,
            cv_folds=cv_folds,
        )

        logger.info("Calibrated champion model: %s", model_name)

        # Set tracker and register
        tracker = CometExperimentTracker(experiment=experiment)
        champion_manager.tracker = tracker

        champion_manager.log_and_register_champ_model(
            local_path=self.artifacts_path,
            pipeline=calibrated_pipeline,
        )

        logger.info("Registered champion model: %s", model_name)

        # Save champion model locally
        champion_model_path = (
            f"{self.artifacts_path}/{champion_manager.champ_model_name}.pkl"
        )
        joblib.dump(model_pipeline, champion_model_path)

        logger.info("Saved champion model to: %s", champion_model_path)

    def run_evaluation_workflow(
        self,
        comet_api_key: str,
        experiment_keys: pd.DataFrame,
        model_selector: ModelSelector,
        valid_features: pd.DataFrame,
        valid_class: np.ndarray,
        champion_manager: ModelChampionManager,
        comparison_metric_name: str,
        deployment_threshold: float,
        cv_folds: int = 5,
    ) -> tuple[str, dict]:
        """Runs complete evaluation workflow: select, evaluate, calibrate, register.

        Args:
            comet_api_key: Comet API key.
            experiment_keys: DataFrame with model names and experiment keys.
            model_selector: ModelSelector instance.
            valid_features: Validation features for calibration.
            valid_class: Validation class labels for calibration.
            champion_manager: ModelChampionManager instance.
            comparison_metric_name: Metric name for deployment decision.
            deployment_threshold: Minimum score required for deployment.
            cv_folds: Number of CV folds for calibration.

        Returns:
            Tuple of (champion_model_name, test_metrics).

        Raises:
            ValueError: If test score is below deployment threshold.
        """
        # Select best model
        best_model_name, best_model_exp_key = model_selector.select_best_model(
            experiment_keys=experiment_keys
        )

        # Load model pipeline
        model_path = f"{self.artifacts_path}/{best_model_name}.pkl"
        model_pipeline = joblib.load(model_path)
        logger.info("Loaded model from: %s", model_path)

        # Create existing experiment for logging
        experiment = ExistingExperiment(
            api_key=comet_api_key, experiment_key=best_model_exp_key
        )

        # Evaluate on test set
        test_metrics = self.evaluate_on_test_set(
            experiment=experiment,
            model_pipeline=model_pipeline,
            model_name=best_model_name,
        )

        # Log test metrics
        experiment.log_metrics(test_metrics)

        # Check deployment threshold
        test_score = test_metrics.get(f"test_{comparison_metric_name}")
        if test_score < deployment_threshold:
            raise ValueError(
                f"Best model score ({test_score:.4f}) is below deployment "
                f"threshold ({deployment_threshold:.4f}). Model not deployed."
            )

        logger.info(
            "Test score (%s: %.4f) meets deployment threshold (%.4f)",
            comparison_metric_name,
            test_score,
            deployment_threshold,
        )

        # Calibrate and register as champion
        self.calibrate_and_register_champion(
            experiment=experiment,
            model_pipeline=model_pipeline,
            model_name=best_model_name,
            valid_features=valid_features,
            valid_class=valid_class,
            champion_manager=champion_manager,
            cv_folds=cv_folds,
        )

        experiment.end()

        return best_model_name, test_metrics
