"""
Test set evaluation orchestration - evaluates models on held-out test data.
"""

import os
from datetime import datetime
from pathlib import PosixPath
from typing import Any, Callable, Optional

# Load comet_ml early to avoid issues with sklearn auto-logging
import comet_ml  # pylint: disable=unused-import
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.training.evaluation.champion import ModelChampionManager
from src.training.evaluation.evaluator import create_model_evaluator
from src.training.evaluation.selector import ModelSelector
from src.training.tracking.experiment import (
    get_tracker_base_config,
    get_tracker_credentials,
)
from src.training.tracking.experiment_tracker import (
    CometExperimentTracker,
    ExperimentTracker,
    MLflowExperimentTracker,
)
from src.utils.logger import get_console_logger

module_name: str = PosixPath(__file__).stem
logger = get_console_logger(module_name)


class TrackerRegistry:
    """Registry for experiment tracker factories. It allows for easy addition
    of new tracker types without modifying existing code. It follows the factory
    pattern to create tracker instances based on registered factory functions. If
    new tracker types are needed, they can be registered without changing the core
    registry implementation.
    """

    def __init__(self):
        """Initialize the registry with an empty tracker dictionary."""
        self._trackers = {}

    def register(
        self, name: str, factory_func: Callable[..., ExperimentTracker]
    ) -> None:
        """Register a tracker factory function.

        Args:
            name: Name of the tracker type (e.g., 'comet', 'mlflow').
            factory_func: Factory function that creates the tracker instance.
        """
        self._trackers[name.lower()] = factory_func

    def create_tracker(self, tracker_type: str, **kwargs) -> ExperimentTracker:
        """Create a tracker instance using registered factory.

        Args:
            tracker_type: Type of tracker to create.
            **kwargs: Arguments to pass to the factory function.

        Returns:
            ExperimentTracker instance.

        Raises:
            ValueError: If tracker type is not registered.
        """
        factory_func = self._trackers.get(tracker_type.lower())
        if not factory_func:
            available_types = ", ".join(self._trackers.keys())
            raise ValueError(
                f"Unsupported tracker type: {tracker_type}. "
                f"Available types: {available_types}"
            )
        return factory_func(**kwargs)


def _create_comet_tracker(
    experiment_instance=None, **kwargs  # pylint: disable=unused-argument
) -> CometExperimentTracker:
    """Factory function for creating Comet tracker. If no experiment instance is provided,
    a dummy disabled experiment is created.

    Args:
        experiment_instance: Pre-initialized Comet experiment instance (optional).
        **kwargs: Additional arguments (ignored for compatibility).

    Returns:
        CometExperimentTracker instance.
    """
    if experiment_instance is not None:
        return CometExperimentTracker(experiment=experiment_instance)
    else:
        # Create a dummy experiment for now - will be set later via set_experiment
        from comet_ml import Experiment

        dummy_experiment = Experiment(disabled=True)
        return CometExperimentTracker(experiment=dummy_experiment)


def _create_mlflow_tracker(
    run_id=None, **kwargs  # pylint: disable=unused-argument
) -> MLflowExperimentTracker:
    """Factory function for creating MLflow tracker. If no run_id is provided,
    a new run will be started.

    Args:
        run_id: Optional MLflow run ID.
        **kwargs: Additional arguments (ignored for compatibility).

    Returns:
        MLflowExperimentTracker instance.
    """
    return MLflowExperimentTracker(run_id=run_id)


# Create and configure the default tracker registry
_default_tracker_registry = TrackerRegistry()
_default_tracker_registry.register("comet", _create_comet_tracker)
_default_tracker_registry.register("mlflow", _create_mlflow_tracker)


def create_evaluation_orchestrator(
    tracker_type: str,
    train_features: pd.DataFrame,
    train_class: np.ndarray,
    test_features: pd.DataFrame,
    test_class: np.ndarray,
    artifacts_path: str,
    fbeta_score_beta: float = 1.0,
    voting_ensemble_name: Optional[str] = None,
    experiment_instance: Optional[Any] = None,
    **tracker_kwargs,
) -> "TestSetEvaluationOrchestrator":
    """Factory function to create TestSetEvaluationOrchestrator with appropriate tracker.

    Args:
        tracker_type: Type of tracker to use (e.g., 'comet', 'mlflow').
        train_features: Training features.
        train_class: Training class labels.
        test_features: Test features.
        test_class: Test class labels.
        artifacts_path: Path to model artifacts.
        fbeta_score_beta: Beta value for fbeta score.
        voting_ensemble_name: Name of voting ensemble model (if exists).
        experiment_instance: Pre-initialized experiment instance (for Comet).
        **tracker_kwargs: Additional arguments for tracker initialization.

    Returns:
        TestSetEvaluationOrchestrator instance with appropriate tracker.

    Raises:
        ValueError: If unsupported tracker_type is provided.
    """

    # Create tracker using registry pattern
    tracker = _default_tracker_registry.create_tracker(
        tracker_type=tracker_type,
        experiment_instance=experiment_instance,
        **tracker_kwargs,
    )

    return TestSetEvaluationOrchestrator(
        tracker=tracker,
        train_features=train_features,
        train_class=train_class,
        test_features=test_features,
        test_class=test_class,
        artifacts_path=artifacts_path,
        fbeta_score_beta=fbeta_score_beta,
        voting_ensemble_name=voting_ensemble_name,
    )


class TestSetEvaluationOrchestrator:
    """Orchestrates model evaluation on test set and champion model registration.

    Single Responsibility: Coordinate test evaluation and champion selection.
    Dependency Inversion: Depends on abstractions (ModelSelector, ModelEvaluator, ExperimentTracker).
    """

    def __init__(
        self,
        tracker: ExperimentTracker,
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
            tracker: Experiment tracker instance for logging.
            train_features: Training features.
            train_class: Training class labels.
            test_features: Test features.
            test_class: Test class labels.
            artifacts_path: Path to model artifacts.
            fbeta_score_beta: Beta value for fbeta score.
            voting_ensemble_name: Name of voting ensemble model (if exists).
        """
        self.tracker = tracker
        self.train_features = train_features
        self.train_class = train_class
        self.test_features = test_features
        self.test_class = test_class
        self.artifacts_path = artifacts_path
        self.fbeta_score_beta = fbeta_score_beta
        self.voting_ensemble_name = voting_ensemble_name

    def evaluate_on_test_set(
        self,
        model_pipeline: "Pipeline",
        model_name: str,
    ) -> dict:
        """Evaluates model on test set.

        Args:
            model_pipeline: Fitted model pipeline (Pipeline).
            model_name: Name of the model being evaluated.

        Returns:
            Dictionary of test metrics.
        """

        is_voting_ensemble = (
            model_name == self.voting_ensemble_name
            if self.voting_ensemble_name
            else False
        )

        evaluator = create_model_evaluator(
            tracker=self.tracker,
            pipeline=model_pipeline,
            train_features=self.train_features,
            train_class=self.train_class,
            valid_features=self.test_features,
            valid_class=self.test_class,
            fbeta_score_beta=self.fbeta_score_beta,
            is_voting_ensemble=is_voting_ensemble,
        )

        # Create class encoder for confusion matrix logging
        # Fit on combined train and test class labels to ensure all labels are known
        class_encoder = LabelEncoder()
        all_class_labels = np.concatenate([self.train_class, self.test_class])
        class_encoder.fit(all_class_labels)

        # Use the public method for test-only evaluation to avoid accessing protected members
        test_scores = evaluator.evaluate_test_set_only(class_encoder=class_encoder)

        test_metrics = evaluator.convert_metrics_from_df_to_dict(
            scores=test_scores, prefix="test_"
        )

        logger.info(
            "Evaluated %s on test set. Test metrics: %s", model_name, test_metrics
        )

        return test_metrics

    def calibrate_and_register_champion(
        self,
        model_pipeline: "Pipeline",
        model_name: str,
        valid_features: pd.DataFrame,
        valid_class: np.ndarray,
        champion_manager: ModelChampionManager,
        cv_folds: int = 5,
    ) -> None:
        """Calibrates and registers champion model.

        Args:
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
        champion_manager.tracker = self.tracker

        champion_manager.log_and_register_champ_model(
            local_path=self.artifacts_path,
            pipeline=calibrated_pipeline,
        )

        try:
            evaluation_exp_name = (
                self.tracker.experiment.get_name()
                if hasattr(self.tracker, "experiment")
                else "unknown"
            )
        except AttributeError:
            evaluation_exp_name = "unknown"
        logger.info(
            "Registered champion model: %s in evaluation experiment: %s",
            model_name,
            evaluation_exp_name,
        )

        # Save champion model locally
        champion_model_path = (
            f"{self.artifacts_path}/{champion_manager.champ_model_name}.pkl"
        )
        joblib.dump(model_pipeline, champion_model_path)

        logger.info("Saved champion model to: %s", champion_model_path)

    def _create_standalone_evaluation_kwargs(
        self, experiment_kwargs: dict, experiment_name: str
    ) -> dict:
        """Create tracker-agnostic kwargs for standalone evaluation experiments.

        Args:
            experiment_kwargs: Original experiment configuration.
            experiment_name: Name for the evaluation experiment.

        Returns:
            Dictionary of experiment kwargs for the specific tracker.
        """
        tracker_type = experiment_kwargs.get("experiment_tracker_type", "comet")

        # Start with base configuration
        kwargs = {
            "experiment_name": experiment_name,
        }

        # Add tracker-specific base configuration using registry
        try:
            base_config = get_tracker_base_config(tracker_type, experiment_kwargs)
            kwargs.update(base_config)
        except ValueError:
            logger.warning(
                "Unknown tracker type: %s, using generic fallback", tracker_type
            )
            # Generic fallback - include non-tracker-type keys
            kwargs.update(
                {
                    k: v
                    for k, v in experiment_kwargs.items()
                    if k not in ["experiment_tracker_type"]
                }
            )

        # Add credentials using the credential provider
        try:
            credentials = get_tracker_credentials(tracker_type)
            kwargs.update(credentials)
        except ValueError:
            logger.warning(
                "Could not get credentials for tracker type: %s", tracker_type
            )

        return kwargs

    def run_evaluation_workflow(
        self,
        model_selector: ModelSelector,
        valid_features: pd.DataFrame,
        valid_class: np.ndarray,
        champion_manager: ModelChampionManager,
        comparison_metric_name: str,
        deployment_threshold: float,
        cv_folds: int = 5,
        experiment_keys: Optional[pd.DataFrame] = None,
        max_eval_experiments: int = 50,
        **experiment_kwargs,
    ) -> tuple[str, dict]:
        """Runs complete evaluation workflow: select, evaluate, calibrate, register.

        Args:
            model_selector: ModelSelector instance.
            valid_features: Validation features for calibration.
            valid_class: Validation class labels for calibration.
            champion_manager: ModelChampionManager instance.
            comparison_metric_name: Metric name for deployment decision.
            deployment_threshold: Minimum score required for deployment.
            cv_folds: Number of CV folds for calibration.
            experiment_keys: Optional DataFrame with model names and experiment keys.
                           If None, ModelSelector will query Comet ML directly.
            max_eval_experiments: Maximum number of recent experiments to consider.
            **experiment_kwargs: Additional arguments for experiment setup (e.g., api_key, experiment_key).

        Returns:
            Tuple of (champion_model_name, test_metrics).

        Raises:
            ValueError: If test score is below deployment threshold.
        """
        # Select best model
        best_model_name, best_experiment_key = model_selector.select_best_model(
            experiment_keys=experiment_keys, max_experiments=max_eval_experiments
        )

        # Try to load model locally, if not found download from Comet ML
        model_path = f"{self.artifacts_path}/{best_model_name}.pkl"

        model_pipeline = None
        if not os.path.exists(model_path):
            logger.info("Model not found locally, downloading from workspace")
            try:
                model_pipeline = self._download_model_from_comet(
                    experiment_key=best_experiment_key,
                    model_name=best_model_name,
                    save_path=model_path,
                )
            except FileNotFoundError as e:
                logger.warning("Could not download model from workspace: %s", str(e))
                logger.warning("Skipping evaluation for model: %s", best_model_name)
                return best_model_name, {}
            except Exception as e:  # pylint: disable=W0718
                logger.warning("Error downloading model from workspace: %s", str(e))
                logger.warning("Skipping evaluation for model: %s", best_model_name)
                return best_model_name, {}
        else:
            try:
                model_pipeline = joblib.load(model_path)
                logger.info("Loaded model from local path: %s", model_path)
            except Exception as e:  # pylint: disable=W0703
                logger.warning(
                    "Could not load model from local path %s: %s", model_path, str(e)
                )
                logger.warning("Skipping evaluation for model: %s", best_model_name)
                return best_model_name, {}

        if model_pipeline is None:
            logger.warning("No model pipeline available for evaluation")
            return best_model_name, {}

        # Handle experiment creation based on tracker type and available experiment ID
        if hasattr(self.tracker, "set_experiment"):
            tracker_type = experiment_kwargs.get("experiment_tracker_type", "comet")

            # Check if we're running from training pipeline (with experiment_keys) or standalone
            if experiment_keys is not None and best_experiment_key:
                # Running from training pipeline - create child/linked experiment
                evaluation_kwargs = {
                    "is_child_experiment": True,
                }

                # Add tracker-specific parent/child linking
                if tracker_type == "comet":
                    evaluation_kwargs["experiment_key"] = best_experiment_key
                elif tracker_type == "mlflow":
                    evaluation_kwargs["parent_run_id"] = best_experiment_key
                # Additional trackers can be added here without modifying existing code

                # Add credentials using the credential provider
                try:
                    credentials = get_tracker_credentials(tracker_type)
                    evaluation_kwargs.update(credentials)
                except ValueError:
                    logger.warning(
                        "Could not get credentials for tracker type: %s", tracker_type
                    )

                self.tracker.set_experiment(**evaluation_kwargs)
                logger.info(
                    "Creating child/linked evaluation experiment for model: %s (parent: %s)",
                    best_model_name,
                    best_experiment_key,
                )
            else:
                # Running standalone - create independent evaluation experiment
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                evaluation_experiment_name = (
                    f"eval_{best_model_name.replace('-', '_')}_{timestamp}"
                )
                evaluation_kwargs = self._create_standalone_evaluation_kwargs(
                    experiment_kwargs, evaluation_experiment_name
                )

                self.tracker.set_experiment(**evaluation_kwargs)
                logger.info(
                    "Creating standalone evaluation experiment: %s",
                    evaluation_experiment_name,
                )

        # Evaluate on test set
        test_metrics = self.evaluate_on_test_set(
            model_pipeline=model_pipeline,
            model_name=best_model_name,
        )

        # Log test metrics with evaluation experiment context
        try:
            evaluation_exp_name = (
                self.tracker.experiment.get_name()
                if hasattr(self.tracker, "experiment")
                else "unknown"
            )
        except AttributeError:
            evaluation_exp_name = "unknown"
        logger.info(
            "Evaluated %s on test set in experiment: %s. Test metrics: %s",
            best_model_name,
            evaluation_exp_name,
            test_metrics,
        )
        self.tracker.log_metrics(test_metrics)

        # Check deployment threshold
        metric_key = f"test_{comparison_metric_name}"
        test_score = test_metrics.get(metric_key)

        if test_score is None:
            available_metrics = ", ".join(test_metrics.keys())
            raise ValueError(
                f"Metric '{metric_key}' not found in test metrics. "
                f"Available metrics: {available_metrics}"
            )

        test_score = float(test_score)
        try:
            evaluation_exp_name = (
                self.tracker.experiment.get_name()
                if hasattr(self.tracker, "experiment")
                else "unknown"
            )
        except AttributeError:
            evaluation_exp_name = "unknown"
        if test_score < deployment_threshold:
            logger.error(
                "Deployment check failed in experiment %s: Best model score (%.4f) is below threshold (%.4f). Model not deployed.",
                evaluation_exp_name,
                test_score,
                deployment_threshold,
            )
            raise ValueError(
                f"Best model score ({test_score:.4f}) is below deployment "
                f"threshold ({deployment_threshold:.4f}). Model not deployed."
            )

        logger.info(
            "Deployment check passed in experiment %s: Test score (%s: %.4f) meets threshold (%.4f)",
            evaluation_exp_name,
            comparison_metric_name,
            test_score,
            deployment_threshold,
        )

        # Calibrate and register as champion
        self.calibrate_and_register_champion(
            model_pipeline=model_pipeline,
            model_name=best_model_name,
            valid_features=valid_features,
            valid_class=valid_class,
            champion_manager=champion_manager,
            cv_folds=cv_folds,
        )

        # End experiment if tracker supports it
        if hasattr(self.tracker, "end_experiment"):
            self.tracker.end_experiment()

        return best_model_name, test_metrics

    def _download_model_from_comet(
        self,
        experiment_key: str,
        model_name: str,
        save_path: str,
    ) -> "Pipeline":
        """Downloads model from Comet ML experiment.

        Args:
            experiment_key: Comet ML experiment key.
            model_name: Name of the model to download.
            save_path: Local path to save the downloaded model.

        Returns:
            Loaded model pipeline.
        """
        # Login and get API
        comet_ml.login()
        api = comet_ml.API()

        # Get experiment
        experiment = api.get_experiment_by_key(experiment_key)

        # Download model assets
        assets = experiment.get_asset_list()
        model_assets = [
            asset
            for asset in assets
            if asset["fileName"].endswith(".pkl") and model_name in asset["fileName"]
        ]

        if not model_assets:
            raise FileNotFoundError(
                f"No model file found for {model_name} in experiment {experiment_key}. "
                f"Available assets: {len(assets)} total. "
                f"Model files (.pkl): {len([a for a in assets if a['fileName'].endswith('.pkl')])}"
            )

        # Download the model file
        model_asset = model_assets[0]  # Take the first matching model
        asset_id = model_asset["assetId"]

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Download and save
        experiment.get_asset(asset_id, save_path)
        logger.info("Downloaded model from Comet ML to: %s", save_path)

        # Load and return the model
        return joblib.load(save_path)
