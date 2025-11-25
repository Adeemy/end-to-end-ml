"""
Experiment management utilities for creating and managing experiments.

This module defines an abstract base class `ExperimentManager` and a concrete implementation
`CometExperimentManager` for handling experiment tracking. Other experiment tracking backends
can be implemented by inheriting from `ExperimentManager`.

Classes:
    ExperimentManager: Abstract base class defining the interface for experiment management.
    CometExperimentManager: Concrete implementation using Comet ML.

Design Decision:
    `ExperimentManager` is implemented as an Abstract Base Class (ABC) to enforce explicit
    intent and stronger runtime safety. By inheriting from `ExperimentManager`, subclasses
    explicitly declare their role as experiment managers, and instantiation is prevented if
    abstract methods are missing. This fits the "Strategy" pattern where we want to ensure
    strict adherence to the experiment tracking interface.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path, PosixPath
from typing import Any, Dict, Optional

import joblib
import mlflow
import mlflow.sklearn
from comet_ml import Experiment as CometExperiment
from sklearn.pipeline import Pipeline

from src.training.tracking.experiment_tracker import (
    CometExperimentTracker,
    ExperimentTracker,
    MLflowExperimentTracker,
)
from src.utils.logger import get_console_logger

module_name: str = PosixPath(__file__).stem
logger = get_console_logger(module_name)


class ExperimentManager(ABC):
    """Abstract base class for experiment management.

    Allows swapping different experiment tracking backends (e.g., Comet ML, MLflow).
    Includes credential management for each tracker type.
    """

    @abstractmethod
    def get_tracker_type(self) -> str:
        """Get the tracker type identifier."""

    @abstractmethod
    def get_credentials(self) -> Dict[str, str]:
        """Get credentials from environment variables."""

    @abstractmethod
    def get_base_config(self, experiment_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Get base configuration for standalone experiments."""

    @abstractmethod
    def should_initialize_project(self, config_params: Dict[str, Any]) -> bool:
        """Check if project initialization is needed."""

    @abstractmethod
    def initialize_project(
        self, project_name: str, workspace_name: str, credentials: Dict[str, str]
    ) -> None:
        """Initialize experiment tracking project."""

    @abstractmethod
    def create_experiment(
        self, project_name: str, experiment_name: str, api_key: Optional[str] = None
    ) -> Any:
        """Creates an experiment instance.

        Args:
            project_name: Name of the project.
            experiment_name: Name of the experiment.
            api_key: API key for the experiment tracking service.

        Returns:
            Experiment instance.
        """
        raise NotImplementedError

    @abstractmethod
    def get_tracker(self, experiment: Any) -> ExperimentTracker:
        """Returns an experiment tracker for the given experiment.

        Args:
            experiment: Experiment instance.

        Returns:
            ExperimentTracker: Experiment tracker instance.
        """
        raise NotImplementedError

    @abstractmethod
    def log_metrics(self, experiment: Any, metrics: dict) -> None:
        """Logs metrics to the experiment.

        Args:
            experiment: Experiment instance.
            metrics: Dictionary of metrics to log.
        """
        raise NotImplementedError

    @abstractmethod
    def log_parameters(self, experiment: Any, params: dict) -> None:
        """Logs parameters to the experiment.

        Args:
            experiment: Experiment instance.
            params: Dictionary of parameters to log.
        """
        raise NotImplementedError

    @abstractmethod
    def log_asset(self, experiment: Any, file_path: str, file_name: str) -> None:
        """Logs an asset file to the experiment.

        Args:
            experiment: Experiment instance.
            file_path: Path to the asset file.
            file_name: Name for the asset in the experiment.
        """
        raise NotImplementedError

    @abstractmethod
    def register_model(
        self,
        experiment: Any,
        pipeline: Pipeline,
        registered_model_name: str,
        artifacts_path: str = "model",
    ) -> None:
        """Saves and registers the model.

        Args:
            experiment: Experiment instance.
            pipeline: Fitted pipeline object.
            registered_model_name: Name of the registered model.
            artifacts_path: Path to save model artifacts.
        """
        raise NotImplementedError

    @abstractmethod
    def end_experiment(self, experiment: Any) -> None:
        """Ends the experiment.

        Args:
            experiment: Experiment instance.
        """
        raise NotImplementedError


def create_experiment_manager(tracker_type: str) -> ExperimentManager:
    """Factory function to create experiment managers based on tracker type.

    Args:
        tracker_type: Type of tracker (e.g., 'comet', 'mlflow').

    Returns:
        ExperimentManager instance for the specified tracker.

    Raises:
        ValueError: If tracker type is not supported.
    """
    # Simple registry dictionary
    _managers = {
        "comet": CometExperimentManager,
        "mlflow": MLflowExperimentManager,
    }

    tracker_type = tracker_type.lower()

    if tracker_type not in _managers:
        available = ", ".join(_managers.keys())
        raise ValueError(
            f"Unsupported tracker type: {tracker_type}. Available: {available}"
        )

    return _managers[tracker_type]()


# Convenience functions for the consolidated API
def get_tracker_credentials(tracker_type: str) -> Dict[str, str]:
    """Get credentials for a tracker type."""
    manager = create_experiment_manager(tracker_type)
    return manager.get_credentials()


def get_tracker_base_config(
    tracker_type: str, experiment_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Get base configuration for a tracker type."""
    manager = create_experiment_manager(tracker_type)
    return manager.get_base_config(experiment_kwargs)


def should_initialize_tracker_project(
    tracker_type: str, config_params: Dict[str, Any]
) -> bool:
    """Check if project initialization is needed."""
    try:
        manager = create_experiment_manager(tracker_type)
        return manager.should_initialize_project(config_params)
    except ValueError:
        return False


def initialize_tracker_project(
    tracker_type: str,
    project_name: str,
    workspace_name: str,
    credentials: Dict[str, str],
) -> None:
    """Initialize project for a tracker."""
    try:
        manager = create_experiment_manager(tracker_type)
        manager.initialize_project(project_name, workspace_name, credentials)
    except ValueError:
        pass  # Silently ignore unsupported trackers


class CometExperimentManager(ExperimentManager):
    """Manages Comet ML experiments including creation, logging, and model registration."""

    def get_tracker_type(self) -> str:
        """Get tracker type identifier."""
        return "comet"

    def get_credentials(self) -> Dict[str, str]:
        """Get Comet ML credentials from environment variables."""
        credentials = {}
        api_key = os.environ.get("COMET_API_KEY")
        if api_key:
            credentials["api_key"] = api_key
        return credentials

    def get_base_config(self, experiment_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Get Comet ML specific base configuration."""
        return {
            "project_name": experiment_kwargs.get("project_name"),
            "workspace": experiment_kwargs.get("workspace_name"),
        }

    def should_initialize_project(self, config_params: Dict[str, Any]) -> bool:
        """Check if Comet project should be initialized."""
        return bool(config_params.get("initiate_comet_project", False))

    def initialize_project(
        self, project_name: str, workspace_name: str, credentials: Dict[str, str]
    ) -> None:
        """Initialize Comet ML project."""
        api_key = credentials.get("api_key")
        if api_key:
            import comet_ml

            comet_ml.init(
                project_name=project_name,
                workspace=workspace_name,
                api_key=api_key,
            )

    def create_experiment(
        self, project_name: str, experiment_name: str, api_key: Optional[str] = None
    ) -> Any:
        """Creates a Comet experiment instance.

        Args:
            project_name: Comet project name.
            experiment_name: Comet experiment name.
            api_key: Comet API key.

        Returns:
            Experiment: Comet experiment instance.

        Raises:
            ValueError: if Comet experiment creation fails or api_key is missing.
        """
        if not api_key:
            raise ValueError("Comet ML requires an API key.")

        try:

            experiment = CometExperiment(api_key=api_key, project_name=project_name)
            experiment.log_code(folder=".")
            experiment.set_name(experiment_name)
            logger.info("Created Comet experiment: %s", experiment_name)
            return experiment
        except ValueError as e:
            raise ValueError(f"Comet experiment creation error --> {e}") from e

    def get_tracker(self, experiment: Any) -> ExperimentTracker:
        """Returns a Comet experiment tracker.

        Args:
            experiment: Comet experiment instance.

        Returns:
            ExperimentTracker: Comet experiment tracker.
        """
        return CometExperimentTracker(experiment=experiment)

    def log_metrics(self, experiment: Any, metrics: dict) -> None:
        """Logs metrics to Comet experiment.

        Args:
            experiment: Comet experiment instance.
            metrics: Dictionary of metrics to log.
        """
        experiment.log_metrics(metrics)

    def log_parameters(self, experiment: Any, params: dict) -> None:
        """Logs parameters to Comet experiment.

        Args:
            experiment: Comet experiment instance.
            params: Dictionary of parameters to log.
        """
        experiment.log_parameters(params)

    def log_asset(self, experiment: Any, file_path: str, file_name: str) -> None:
        """Logs an asset file to Comet experiment.

        Args:
            experiment: Comet experiment instance.
            file_path: Path to the asset file.
            file_name: Name for the asset in Comet.
        """
        experiment.log_asset(file_data=file_path, file_name=file_name)

    def register_model(
        self,
        experiment: Any,
        pipeline: Pipeline,
        registered_model_name: str,
        artifacts_path: str = "model",
    ) -> None:
        """Saves and registers the model to Comet experiment.

        Args:
            experiment: Comet experiment instance.
            pipeline: Fitted pipeline object.
            registered_model_name: Name of the registered model.
            artifacts_path: Path to save model artifacts.
        """
        # Ensure artifacts directory exists
        artifacts_dir = Path(artifacts_path)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Create full model path
        model_path = artifacts_dir / f"{registered_model_name}.pkl"

        # Save model locally
        joblib.dump(pipeline, str(model_path))
        logger.info("Saved model to: %s", model_path)

        # Log model to Comet ML
        experiment.log_model(
            name=registered_model_name,
            file_or_folder=str(model_path),
            overwrite=False,
        )
        experiment.register_model(model_name=registered_model_name)
        logger.info("Registered model: %s", registered_model_name)

    def end_experiment(self, experiment: Any) -> None:
        """Ends a Comet experiment.

        Args:
            experiment: Comet experiment instance.
        """
        experiment.end()


class MLflowExperimentManager(ExperimentManager):
    """Manages MLflow experiments including creation, logging, and model registration.

    Single Responsibility: Handle all MLflow experiment interactions.

    Note: MLflow typically doesn't require API keys for local usage, but may need them
    for remote tracking servers or managed services like Databricks.
    """

    def get_tracker_type(self) -> str:
        """Get tracker type identifier."""
        return "mlflow"

    def get_credentials(self) -> Dict[str, str]:
        """Get MLflow credentials from environment variables."""
        credentials = {}
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if tracking_uri:
            credentials["tracking_uri"] = tracking_uri

        username = os.environ.get("MLFLOW_USERNAME")
        if username:
            credentials["username"] = username

        password = os.environ.get("MLFLOW_PASSWORD")
        if password:
            credentials["password"] = password

        return credentials

    def get_base_config(self, experiment_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Get MLflow specific base configuration."""
        return {}  # MLflow uses experiment_name directly

    def should_initialize_project(self, config_params: Dict[str, Any]) -> bool:
        """MLflow doesn't require explicit project initialization."""
        return False

    def initialize_project(
        self, project_name: str, workspace_name: str, credentials: Dict[str, str]
    ) -> None:
        """MLflow doesn't require project initialization."""
        pass  # pylint: disable=unnecessary-pass

    def create_experiment(
        self, project_name: str, experiment_name: str, api_key: Optional[str] = None
    ) -> Any:
        """Creates an MLflow run within an experiment.

        Args:
            project_name: Project name of MLflow experiment.
            experiment_name: Run name within the experiment (maps to Comet's experiment_name).
            api_key: Optional API key - not typically needed for local MLflow usage.

        Returns:
            MLflow run object.

        Raises:
            ImportError: if MLflow is not installed.
            ValueError: if experiment creation fails.
        """

        try:
            # Set experiment and start run with proper context
            mlflow.set_experiment(project_name)
            run = mlflow.start_run(run_name=experiment_name)
            return run
        except Exception as e:
            raise ValueError(f"MLflow experiment creation error --> {e}") from e

    def get_tracker(self, experiment: Any) -> ExperimentTracker:
        """Returns an MLflow experiment tracker.

        Args:
            experiment: MLflow run object.

        Returns:
            ExperimentTracker: MLflow experiment tracker.
        """
        return MLflowExperimentTracker(run_id=experiment.info.run_id)

    def log_metrics(self, experiment: Any, metrics: dict) -> None:
        """Logs metrics to MLflow run.

        Args:
            experiment: MLflow run object.
            metrics: Dictionary of metrics to log.
        """
        logger.info("Logging metrics to experiment ID: %s", experiment.info.run_id)
        mlflow.log_metrics(metrics)

    def log_parameters(self, experiment: Any, params: dict) -> None:
        """Logs parameters to MLflow run.

        Args:
            experiment: MLflow run object.
            params: Dictionary of parameters to log.
        """
        logger.info("Logging parameters to experiment ID: %s", experiment.info.run_id)
        mlflow.log_params(params)

    def log_asset(self, experiment: Any, file_path: str, file_name: str) -> None:
        """Logs an asset file to MLflow run.

        Args:
            experiment: MLflow run object.
            file_path: Path to the asset file.
            file_name: Name for the asset.
        """
        logger.info("Logging assets to experiment ID: %s", experiment.info.run_id)
        mlflow.log_artifact(file_path, artifact_path=f"assets/{file_name}")

    def register_model(
        self,
        experiment: Any,
        pipeline: Pipeline,
        registered_model_name: str,
        artifacts_path: str = "model",
    ) -> None:
        """Saves and registers the model to MLflow.

        Args:
            experiment: MLflow run object.
            pipeline: Fitted pipeline object.
            registered_model_name: Name of the registered model.
            artifacts_path: Path to save model artifacts.
        """

        # Log the model to MLflow
        try:
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path=registered_model_name,
                registered_model_name=registered_model_name,
            )

            # Also save model locally for consistency with Comet implementation
            artifacts_dir = Path(artifacts_path)
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            model_path = artifacts_dir / f"{registered_model_name}.pkl"
            joblib.dump(pipeline, str(model_path))
            logger.info("Model also saved locally to: %s", model_path)

        except Exception as e:
            logger.error("Failed to log model to MLflow: %s", e)
            raise
        logger.info("Model registration completed: %s", registered_model_name)

    def end_experiment(self, experiment: Any) -> None:
        """Ends an MLflow run.

        Args:
            experiment: MLflow run object.
        """
        mlflow.end_run()
