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

from abc import ABC, abstractmethod
from pathlib import PosixPath
from typing import Any

import joblib
from comet_ml import Experiment
from sklearn.pipeline import Pipeline

from src.utils.logger import get_console_logger

module_name: str = PosixPath(__file__).stem
logger = get_console_logger(module_name)


class ExperimentManager(ABC):
    """Abstract base class for experiment management.

    Allows swapping different experiment tracking backends (e.g., Comet ML, MLflow).
    """

    @abstractmethod
    def create_experiment(
        self, comet_api_key: str, project_name: str, experiment_name: str
    ) -> Any:
        """Creates an experiment object."""
        raise NotImplementedError

    @abstractmethod
    def log_metrics(self, experiment: Any, metrics: dict) -> None:
        """Logs metrics to the experiment."""
        raise NotImplementedError

    @abstractmethod
    def log_parameters(self, experiment: Any, params: dict) -> None:
        """Logs parameters to the experiment."""
        raise NotImplementedError

    @abstractmethod
    def log_asset(self, experiment: Any, file_path: str, file_name: str) -> None:
        """Logs an asset file to the experiment."""
        raise NotImplementedError

    @abstractmethod
    def register_model(
        self,
        experiment: Any,
        pipeline: Pipeline,
        registered_model_name: str,
        artifacts_path: str,
    ) -> None:
        """Saves and registers the model."""
        raise NotImplementedError

    @abstractmethod
    def end_experiment(self, experiment: Any) -> None:
        """Ends the experiment."""
        raise NotImplementedError


class CometExperimentManager(ExperimentManager):
    """Manages Comet ML experiments including creation, logging, and model registration.

    Single Responsibility: Handle all Comet ML experiment interactions.
    """

    def create_experiment(
        self, comet_api_key: str, project_name: str, experiment_name: str
    ) -> Experiment:
        """Creates a Comet experiment object.

        Args:
            comet_api_key: Comet API key.
            project_name: Comet project name.
            experiment_name: Comet experiment name.

        Returns:
            Experiment: Comet experiment object.

        Raises:
            ValueError: if Comet experiment creation fails.
        """
        try:
            experiment = Experiment(api_key=comet_api_key, project_name=project_name)
            experiment.log_code(folder=".")
            experiment.set_name(experiment_name)
            logger.info("Created Comet experiment: %s", experiment_name)
            return experiment
        except ValueError as e:
            raise ValueError(f"Comet experiment creation error --> {e}") from e

    def log_metrics(self, experiment: Experiment, metrics: dict) -> None:
        """Logs metrics to Comet experiment.

        Args:
            experiment: Comet experiment object.
            metrics: Dictionary of metrics to log.
        """
        experiment.log_metrics(metrics)

    def log_parameters(self, experiment: Experiment, params: dict) -> None:
        """Logs parameters to Comet experiment.

        Args:
            experiment: Comet experiment object.
            params: Dictionary of parameters to log.
        """
        experiment.log_parameters(params)

    def log_asset(self, experiment: Experiment, file_path: str, file_name: str) -> None:
        """Logs an asset file to Comet experiment.

        Args:
            experiment: Comet experiment object.
            file_path: Path to the asset file.
            file_name: Name for the asset in Comet.
        """
        experiment.log_asset(file_data=file_path, file_name=file_name)

    def register_model(
        self,
        experiment: Experiment,
        pipeline: Pipeline,
        registered_model_name: str,
        artifacts_path: str,
    ) -> None:
        """Saves and registers the model to Comet experiment.

        Args:
            experiment: Comet experiment object.
            pipeline: Fitted pipeline object.
            registered_model_name: Name of the registered model.
            artifacts_path: Path to save model artifacts.
        """
        model_path = f"{artifacts_path}/{registered_model_name}.pkl"
        joblib.dump(pipeline, model_path)
        experiment.log_model(
            name=registered_model_name,
            file_or_folder=model_path,
            overwrite=False,
        )
        experiment.register_model(model_name=registered_model_name)
        logger.info("Registered model: %s", registered_model_name)

    def end_experiment(self, experiment: Experiment) -> None:
        """Ends a Comet experiment.

        Args:
            experiment: Comet experiment object.
        """
        experiment.end()
