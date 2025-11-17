"""
Experiment management utilities for creating and managing Comet ML experiments.
"""

from pathlib import PosixPath

import joblib
from comet_ml import Experiment
from sklearn.pipeline import Pipeline

from src.utils.logger import get_console_logger

module_name: str = PosixPath(__file__).stem
logger = get_console_logger(module_name)


class ExperimentManager:
    """Manages Comet ML experiments including creation, logging, and model registration.

    Single Responsibility: Handle all Comet ML experiment interactions.
    """

    @staticmethod
    def create_experiment(
        comet_api_key: str, project_name: str, experiment_name: str
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

    @staticmethod
    def log_metrics(experiment: Experiment, metrics: dict) -> None:
        """Logs metrics to Comet experiment.

        Args:
            experiment: Comet experiment object.
            metrics: Dictionary of metrics to log.
        """
        experiment.log_metrics(metrics)

    @staticmethod
    def log_parameters(experiment: Experiment, params: dict) -> None:
        """Logs parameters to Comet experiment.

        Args:
            experiment: Comet experiment object.
            params: Dictionary of parameters to log.
        """
        experiment.log_parameters(params)

    @staticmethod
    def log_asset(experiment: Experiment, file_path: str, file_name: str) -> None:
        """Logs an asset file to Comet experiment.

        Args:
            experiment: Comet experiment object.
            file_path: Path to the asset file.
            file_name: Name for the asset in Comet.
        """
        experiment.log_asset(file_data=file_path, file_name=file_name)

    @staticmethod
    def register_model(
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

    @staticmethod
    def end_experiment(experiment: Experiment) -> None:
        """Ends a Comet experiment.

        Args:
            experiment: Comet experiment object.
        """
        experiment.end()
