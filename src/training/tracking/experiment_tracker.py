"""
Abstract experiment tracking interface and concrete implementations.
"""

import json
import logging
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import mlflow
import numpy as np
from comet_ml import ExistingExperiment, Experiment
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class ExperimentTracker(ABC):
    """Abstract base class for experiment tracking backends."""

    @abstractmethod
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a single metric value.

        Args:
            name: Metric name.
            value: Metric value.
            step: Optional step/iteration number.
        """

    @abstractmethod
    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric names and values.
            step: Optional step/iteration number.
        """

    @abstractmethod
    def log_parameter(self, name: str, value: Any) -> None:
        """Log a single parameter.

        Args:
            name: Parameter name.
            value: Parameter value.
        """

    @abstractmethod
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters at once.

        Args:
            params: Dictionary of parameter names and values.
        """

    @abstractmethod
    def log_figure(
        self, figure_name: str, figure: Figure, step: Optional[int] = None, **kwargs
    ) -> None:
        """Log a matplotlib figure.

        Args:
            figure_name: Name for the figure.
            figure: Matplotlib figure object.
            step: Optional step/iteration number.
            **kwargs: Additional backend-specific arguments (e.g., overwrite for Comet).
        """

    @abstractmethod
    def log_confusion_matrix(
        self,
        matrix: np.ndarray,
        title: str,
        labels: Optional[list] = None,
        file_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log a confusion matrix.

        Args:
            matrix: Confusion matrix as numpy array.
            title: Title for the confusion matrix.
            labels: Optional class labels.
            file_name: Optional file name for saving.
            **kwargs: Additional backend-specific arguments.
        """

    @abstractmethod
    def log_model(
        self,
        name: str,
        file_or_folder: Union[str, Path],
        overwrite: bool = False,
        **kwargs,
    ) -> None:
        """Log a model artifact.

        Args:
            name: Model name.
            file_or_folder: Path to model file or folder.
            overwrite: Whether to overwrite existing model.
            **kwargs: Additional backend-specific arguments.
        """

    @abstractmethod
    def log_asset(self, file_path: str, file_name: str, **kwargs) -> None:
        """Log an asset file.

        Args:
            file_path: Path to the asset file.
            file_name: Name for the asset.
            **kwargs: Additional backend-specific arguments.
        """

    @abstractmethod
    def register_model(self, model_name: str, **kwargs) -> None:
        """Register a model in the model registry.

        Args:
            model_name: Name for the registered model.
            **kwargs: Additional backend-specific arguments.
        """

    @abstractmethod
    def get_metric(self, metric_name: str) -> Optional[float]:
        """Retrieve a logged metric value.

        Args:
            metric_name: Name of the metric to retrieve.

        Returns:
            Metric value if found, None otherwise.
        """

    @abstractmethod
    def end(self) -> None:
        """End the experiment tracking session."""

    def end_experiment(self) -> None:
        """Alias for end() method for compatibility."""
        self.end()


class CometExperimentTracker(ExperimentTracker):
    """Comet ML experiment tracker implementation."""

    def __init__(self, experiment: Any) -> None:
        """Initialize with a Comet experiment instance.

        Args:
            experiment: Comet Experiment or ExistingExperiment instance.
        """
        self.experiment = experiment

    def set_experiment(self, **kwargs) -> None:
        """Set experiment for Comet tracking.

        Args:
            **kwargs: Can contain:
                     - 'api_key' + 'experiment_key' for existing experiment
                     - 'api_key' + 'experiment_name' for new experiment
                     - 'is_child_experiment': True to create child experiment
        """

        if "experiment_key" in kwargs and "api_key" in kwargs:
            if kwargs.get("is_child_experiment", False):
                # Create a child experiment under the parent
                try:
                    parent_exp = ExistingExperiment(
                        api_key=kwargs["api_key"],
                        experiment_key=kwargs["experiment_key"],
                    )

                    self.experiment = Experiment(
                        api_key=kwargs["api_key"],
                        experiment_name=f"eval_{parent_exp.get_name() or 'model'}",
                        project_name=parent_exp.get_project_name(),
                        workspace=parent_exp.get_workspace(),
                    )

                    # Log parent relationship
                    self.experiment.log_parameter(
                        "parent_experiment_key", kwargs["experiment_key"]
                    )
                    self.experiment.log_parameter("evaluation_type", "test_evaluation")

                except Exception:  # pylint: disable=broad-except
                    # Fallback to existing experiment
                    self.experiment = ExistingExperiment(
                        api_key=kwargs["api_key"],
                        experiment_key=kwargs["experiment_key"],
                    )
            else:
                # Use existing experiment
                self.experiment = ExistingExperiment(
                    api_key=kwargs["api_key"], experiment_key=kwargs["experiment_key"]
                )
        elif "api_key" in kwargs:
            # Create new experiment
            experiment_name = kwargs.pop("experiment_name", None)
            self.experiment = Experiment(
                api_key=kwargs["api_key"],
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["api_key", "is_child_experiment"]
                },
            )

            # Set the experiment display name if provided
            if experiment_name:
                self.experiment.set_name(experiment_name)
        # If no api_key provided, assume experiment is already set

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a single metric value to Comet."""
        self.experiment.log_metric(name=name, value=value, step=step)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log multiple metrics to Comet."""
        self.experiment.log_metrics(metrics, step=step)

    def log_parameter(self, name: str, value: Any) -> None:
        """Log a single parameter to Comet."""
        self.experiment.log_parameter(name, value)

    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters to Comet."""
        self.experiment.log_parameters(params)

    def log_figure(
        self, figure_name: str, figure: Figure, step: Optional[int] = None, **kwargs
    ) -> None:
        """Log a matplotlib figure to Comet."""
        overwrite = kwargs.get("overwrite", False)
        self.experiment.log_figure(
            figure_name=figure_name, figure=figure, step=step, overwrite=overwrite
        )

    def log_confusion_matrix(
        self,
        matrix: np.ndarray,
        title: str,
        labels: Optional[list] = None,
        file_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log a confusion matrix to Comet."""
        self.experiment.log_confusion_matrix(
            matrix=matrix, title=title, labels=labels, file_name=file_name
        )

    def log_model(
        self,
        name: str,
        file_or_folder: Union[str, Path],
        overwrite: bool = False,
        **kwargs,
    ) -> None:
        """Log a model artifact to Comet."""
        self.experiment.log_model(
            name=name, file_or_folder=str(file_or_folder), overwrite=overwrite
        )

    def log_asset(self, file_path: str, file_name: str, **kwargs) -> None:
        """Log an asset file to Comet.

        Args:
            file_path: Path to the asset file.
            file_name: Name for the asset.
            **kwargs: Additional arguments (unused for Comet).
        """
        self.experiment.log_asset(file_data=file_path, file_name=file_name)

    def register_model(self, model_name: str, **kwargs) -> None:
        """Register a model in Comet's model registry."""
        self.experiment.register_model(model_name=model_name)

    def get_metric(self, metric_name: str) -> Optional[float]:
        """Retrieve a logged metric value from Comet."""
        metrics = self.experiment.get_metrics(metric_name)
        if metrics:
            return float(metrics[0]["metricValue"])
        return None

    def end(self) -> None:
        """End the Comet experiment."""
        self.experiment.end()


class MLflowExperimentTracker(ExperimentTracker):
    """MLflow experiment tracker implementation."""

    def __init__(self, run_id: Optional[str] = None) -> None:
        """Initialize MLflow tracker.

        Args:
            run_id: Optional MLflow run ID. If None, uses active run.
        """

        self.mlflow = mlflow
        self.run_id = run_id
        self._metrics_cache: Dict[str, float] = {}

    def set_experiment(self, **kwargs) -> None:
        """Set experiment for MLflow tracking.

        Args:
            **kwargs: Should contain MLflow-specific parameters like 'run_id', 'experiment_id', etc.
        """

        if "run_id" in kwargs:
            # Use existing run
            mlflow.start_run(run_id=kwargs["run_id"])
            self.run_id = kwargs["run_id"]
        elif "experiment_id" in kwargs:
            # Start new run in existing experiment
            mlflow.set_experiment(experiment_id=kwargs["experiment_id"])
            mlflow.start_run()
        elif "experiment_name" in kwargs:
            # Start run in experiment by name
            mlflow.set_experiment(kwargs["experiment_name"])
            mlflow.start_run()
        # If no specific parameters, assume run is already active

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a single metric value to MLflow."""
        self.mlflow.log_metric(name, value, step=step)
        self._metrics_cache[name] = value

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log multiple metrics to MLflow."""
        self.mlflow.log_metrics(metrics, step=step)
        self._metrics_cache.update(metrics)

    def log_parameter(self, name: str, value: Any) -> None:
        """Log a single parameter to MLflow."""
        self.mlflow.log_param(name, value)

    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters to MLflow."""
        self.mlflow.log_params(params)

    def log_figure(
        self, figure_name: str, figure: Figure, step: Optional[int] = None, **kwargs
    ) -> None:
        """Log a matplotlib figure to MLflow."""
        # MLflow doesn't have direct figure logging, so save as artifact

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            figure.savefig(tmp.name, bbox_inches="tight")
            self.mlflow.log_artifact(
                tmp.name, artifact_path=f"figures/{figure_name}.png"
            )

    def log_confusion_matrix(
        self,
        matrix: np.ndarray,
        title: str,
        labels: Optional[list] = None,
        file_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log a confusion matrix to MLflow."""

        # Save confusion matrix as JSON artifact
        # Convert numpy types to native Python types to avoid JSON serialization issues
        cm_data = {
            "matrix": matrix.astype(
                int
            ).tolist(),  # Convert to int to avoid int64 JSON serialization issues
            "title": str(title),
            "labels": [str(label) for label in labels] if labels else [],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(cm_data, tmp)
            artifact_name = file_name if file_name else f"{title}.json"
            self.mlflow.log_artifact(
                tmp.name, artifact_path=f"confusion_matrices/{artifact_name}"
            )

    def log_model(
        self,
        name: str,
        file_or_folder: Union[str, Path],
        overwrite: bool = False,
        **kwargs,
    ) -> None:
        """Log a model artifact to MLflow."""
        # For MLflow, we need to log the actual model object, not just the file
        # Load the model from the pickle file and log it properly
        import joblib

        try:
            # Load the model from the pickle file
            model = joblib.load(str(file_or_folder))

            # Log the model using MLflow's sklearn integration
            # This creates a proper MLflow model with all artifacts and metadata
            self.mlflow.sklearn.log_model(
                sk_model=model,
                name=name,  # This will be the path in the MLflow run
                registered_model_name=None,  # Don't register here, do it separately
            )
            logger.info("Successfully logged model %s to MLflow", name)

        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to load and log model %s: %s", name, e)
            # Fallback to just logging as artifact
            self.mlflow.log_artifact(
                str(file_or_folder), artifact_path=f"models/{name}"
            )
            logger.warning("Fell back to logging model as artifact only")

    def log_asset(self, file_path: str, file_name: str, **kwargs) -> None:
        """Log an asset file to MLflow.

        Args:
            file_path: Path to the asset file.
            file_name: Name for the asset.
            **kwargs: Additional arguments (unused for MLflow).
        """
        self.mlflow.log_artifact(file_path, artifact_path=f"assets/{file_name}")

    def register_model(self, model_name: str, **kwargs) -> None:
        """Register a model in MLflow's model registry."""
        try:
            # Get the current run to construct the model URI
            run_id = self.run_id or self.mlflow.active_run().info.run_id

            # Use the artifact path that was used in log_model
            # The champion model is logged with the model name as artifact path
            model_uri = f"runs:/{run_id}/{model_name}"

            # Register the model in the MLflow model registry
            registered_model = self.mlflow.register_model(model_uri, model_name)
            logger.info(
                "Successfully registered model '%s' in MLflow registry. Version: %s",
                model_name,
                registered_model.version,
            )

        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to register model '%s' in MLflow: %s", model_name, e)
            # Don't raise the exception to avoid breaking the workflow
            logger.warning(
                "MLflow model registration failed, but local model file is available for deployment"
            )

    def get_metric(self, metric_name: str) -> Optional[float]:
        """Retrieve a logged metric value from MLflow."""
        # First check cache
        if metric_name in self._metrics_cache:
            return self._metrics_cache[metric_name]

        # Otherwise fetch from MLflow
        run_id = self.run_id or self.mlflow.active_run().info.run_id
        client = self.mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        return run.data.metrics.get(metric_name)

    def end(self) -> None:
        """End the MLflow run."""
        if self.mlflow.active_run():
            self.mlflow.end_run()
