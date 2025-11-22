"""
Abstract experiment tracking interface and concrete implementations.
"""

import json
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import mlflow
import numpy as np
from matplotlib.figure import Figure


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


class CometExperimentTracker(ExperimentTracker):
    """Comet ML experiment tracker implementation."""

    def __init__(self, experiment: Any) -> None:
        """Initialize with a Comet experiment instance.

        Args:
            experiment: Comet Experiment or ExistingExperiment instance.
        """
        self.experiment = experiment

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
        cm_data = {
            "matrix": matrix.tolist(),
            "title": title,
            "labels": labels if labels else [],
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
        # Note: overwrite parameter not used in MLflow as it handles versioning differently
        self.mlflow.log_artifact(str(file_or_folder), artifact_path=f"models/{name}")

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
        # Get the current run to construct the model URI
        run_id = self.run_id or self.mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        self.mlflow.register_model(model_uri, model_name)

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
