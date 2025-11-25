"""
This utility module includes functions to download a model and
its config params for scoring prod data.
"""

from abc import ABC, abstractmethod
from pathlib import PosixPath
from typing import Optional

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from comet_ml import API
from fastapi import Body
from sklearn.pipeline import Pipeline

from src.training.schemas import Config


class ModelLoader(ABC):
    """Abstract base class for model loading from different tracker registries.

    Allows swapping different model loading backends (e.g., Comet ML, MLflow).
    """

    @abstractmethod
    def load_model(
        self,
        model_name: str,
        artifacts_path: Optional[PosixPath] = None,
        workspace_name: Optional[str] = None,
        stage: str = "latest",
    ) -> Pipeline:
        """Loads a model from the tracker's model registry.

        Args:
            model_name (str): registered model name.
            artifacts_path (Optional[PosixPath]): path to save model artifacts.
            workspace_name (Optional[str]): workspace name (required for some trackers).
            stage (str): model stage ("latest", "staging", "production").

        Returns:
            Pipeline: trained model.
        """
        raise NotImplementedError

    @abstractmethod
    def get_tracker_type(self) -> str:
        """Returns the tracker type identifier.

        Returns:
            str: tracker type (e.g., 'mlflow', 'comet').
        """
        raise NotImplementedError


class MLflowModelLoader(ModelLoader):
    """Loads scoring models for inference from MLflow model registry.

    Attributes:
        mlflow_tracking_uri (Optional[str]): MLflow tracking URI.
    """

    def __init__(self, mlflow_tracking_uri: Optional[str] = None) -> None:
        # Set MLflow tracking URI if provided
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)

    def get_tracker_type(self) -> str:
        """Returns 'mlflow'."""
        return "mlflow"

    def load_model(
        self,
        model_name: str,
        artifacts_path: Optional[PosixPath] = None,
        workspace_name: Optional[str] = None,
        stage: str = "latest",
    ) -> Pipeline:
        """Loads a model from MLflow registry.

        Args:
            model_name (str): registered model name.
            artifacts_path (Optional[PosixPath]): path to save model artifacts.
            workspace_name (Optional[str]): ignored for MLflow.
            stage (str): model stage ("latest", "staging", "production").

        Returns:
            Pipeline: trained model.
        """
        try:
            # Load model from MLflow registry
            model_uri = f"models:/{model_name}/{stage}"
            model = mlflow.sklearn.load_model(model_uri)
            return model
        except Exception as exe:  # pylint: disable=broad-except
            # Fallback: try to load from local artifacts if MLflow fails
            if artifacts_path and (artifacts_path / f"{model_name}.pkl").exists():
                return joblib.load(str(artifacts_path / f"{model_name}.pkl"))
            else:
                raise RuntimeError(
                    f"Could not load model '{model_name}' from MLflow registry or local artifacts"
                ) from exe


class CometModelLoader(ModelLoader):
    """Loads scoring models for inference from Comet ML model registry.

    Attributes:
        comet_api (Optional[API]): Comet ML API instance.
    """

    def __init__(self, comet_api_key: Optional[str] = None) -> None:
        # Create Comet API instance if provided
        if comet_api_key:
            self.comet_api = API(api_key=comet_api_key)
        else:
            self.comet_api = None

    def get_tracker_type(self) -> str:
        """Returns 'comet'."""
        return "comet"

    def load_model(
        self,
        model_name: str,
        artifacts_path: Optional[PosixPath] = None,
        workspace_name: Optional[str] = None,
        stage: str = "latest",
    ) -> Pipeline:
        """Loads a model from Comet ML registry.

        Args:
            model_name (str): registered model name.
            artifacts_path (Optional[PosixPath]): path to save model artifacts.
            workspace_name (Optional[str]): workspace name (required for Comet).
            stage (str): model stage (ignored for Comet, uses latest version).

        Returns:
            Pipeline: trained model.
        """
        if not self.comet_api:
            raise RuntimeError(
                "Comet API not initialized. Provide comet_api_key to CometModelLoader."
            )

        if not workspace_name:
            raise ValueError("workspace_name is required for Comet ML models")

        try:
            # Get the latest version of the model
            model_versions = self.comet_api.get_registry_model_versions(
                workspace=workspace_name,
                registry_name=model_name,
            )
            latest_version = model_versions[-1]

            # Download the latest version of the model
            model = self.comet_api.get_model(
                workspace=workspace_name,
                model_name=model_name,
            )
            model.download(latest_version, artifacts_path, expand=True)

            return joblib.load(f"{str(artifacts_path)}/{model_name}.pkl")
        except Exception as exe:  # pylint: disable=broad-except
            # Fallback: try to load from local artifacts if Comet fails
            if artifacts_path and (artifacts_path / f"{model_name}.pkl").exists():
                return joblib.load(str(artifacts_path / f"{model_name}.pkl"))
            else:
                raise RuntimeError(
                    f"Could not load model '{model_name}' from Comet registry or local artifacts"
                ) from exe


def create_model_loader(tracker_type: str, **kwargs) -> ModelLoader:
    """Factory function to create model loaders based on tracker type.

    Args:
        tracker_type (str): Type of tracker (e.g., 'comet', 'mlflow').
        **kwargs: Additional arguments to pass to the model loader constructor.

    Returns:
        ModelLoader: Model loader instance for the specified tracker.

    Raises:
        ValueError: If tracker type is not supported.
    """
    # Registry dictionary
    _model_loaders = {
        "comet": CometModelLoader,
        "mlflow": MLflowModelLoader,
    }

    tracker_type = tracker_type.lower()

    if tracker_type not in _model_loaders:
        available = ", ".join(_model_loaders.keys())
        raise ValueError(
            f"Unsupported tracker type: {tracker_type}. Available: {available}"
        )

    return _model_loaders[tracker_type](**kwargs)


class ModelLoaderManager:
    """High-level manager for loading models with automatic tracker detection.

    This class provides a unified interface for loading models from any tracker,
    automatically detecting the tracker type from configuration.
    """

    def __init__(
        self,
        comet_api_key: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
    ) -> None:
        self.comet_api_key = comet_api_key
        self.mlflow_tracking_uri = mlflow_tracking_uri

    def get_config_params(self, config_yaml_abs_path: str) -> tuple:
        """Extracts training and model configurations.

        Args:
            config_yaml_abs_path (str): path to config yaml file.

        Returns:
            tuple: Tuple containing:
                - tracker_type (str): experiment tracker type (e.g., 'mlflow' or 'comet').
                - workspace_name (str): workspace name (for Comet) or None (for MLflow).
                - model_name (str): registered model name.
                - num_col_names (list): list of numerical column names.
                - cat_col_names (list): list of categorical column names.
                - hf_data_source (str): HuggingFace dataset source.

        """

        # Get config params
        config = Config(config_path=config_yaml_abs_path)
        tracker_type = config.params["train"]["experiment_tracker"]
        model_name = config.params["modelregistry"]["champion_model_name"]
        hf_data_source = config.params["data"]["raw_dataset_source"]

        # Get workspace name for Comet, None for MLflow
        workspace_name = None
        if tracker_type.lower() == "comet":
            workspace_name = config.params["train"].get("workspace_name")

        num_col_names = config.params["data"]["num_col_names"]
        cat_col_names = config.params["data"]["cat_col_names"]

        return (
            tracker_type,
            workspace_name,
            model_name,
            num_col_names,
            cat_col_names,
            hf_data_source,
        )

    def load_model(
        self,
        tracker_type: str,
        model_name: str,
        artifacts_path: Optional[PosixPath] = None,
        workspace_name: Optional[str] = None,
        stage: str = "latest",
    ) -> Pipeline:
        """Loads a registered model from selected tracker registry.

        Args:
            tracker_type (str): experiment tracker type (e.g., 'mlflow' or 'comet').
            model_name (str): registered model name.
            artifacts_path (Optional[PosixPath]): path to save model artifacts.
            workspace_name (Optional[str]): workspace name (required for Comet).
            stage (str): model stage ("latest", "staging", "production").

        Returns:
            Pipeline: trained model.
        """

        # Create appropriate model loader based on tracker type
        if tracker_type.lower() == "mlflow":
            loader = create_model_loader(
                "mlflow", mlflow_tracking_uri=self.mlflow_tracking_uri
            )
        elif tracker_type.lower() == "comet":
            loader = create_model_loader("comet", comet_api_key=self.comet_api_key)
        else:
            raise ValueError(f"Unsupported tracker type: {tracker_type}")

        return loader.load_model(
            model_name=model_name,
            artifacts_path=artifacts_path,
            workspace_name=workspace_name,
            stage=stage,
        )


def predict(model: Pipeline, data: dict = Body(...)) -> dict:
    """Predicts the probability of a positive outcome.

    Args:
        data (dict): dictionary containing the input data.
        model (sklearn.pipeline.Pipeline): fitted model.

    Returns:
        dict: dictionary containing the predicted probability.
    """

    # Convert input dictionary to data frame required by the model
    data_df = pd.json_normalize(data)

    preds = model.predict_proba(data_df)[0]
    return {"Predicted Probability": round(preds[1], 3)}
