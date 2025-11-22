"""
This utility module includes functions to download a model and
its config params for scoring prod data.
"""

from pathlib import PosixPath

import joblib
import pandas as pd
from comet_ml import API
from fastapi import Body
from sklearn.pipeline import Pipeline

from src.training.utils.config.config import Config


class ModelLoader:
    """Loads scoring models for inference. It creates Comet API instance
    if comet_api is not provided and comet_api_key is provided.

    Attributes:
        comet_api (comet_ml.API): Comet API instance needed to download models.
    """

    def __init__(self, comet_api_key: str = None) -> None:
        # Create Comet API instance if comet_api is not provided
        # Note: comet_api_key was made an attribute to make easier to mock
        # it during unit testing download_model method compared to creating
        # it inside the method.
        if comet_api_key is not None:
            self.comet_api = API(api_key=comet_api_key)

    def get_config_params(self, config_yaml_abs_path: str) -> tuple:
        """Extracts training and model configurations, like workspace info
        and registered model name.

        Args:
            config_yaml_abs_path (str): path to config yaml file.

        Returns:
            tuple: Tuple containing:
                - workspace_name (str): Comet workspace name.
                - model_name (str): registered model name.
                - num_col_names (list): list of numerical column names.
                - cat_col_names (list): list of categorical column names.
                - hf_data_source (str): HuggingFace dataset source.

        """

        # Get config params
        config = Config(config_path=config_yaml_abs_path)
        workspace_name = config.params["train"]["workspace_name"]
        model_name = config.params["modelregistry"]["champion_model_name"]
        hf_data_source = config.params["data"]["raw_dataset_source"]

        num_col_names = config.params["data"]["num_col_names"]
        cat_col_names = config.params["data"]["cat_col_names"]

        return (
            workspace_name,
            model_name,
            num_col_names,
            cat_col_names,
            hf_data_source,
        )

    def download_model(
        self,
        comet_workspace: str,
        model_name: str,
        artifacts_path: PosixPath,
    ) -> Pipeline:
        """Downloads a registered model from Comet workspace.

        Args:
            comet_workspace (str): Comet workspace name.
            model_name (str): registered model name.
            artifacts_path (PosixPath): path to save model artifacts.

        Returns:
            model (sklearn.pipeline.Pipeline): trained model.
        """

        # Get the latest version of the model
        model_versions = self.comet_api.get_registry_model_versions(
            workspace=comet_workspace,
            registry_name=model_name,
        )
        latest_version = model_versions[-1]

        # Download the latest version of the model
        model = self.comet_api.get_model(
            workspace=comet_workspace,
            model_name=model_name,
        )
        model.download(latest_version, artifacts_path, expand=True)

        return joblib.load(f"{str(artifacts_path)}/{model_name}.pkl")


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
