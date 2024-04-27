"""
Test functions for the ModelLoader class and predict function 
in the inference module src/inference/utils/model.py.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.inference.utils.model import ModelLoader, predict
from src.training.utils.config import Config
from src.utils.path import PARENT_DIR


@pytest.fixture
def model_loader():
    return ModelLoader()


def test_get_config_params(model_loader):
    """Tests that get_config_params returns the correct configuration parameters."""

    config_yaml_abs_path = f"{str(PARENT_DIR)}/config/training-config.yml"

    config = Config(config_path=config_yaml_abs_path)
    comet_workspace_name = config.params["train"]["comet_workspace_name"]
    model_name = config.params["modelregistry"]["champion_model_name"]
    hf_data_source = config.params["data"]["raw_dataset_source"]

    num_col_names = config.params["data"]["num_col_names"]
    cat_col_names = config.params["data"]["cat_col_names"]

    result = model_loader.get_config_params(config_yaml_abs_path)

    assert result == (
        comet_workspace_name,
        model_name,
        num_col_names,
        cat_col_names,
        hf_data_source,
    )


def test_download_model(mocker, model_loader):
    """Tests that download_model returns a registered model."""

    # Mock the API class and its methods
    mock_api = mocker.MagicMock()

    # Set the mock api to the model_loader api attribute
    model_loader.comet_api = mock_api

    # Mock joblib and its methods
    mocker.patch("joblib.load", return_value="trained_model")

    # Define the test parameters
    comet_workspace = "workspace"
    model_name = "model"
    artifacts_path = Path("/path/to/artifacts")

    # Call the download_model method
    result = model_loader.download_model(
        comet_workspace,
        model_name,
        artifacts_path,
    )

    # Check the result
    assert result == "trained_model"

    # Check that the API methods were called with the correct arguments
    mock_api.get_model.assert_called_once_with(
        workspace=comet_workspace,
        model_name=model_name,
    )
    mock_api.get_registry_model_versions.assert_called_once_with(
        workspace=comet_workspace,
        registry_name=model_name,
    )


def test_predict():
    """Tests the predict function."""

    # Define the input data
    data = {
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
    }

    # Define the expected output
    expected_output = {"Predicted Probability": 0.75}

    # Create a mock model
    class MockModel:
        def predict_proba(self, data_df: pd.DataFrame):
            return [[0.25, 0.75]]

    # Call the predict function
    result = predict(MockModel(), data)

    # Check the result
    assert result == expected_output
