"""
Test functions for the ModelLoader class and predict function
in the inference module src/inference/utils/model.py.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.inference.utils.model import ModelLoaderManager, predict
from src.training.schemas import Config
from src.utils.path import PARENT_DIR


@pytest.fixture
def model_loader():
    return ModelLoaderManager()


def test_get_config_params(model_loader):  # pylint: disable=redefined-outer-name
    """Tests that get_config_params returns the correct configuration parameters."""

    config_yaml_abs_path = f"{str(PARENT_DIR)}/config/training-config.yml"

    config = Config(config_path=config_yaml_abs_path)
    tracker_type = config.params["train"]["experiment_tracker"]
    workspace_name = (
        config.params["train"]["workspace_name"]
        if tracker_type.lower() == "comet"
        else None
    )
    model_name = config.params["modelregistry"]["champion_model_name"]
    hf_data_source = config.params["data"]["raw_dataset_source"]

    num_col_names = config.params["data"]["num_col_names"]
    cat_col_names = config.params["data"]["cat_col_names"]

    result = model_loader.get_config_params(config_yaml_abs_path)

    assert result == (
        tracker_type,
        workspace_name,
        model_name,
        num_col_names,
        cat_col_names,
        hf_data_source,
    )


def test_load_model_comet(mocker, model_loader):  # pylint: disable=redefined-outer-name
    """Tests that load_model returns a registered model from Comet ML."""

    # Mock the API class and its methods
    mock_api = mocker.MagicMock()
    mock_api.get_registry_model_versions.return_value = ["1.0.0"]

    # Mock comet model download
    mock_model = mocker.MagicMock()
    mock_api.get_model.return_value = mock_model

    # Set the mock api to the model_loader api attribute
    model_loader.comet_api_key = "fake_key"

    # Mock joblib and its methods
    mocker.patch("joblib.load", return_value="trained_model")

    # Mock the create_model_loader to return a CometModelLoader with our mock API
    mock_comet_loader = mocker.MagicMock()
    mock_comet_loader.load_model.return_value = "trained_model"
    mocker.patch(
        "src.inference.utils.model.create_model_loader", return_value=mock_comet_loader
    )

    # Define the test parameters
    tracker_type = "comet"
    comet_workspace = "workspace"
    model_name = "model"
    artifacts_path = Path("/path/to/artifacts")

    # Call the load_model method
    result = model_loader.load_model(
        tracker_type=tracker_type,
        model_name=model_name,
        artifacts_path=artifacts_path,
        workspace_name=comet_workspace,
    )

    # Check the result
    assert result == "trained_model"


def test_load_model_mlflow(
    mocker, model_loader
):  # pylint: disable=redefined-outer-name
    """Tests that load_model returns a registered model from MLflow."""

    # Mock the MLflow model loader
    mock_mlflow_loader = mocker.MagicMock()
    mock_mlflow_loader.load_model.return_value = "trained_mlflow_model"
    mocker.patch(
        "src.inference.utils.model.create_model_loader", return_value=mock_mlflow_loader
    )

    # Define the test parameters
    tracker_type = "mlflow"
    model_name = "model"
    artifacts_path = Path("/path/to/artifacts")

    # Call the load_model method
    result = model_loader.load_model(
        tracker_type=tracker_type,
        model_name=model_name,
        artifacts_path=artifacts_path,
    )

    # Check the result
    assert result == "trained_mlflow_model"


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
        def predict_proba(
            self, data_df: pd.DataFrame
        ):  # pylint: disable=unused-argument
            return [[0.25, 0.75]]

    # Call the predict function
    result = predict(MockModel(), data)

    # Check the result
    assert result == expected_output
