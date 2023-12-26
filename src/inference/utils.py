"""
This utility module includes functions to download a model and its
config params for scoring prod data
"""

import sys
from pathlib import Path, PosixPath

import joblib
from comet_ml import API

sys.path.append(str(Path(__file__).parent.resolve().parent.parent))

from src.training.utils.config import Config


########################################################
def get_config_params(config_yaml_abs_path: str):
    """Extracts training and model configurations, like workspace info
    and registered champion model name."""

    # Get model settings
    config = Config(config_path=config_yaml_abs_path)
    COMET_WORKSPACE_NAME = config.params["train"]["params"]["comet_workspace_name"]
    CHAMPION_MODEL_NAME = config.params["modelregistry"]["params"][
        "champion_model_name"
    ]
    HF_DATA_SOURCE = config.params["data"]["params"]["raw_dataset_source"]

    num_col_names = config.params["data"]["params"]["num_col_names"]
    cat_col_names = config.params["data"]["params"]["cat_col_names"]

    return (
        COMET_WORKSPACE_NAME,
        CHAMPION_MODEL_NAME,
        num_col_names,
        cat_col_names,
        HF_DATA_SOURCE,
    )


def download_model(
    comet_workspace: str,
    comet_api_key: str,
    model_name: str,
    artifacts_path: PosixPath,
):
    """Downloads a registered model from Comet workspace."""

    # Create API instances
    comet_api = API(api_key=comet_api_key)

    # Download a model from Hugging Face
    # Note: log into Hugging Face CLI using huggingface-cli login --token $HUGGINGFACE_TOKEN
    _model = comet_api.get_model(
        workspace=comet_workspace,
        model_name=model_name,
    )
    model_versions = comet_api.get_registry_model_versions(
        workspace=comet_workspace,
        registry_name=model_name,
    )
    _model.download(model_versions[-1], artifacts_path, expand=True)

    return joblib.load(f"{str(artifacts_path)}/{model_name}.pkl")
