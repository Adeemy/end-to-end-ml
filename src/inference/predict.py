"""
Command-line interface for model predictions on individual data samples.

This script provides a CLI wrapper for making predictions using trained models
stored in experiment tracking backends. Unlike the FastAPI service, this is
designed for batch processing or individual prediction requests from the
command line.

Usage:
    python predict.py --config_yaml_path path/to/config.yml --input_data '{...}'

Supports both MLflow and Comet ML model registries for loading trained models.
"""

import argparse
import logging
import os

from dotenv import load_dotenv

from src.inference.utils.model import ModelLoaderManager, predict
from src.utils.logger import get_console_logger
from src.utils.path import ARTIFACTS_DIR

load_dotenv()

########################################################


def main(config_yaml_path: str, input_data: dict, logger: logging.Logger) -> None:
    """Loads the champion model and scores the input data.

    Args:
        config_yaml_path (str): path to the config yaml file.
        input_data (dict): dictionary containing the input data.
        logger (logging.Logger): logger object.
    """

    # Initialize ModelLoader with environment variables
    comet_api_key = os.environ.get("COMET_API_KEY")
    load_model = ModelLoaderManager(comet_api_key=comet_api_key)

    # Extract config params including tracker type
    (
        tracker_type,
        workspace_name,
        champ_model_name,
        *_,
    ) = load_model.get_config_params(config_yaml_abs_path=config_yaml_path)

    # Load champion model from the appropriate registry
    model = load_model.load_model(
        tracker_type=tracker_type,
        model_name=champ_model_name,
        artifacts_path=ARTIFACTS_DIR,
        workspace_name=workspace_name,
    )

    prediction = predict(model, input_data)
    logger.info("\n\n\n%s\n\n\n", prediction)


if __name__ == "__main__":
    # Sample of prod data for testing
    sample_data = {
        "BMI": 29.0,
        "PhysHlth": 0,
        "Age": "65 to 69",
        "HighBP": "0",
        "HighChol": "1",
        "CholCheck": "0",
        "Smoker": "1",
        "Stroke": "1",
        "HeartDiseaseorAttack": "0",
        "PhysActivity": "1",
        "Fruits": "1",
        "Veggies": "1",
        "HvyAlcoholConsump": "1",
        "AnyHealthcare": "1",
        "NoDocbcCost": "1",
        "GenHlth": "Poor",
        "MentHlth": "1",
        "DiffWalk": "1",
        "Sex": "1",
        "Education": "1",
        "Income": "7",
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml_path",
        type=str,
        default="./config.yml",
        help="Path to the configuration yaml file.",
    )
    parser.add_argument(
        "--logger_path",
        type=str,
        default="./logger.conf",
        help="Path to the logger configuration file.",
    )

    args = parser.parse_args()

    # Get the logger objects by name
    console_logger = get_console_logger("data_logger")

    main(
        config_yaml_path=args.config_yaml_path,
        input_data=sample_data,
        logger=console_logger,
    )
