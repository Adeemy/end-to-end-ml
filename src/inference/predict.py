"""
This script wraps an API endpoint around a model to score
production data via API calls.
"""

import argparse
import logging
import logging.config
import os
import sys

from dotenv import load_dotenv

from src.inference.utils.model import ModelLoader, predict
from src.utils.logger import LoggerWriter
from src.utils.path import ARTIFACTS_DIR

load_dotenv()

########################################################


def main(
    config_yaml_path: str, api_key: str, input_data: dict, logger: logging.Logger
) -> None:
    """Loads the champion model and scores the input data.

    Args:
        config_yaml_path (str): path to the config yaml file.
        api_key (str): Comet API key.
        input_data (dict): dictionary containing the input data.

    Returns:
        None.
    """

    # Extracts config params
    load_model = ModelLoader(comet_api_key=api_key)
    (
        comet_ws,
        champ_model_name,
        *_,
    ) = load_model.get_config_params(config_yaml_abs_path=config_yaml_path)

    # Download champion model
    model = load_model.download_model(
        comet_workspace=comet_ws,
        model_name=champ_model_name,
        artifacts_path=ARTIFACTS_DIR,
    )

    prediction = predict(model, input_data)
    logger.info(f"\n\n\n{prediction=}\n\n\n")


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

    # Load the configuration file
    try:
        logging.config.fileConfig(args.logger_path)
    except KeyError as e:
        raise KeyError(
            f"Failed to load logger configuration file: {args.logger_path}"
        ) from e

    # Get the logger objects by name
    console_logger = logging.getLogger("console_logger")
    print_logger = logging.getLogger("print_logger")

    # Create a LoggerWriter object using the console logger and the print logger
    writer = LoggerWriter(console_logger, print_logger)

    # Redirect sys.stdout to the LoggerWriter object
    sys.stdout = writer

    main(
        config_yaml_path=args.config_yaml_path,
        api_key=os.environ["COMET_API_KEY"],
        input_data=sample_data,
        logger=console_logger,
    )
