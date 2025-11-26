"""
Command-line interface for batch model predictions from parquet files.

This script provides a CLI wrapper for making predictions using trained models
stored in experiment tracking backends. Supports both individual sample predictions
and batch processing of parquet files from the command line.

Usage:
    # Sample prediction (original behavior)
    python predict.py --config_yaml_path path/to/config.yml --logger_path path/to/logging.conf

    # Batch prediction from parquet file
    python predict.py --config_yaml_path path/to/config.yml --logger_path path/to/logging.conf --input_file data.parquet

Supports both MLflow and Comet ML model registries for loading trained models.
"""

import argparse
import logging
import os
from pathlib import Path, PosixPath

import pandas as pd
from dotenv import load_dotenv

from src.inference.utils.model import ModelLoaderManager, predict
from src.training.schemas import Config
from src.utils.logger import get_console_logger
from src.utils.path import ARTIFACTS_DIR

load_dotenv()

module_name: str = PosixPath(__file__).stem
console_logger = get_console_logger(module_name)

########################################################


def predict_from_file(
    config_yaml_path: str,
    input_file_path: str,
    output_file_path: str = None,
    logger: logging.Logger = None,
) -> pd.DataFrame:
    """
    Generate predictions for data in a parquet file.

    Args:
        config_yaml_path (str): Path to the config yaml file.
        input_file_path (str): Path to input parquet file.
        output_file_path (str, optional): Path to save predictions. If None, prints to console.
        logger (logging.Logger): Logger object.

    Returns:
        pd.DataFrame: DataFrame with input data and predictions.
    """

    # Load data from parquet file
    try:
        data_df = pd.read_parquet(input_file_path)
        logger.info("Loaded %d records from %s", len(data_df), input_file_path)
    except (FileNotFoundError, pd.errors.ParserError, OSError) as e:
        logger.error("Failed to load parquet file %s: %s", input_file_path, e)
        raise RuntimeError(f"Failed to load parquet file: {e}") from e

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

    # Generate predictions
    try:
        predictions = model.predict_proba(data_df)

        # Add predictions to the dataframe
        data_df["predicted_probability"] = [pred[1] for pred in predictions]
        data_df["prediction"] = (data_df["predicted_probability"] > 0.5).astype(int)

        logger.info("Generated predictions for %d records", len(data_df))

        # Save or display results
        if output_file_path:
            data_df.to_parquet(output_file_path, index=False)
            logger.info("Predictions saved to %s", output_file_path)
        else:
            print("\nPrediction Results (showing first 10 rows):")
            print("=" * 60)
            print(
                data_df[["predicted_probability", "prediction"]]
                .head(10)
                .to_string(index=False)
            )
            print(f"\nTotal records processed: {len(data_df)}")

        return data_df

    except (ValueError, AttributeError, KeyError) as e:
        logger.error("Prediction failed: %s", e)
        raise RuntimeError(f"Prediction failed: {e}") from e


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

    parser = argparse.ArgumentParser(
        description="Generate batch or sample predictions using trained ML models"
    )
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
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to input parquet file for batch prediction (optional).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to save batch predictions (optional, defaults to console output).",
    )

    args = parser.parse_args()

    # Read inference config for default paths if not provided via CLI
    config = Config(config_path=args.config_yaml_path)
    inference_config = config.params.get("inference", {}).get("batch_prediction", {})

    # Use CLI args if provided, otherwise use config defaults
    input_file = args.input_file or inference_config.get("input_file")
    output_file = args.output_file or inference_config.get("output_file")

    if input_file:
        # Batch prediction mode
        if not Path(input_file).exists():
            console_logger.error("Input file %s does not exist", input_file)
            raise FileNotFoundError(f"Input file {input_file} does not exist")

        console_logger.info("Starting batch prediction on file: %s", input_file)
        predict_from_file(
            config_yaml_path=args.config_yaml_path,
            input_file_path=input_file,
            output_file_path=output_file,
            logger=console_logger,
        )
    else:
        # Sample prediction mode (original behavior)
        console_logger.info("Running sample prediction with hardcoded data")
        main(
            config_yaml_path=args.config_yaml_path,
            input_data=sample_data,
            logger=console_logger,
        )
