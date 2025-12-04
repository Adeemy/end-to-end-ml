"""
Command-line interface for batch model predictions from parquet files.

This script provides a CLI wrapper for making predictions using trained models
stored in experiment tracking backends. Requires a configuration YAML file and
input data in parquet format.

Usage:
    # Batch prediction from parquet file
    python predict.py --config_yaml_path path/to/config.yml --logger_path path/to/logging.conf --input_file data.parquet

Supports both MLflow and Comet ML model registries for loading trained models.
"""

import argparse
import logging
import os
from pathlib import Path, PosixPath

from dotenv import load_dotenv

from src.inference.utils.helpers import predict_from_file
from src.training.schemas import Config
from src.utils.logger import get_console_logger

load_dotenv()

module_name: str = PosixPath(__file__).stem
console_logger = get_console_logger(module_name)

# Required API keys for model loading
COMET_API_KEY = os.environ.get("COMET_API_KEY")


def main(
    config_yaml_path: str,
    logger: logging.Logger,
    input_file: str = None,
    output_file: str = None,
) -> None:
    """Main function for batch prediction.

    Args:
        config_yaml_path: Path to configuration YAML file
        input_file: Path to input parquet file for batch prediction
        output_file: Path to save batch predictions (optional)
        logger: Logger instance for logging
    """

    # Read inference config for default paths if not provided via CLI
    config = Config(config_path=config_yaml_path)
    inference_config = config.params.get("inference", {}).get("batch_prediction", {})

    # Use provided args if available, otherwise use config defaults
    input_file = input_file or inference_config.get("input_file")
    output_file = output_file or inference_config.get("output_file")

    # Batch prediction mode
    if not Path(input_file).exists():
        logger.error("Input file %s does not exist", input_file)
        raise FileNotFoundError(f"Input file {input_file} does not exist")

    logger.info("Starting batch prediction on file: %s", input_file)
    predict_from_file(
        config_yaml_path=config_yaml_path,
        input_file_path=input_file,
        output_file_path=output_file,
        logger=logger,
    )


if __name__ == "__main__":
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

    # Call main function with parsed arguments
    main(
        config_yaml_path=args.config_yaml_path,
        input_file=args.input_file,
        output_file=args.output_file,
        logger=console_logger,
    )
