"""
Imports original dataset from UCI data repository and stores it local path.
Also, 5% of the original dataset is reserved as inference set, which simulates 
production data that will be scored by the deployed model in inference pipeline.

The raw dataset was released by the CDC and it was imported from the following 
UCI repo: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators.

Note: this script is only used in the beginning of this project just to generate
data for the project and it isn't part of feature or inference pipelines.
"""

import argparse
import logging
import logging.config
from pathlib import PosixPath

import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.feature_store.utils.config import Config
from src.feature_store.utils.prep import DataSplitter
from src.utils.logger import get_console_logger
from src.utils.path import DATA_DIR


def import_data(config_yaml_path: str) -> pd.DataFrame:
    """Imports raw data from UCI data repository.

    Args:
        config_yaml_path (str): path to the config yaml file.

    Returns:
        raw_dataset (pd.DataFrame): raw dataset.
    """

    # Get configuration parameters
    config = Config(config_path=config_yaml_path)
    uci_dataset_id = config.params["data"]["uci_raw_data_num"]
    pk_col_name = config.params["data"]["pk_col_name"]
    class_column_name = config.params["data"]["class_col_name"]
    date_col_names = config.params["data"]["date_col_names"]
    datetime_col_names = config.params["data"]["datetime_col_names"]
    num_col_names = config.params["data"]["num_col_names"]
    cat_col_names = config.params["data"]["cat_col_names"]

    # Import raw data
    raw_data = fetch_ucirepo(id=uci_dataset_id)
    raw_dataset = raw_data.data.features.copy()
    raw_dataset[pk_col_name] = raw_data.data.ids.loc[:, [pk_col_name]]
    raw_dataset[class_column_name] = raw_data.data.targets.loc[:, [class_column_name]]

    # Select relevant columns
    required_input_col_names = (
        [pk_col_name]
        + date_col_names
        + datetime_col_names
        + num_col_names
        + cat_col_names
        + [class_column_name]
    )
    raw_dataset = raw_dataset[required_input_col_names]

    return raw_dataset


def split_data(
    raw_dataset: pd.DataFrame, config_yaml_path: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits raw dataset into training and inference sets.

    Args:
        raw_dataset (pd.DataFrame): raw dataset.
        config_yaml_path (str): path to the config yaml file.

    Returns:
        raw_dataset (pd.DataFrame): raw dataset for model development.
        inference_set (pd.DataFrame): inference set for production data simulation.
    """

    # Get configuration parameters
    config = Config(config_path=config_yaml_path)
    inference_set_ratio = float(config.params["data"]["inference_set_ratio"])
    random_seed = int(config.params["data"]["random_seed"])
    original_split_type = config.params["data"]["original_split_type"]
    pk_col_name = config.params["data"]["pk_col_name"]
    class_column_name = config.params["data"]["class_col_name"]

    # Create inference set from raw dataset to simulate production data
    train_valid_splitter = DataSplitter(
        dataset=raw_dataset,
        primary_key_col_name=pk_col_name,
        class_col_name=class_column_name,
    )

    raw_dataset, inference_set = train_valid_splitter.split_dataset(
        split_type=original_split_type,
        train_set_size=1 - inference_set_ratio,
        split_random_seed=random_seed,
    )

    return raw_dataset, inference_set


def main(
    config_yaml_path: str,
    data_dir: PosixPath,
    logger: logging.Logger,
) -> None:
    """Imports raw dataset from UCI data repository and creates training data and
    inference set.

    Args:
        config_yaml_path (str): path to the config yaml file.
        data_dir (PosixPath): path to the data directory.
        logger (logging.Logger): logger object.
    """

    logger.info(
        "Directory of data perprocessing and transformation config file: %s",
        config_yaml_path,
    )

    # Get configuration parameters
    config = Config(config_path=config_yaml_path)
    inference_set_ratio = float(config.params["data"]["inference_set_ratio"])
    raw_dataset_file_name = config.params["files"]["raw_dataset_file_name"]
    inference_set_file_name = config.params["files"]["inference_set_file_name"]

    #################################
    # Import raw dataset
    raw_dataset = import_data(config_yaml_path=config_yaml_path)

    # Split raw dataset into training and inference sets
    raw_dataset, inference_set = split_data(
        raw_dataset=raw_dataset, config_yaml_path=config_yaml_path
    )

    # Save data splits in feature_repo before uploading
    # them to Hugging Face (Bena345/cdc-diabetes-health-indicators)
    raw_dataset.to_parquet(data_dir / raw_dataset_file_name, index=False)
    inference_set.to_parquet(
        data_dir / inference_set_file_name,
        index=False,
    )

    logger.info("Inference and raw dataset (for model development) generated.")
    logger.info(
        "Ratio of raw dataset out of original dataset: " + "%.1f%% (%d rows).",
        100 * (1 - inference_set_ratio),
        raw_dataset.shape[0],
    )
    logger.info(
        "Ratio of inference set out of original dataset: " + "%.1f%% (%d rows).",
        100 * inference_set_ratio,
        inference_set.shape[0],
    )


###########################################################
if __name__ == "__main__":
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
    console_logger = get_console_logger("gen_initial_data_logger")

    console_logger.info("Generating Raw Dataset Starts ...")

    main(
        config_yaml_path=args.config_yaml_path, data_dir=DATA_DIR, logger=console_logger
    )
