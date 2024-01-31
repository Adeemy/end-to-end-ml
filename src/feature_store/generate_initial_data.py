"""
This code imports original dataset from UCI data repository and stores it
local path. Also, 5% of the original dataset is reserved as inference set, 
which simulates production data that will be scored by the deployed model 
in inference pipeline.

The raw dataset was released by the CDC and it was imported from the following 
UCI repo: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators.

Note that this script is only used in the beginning of this project just to
generate data for the project and it isn't part of feature or inference pipelines.
"""

import argparse
import logging
import logging.config
import sys
from pathlib import PosixPath

from ucimlrepo import fetch_ucirepo

from src.feature_store.utils.config import Config
from src.feature_store.utils.prep import DataSplitter
from src.utils.logger import LoggerWriter
from src.utils.path import DATA_DIR


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

    Returns:
        None.
    """

    logger.info(
        f"Directory of data perprocessing and transformation config file: {config_yaml_path}"
    )

    #################################
    # Import data preprocessing config params and check inputs
    config = Config(config_path=config_yaml_path)

    # Specify variable types and data source from config file
    uci_dataset_id = config.params["data"]["params"]["uci_raw_data_num"]
    inference_set_ratio = float(config.params["data"]["params"]["inference_set_ratio"])
    random_seed = int(config.params["data"]["params"]["random_seed"])
    original_split_type = config.params["data"]["params"]["original_split_type"]
    pk_col_name = config.params["data"]["params"]["pk_col_name"]
    class_column_name = config.params["data"]["params"]["class_col_name"]
    date_col_names = config.params["data"]["params"]["date_col_names"]
    datetime_col_names = config.params["data"]["params"]["datetime_col_names"]
    num_col_names = config.params["data"]["params"]["num_col_names"]
    cat_col_names = config.params["data"]["params"]["cat_col_names"]
    raw_dataset_file_name = config.params["files"]["params"]["raw_dataset_file_name"]
    inference_set_file_name = config.params["files"]["params"][
        "inference_set_file_name"
    ]

    #################################
    # Import raw data
    required_input_col_names = (
        [pk_col_name]
        + date_col_names
        + datetime_col_names
        + num_col_names
        + cat_col_names
        + [class_column_name]
    )
    raw_data = fetch_ucirepo(id=uci_dataset_id)
    raw_dataset = raw_data.data.features.copy()
    raw_dataset[pk_col_name] = raw_data.data.ids.loc[:, [pk_col_name]]
    raw_dataset[class_column_name] = raw_data.data.targets.loc[:, [class_column_name]]

    # Select relevant columns by removing irrelevant or erroneous columns (if any)
    raw_dataset = raw_dataset[required_input_col_names]

    #################################

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

    # Save data splits in feature_repo before uploading
    # them to Hugging Face (Bena345/cdc-diabetes-health-indicators)
    raw_dataset.to_parquet(data_dir / raw_dataset_file_name, index=False)
    inference_set.to_parquet(
        data_dir / inference_set_file_name,
        index=False,
    )

    logger.info("Inference and raw dataset (for model development) were generated.")
    logger.info(
        f"""Ratio of raw dataset out of original dataset: 
                {'{:0.1f}'.format(100 * (1-inference_set_ratio))}% ({raw_dataset.shape[0]} rows)."""
    )
    logger.info(
        f"""Ratio of inference set out of original dataset:
                 {'{:0.1f}'.format(100 * inference_set_ratio)}% ({inference_set.shape[0]} rows)."""
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

    # Load the configuration file
    logging.config.fileConfig(args.logger_path)

    # Get the logger objects by name
    console_logger = logging.getLogger("console_logger")
    print_logger = logging.getLogger("print_logger")

    # Create a LoggerWriter object using the console logger and the print logger
    writer = LoggerWriter(console_logger, print_logger)

    # Redirect sys.stdout to the LoggerWriter object
    sys.stdout = writer

    console_logger.info("Generating Raw Dataset Starts ...")

    main(
        config_yaml_path=args.config_yaml_path, data_dir=DATA_DIR, logger=console_logger
    )
