"""
Imports original dataset from UCI data repository and stores it locally.
Also, 5% of the original dataset is reserved as an inference set, which simulates
production data that will be scored by the deployed model in the inference pipeline.

The raw dataset was released by the CDC and imported from the following
UCI repo: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators.

Note: This script is only used at the beginning of this project to generate
data for the project and is not part of feature or inference pipelines.
"""

import argparse
import logging
from pathlib import PosixPath
from typing import Tuple

import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.feature_store.utils.config import (
    Config,
    DataConfig,
    FilesConfig,
    build_feature_store_config,
)
from src.feature_store.utils.prep import (
    DataSplitter,
    RandomSplitStrategy,
    TimeBasedSplitStrategy,
)
from src.utils.config_loader import load_config
from src.utils.logger import get_console_logger
from src.utils.path import DATA_DIR


def import_data(data_config: DataConfig) -> pd.DataFrame:
    """Import data from UCI repository.

    Args:
        data_config: Data configuration parameters.

    Returns:
        pd.DataFrame: Raw dataset.
    """
    raw_data = fetch_ucirepo(id=data_config.uci_raw_data_num)

    # Extract features, ids, and targets
    raw_dataset = raw_data.data.features.copy()
    raw_dataset[data_config.pk_col_name] = raw_data.data.ids.loc[
        :, [data_config.pk_col_name]
    ]
    raw_dataset[data_config.class_col_name] = raw_data.data.targets.loc[
        :, [data_config.class_col_name]
    ]

    # Select required columns
    required_columns = (
        [data_config.pk_col_name]
        + data_config.date_col_names
        + data_config.datetime_col_names
        + data_config.num_col_names
        + data_config.cat_col_names
        + [data_config.class_col_name]
    )

    return raw_dataset[required_columns]


def split_data(
    raw_dataset: pd.DataFrame, data_config: DataConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits raw dataset into training and inference sets.

    Args:
        raw_dataset: Raw dataset.
        data_config: Data configuration parameters.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Training and inference datasets.

    Raises:
        ValueError: If an unsupported split type is provided.
    """
    splitter = DataSplitter(
        dataset=raw_dataset,
        primary_key_col_name=data_config.pk_col_name,
        class_col_name=data_config.class_col_name,
    )

    # Select split strategy
    if data_config.original_split_type == "random":
        split_strategy = RandomSplitStrategy(
            class_col_name=data_config.class_col_name,
            train_set_size=1 - data_config.inference_set_ratio,
            random_seed=data_config.random_seed,
        )
    elif data_config.original_split_type == "time":
        split_strategy = TimeBasedSplitStrategy(
            date_col_name=data_config.event_timestamp_col_name,
            cutoff_date=data_config.split_cutoff_date,
            date_format=data_config.split_date_col_format,
        )
    else:
        raise ValueError(f"Unsupported split type: {data_config.original_split_type}")

    return splitter.split_dataset(split_strategy)


def save_datasets(
    raw_dataset: pd.DataFrame,
    inference_set: pd.DataFrame,
    files_config: FilesConfig,
    data_dir: PosixPath,
) -> None:
    """Save datasets to disk.

    Args:
        raw_dataset: Raw dataset for model development.
        inference_set: Inference dataset.
        files_config: File configuration parameters.
        data_dir: Path to the data directory.
    """
    raw_dataset.to_parquet(data_dir / files_config.raw_dataset_file_name, index=False)
    inference_set.to_parquet(
        data_dir / files_config.inference_set_file_name, index=False
    )


def log_dataset_info(
    logger: logging.Logger,
    raw_dataset: pd.DataFrame,
    inference_set: pd.DataFrame,
    inference_set_ratio: float,
) -> None:
    """Log information about the datasets.

    Args:
        logger: Logger object.
        raw_dataset: Raw dataset.
        inference_set: Inference dataset.
        inference_set_ratio: Ratio of the inference set.
    """
    logger.info("Inference and raw dataset (for model development) generated.")
    logger.info(
        "Ratio of raw dataset out of original dataset: %.1f%% (%d rows).",
        100 * (1 - inference_set_ratio),
        raw_dataset.shape[0],
    )
    logger.info(
        "Ratio of inference set out of original raw dataset: %.1f%% (%d rows).",
        100 * inference_set_ratio,
        inference_set.shape[0],
    )


def main(config_yaml_path: str, data_dir: PosixPath, logger: logging.Logger) -> None:
    """Main function to import raw dataset and create training and inference sets.

    Args:
        config_yaml_path: Path to the configuration YAML file.
        data_dir: Path to the data directory.
        logger: Logger object.
    """

    config = load_config(
        config_class=Config,
        builder_func=build_feature_store_config,
        config_path=config_yaml_path,
    )

    raw_dataset = import_data(config.data)
    logger.info("Raw dataset imported from UCI repository.")

    raw_dataset, inference_set = split_data(raw_dataset, config.data)
    logger.info("Raw dataset was split into training and inference sets.")

    save_datasets(raw_dataset, inference_set, config.files, data_dir)
    logger.info("Datasets saved to disk.")

    log_dataset_info(
        logger, raw_dataset, inference_set, config.data.inference_set_ratio
    )


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

    # Get the console logger object
    module_name: str = PosixPath(__file__).stem

    console_logger = get_console_logger(module_name)
    console_logger.info("Generating raw dataset starts ...")

    main(
        config_yaml_path=args.config_yaml_path, data_dir=DATA_DIR, logger=console_logger
    )
