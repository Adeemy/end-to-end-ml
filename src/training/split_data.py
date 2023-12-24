"""
This script extracts training data from feature store,
i.e., features and class labels. 
"""

#################################
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.getcwd())
from pathlib import PosixPath

from utils.config import Config
from utils.path import DATA_DIR

from src.feature_store.utils.prep import DataSplitter

#################################


def main(config_yaml_abs_path: str, data_dir: PosixPath):
    """Splits dataset into train and test sets."""

    print(
        """\n
    ---------------------------------------------------------------------
    --- Splitting Preprocessed Data into Train and Test Sets Starts ...
    ---------------------------------------------------------------------\n"""
    )

    # Specify required column names by data type
    config = Config(config_path=config_yaml_abs_path)
    DATASET_SPLIT_TYPE = config.params["data"]["params"]["split_type"]
    DATASET_SPLIT_SEED = config.params["data"]["params"]["split_rand_seed"]
    SPLIT_DATE_COL_NAME = config.params["data"]["params"]["split_date_col_name"]
    SPLIT_CUTOFF_DATE = config.params["data"]["params"]["train_test_split_curoff_date"]
    SPLIT_DATE_FORMAT = config.params["data"]["params"]["split_date_col_format"]
    PRIMARY_KEY = config.params["data"]["params"]["pk_col_name"]
    CLASS_COL_NAME = config.params["data"]["params"]["class_col_name"]
    date_col_names = config.params["data"]["params"]["date_col_names"]
    datetime_col_names = config.params["data"]["params"]["datetime_col_names"]
    num_col_names = config.params["data"]["params"]["num_col_names"]
    cat_col_names = config.params["data"]["params"]["cat_col_names"]

    # Check inputs
    try:
        input_data_split_seed = int(DATASET_SPLIT_SEED)
    except ValueError as e:
        raise ValueError(
            f"split_random_seed must be integer type. Got {DATASET_SPLIT_SEED}"
        ) from e

    try:
        input_split_cutoff_date = None
        if SPLIT_CUTOFF_DATE is not None:
            input_split_cutoff_date = datetime.strptime(
                SPLIT_CUTOFF_DATE, SPLIT_DATE_FORMAT
            ).date()
    except ValueError as e:
        raise ValueError(
            f"SPLIT_CUTOFF_DATE must be a date (format {SPLIT_DATE_FORMAT}) or None if split type is 'random'. Got {SPLIT_CUTOFF_DATE}"
        ) from e

    # Get prepared data from feature store
    preprocessed_data = pd.read_parquet(
        path="./src/feature_store/feature_repo/data/preprocessed_data.parquet"
    )

    # Select specified features
    required_input_col_names = (
        [PRIMARY_KEY]
        + date_col_names
        + datetime_col_names
        + num_col_names
        + cat_col_names
        + [CLASS_COL_NAME]
    )
    preprocessed_data = preprocessed_data[required_input_col_names].copy()

    print("\nTrain snd inference sets split:\n")
    data_splitter = DataSplitter(
        dataset=preprocessed_data,
        primary_key_col_name=PRIMARY_KEY,
        class_col_name=CLASS_COL_NAME,
    )

    train_set, test_set = data_splitter.split_dataset(
        split_type=DATASET_SPLIT_TYPE,
        train_set_size=0.8,
        split_random_seed=input_data_split_seed,
        split_date_col_name=SPLIT_DATE_COL_NAME,
        split_cutoff_date=input_split_cutoff_date,
        split_date_col_format=SPLIT_DATE_FORMAT,
    )

    # Store train and test sets locally
    # Note: should be registered and tagged for reproducibility.
    train_set.to_parquet(
        data_dir / "train.parquet",
        index=False,
    )

    test_set.to_parquet(
        data_dir / "test.parquet",
        index=False,
    )

    print("\nTrain and test sets were created.\n")


if __name__ == "__main__":
    main(config_yaml_abs_path=sys.argv[1], data_dir=DATA_DIR)
