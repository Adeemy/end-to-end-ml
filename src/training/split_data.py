"""
This script extracts preprocessed data from feature store,
i.e., features and class labels, and creates data splits
for model training. 
"""

import os
import sys
from datetime import datetime

# import pandas as pd
from feast import FeatureStore
from feast.infra.offline_stores.file_source import SavedDatasetFileStorage

sys.path.insert(0, os.getcwd())
from pathlib import PosixPath

from utils.config import Config
from utils.path import DATA_DIR

from src.feature_store.utils.prep import DataSplitter

#################################


def main(feast_repo_dir: str, config_yaml_abs_path: str, data_dir: PosixPath):
    """Splits dataset into train and test sets."""

    print(
        """\n
    ---------------------------------------------------------------------
    --- Splitting Preprocessed Data into Train and Test Sets Starts ...
    ---------------------------------------------------------------------\n"""
    )

    # Specify required column names by data type
    store = FeatureStore(repo_path=feast_repo_dir)
    config = Config(config_path=config_yaml_abs_path)
    DATASET_SPLIT_TYPE = config.params["data"]["params"]["split_type"]
    DATASET_SPLIT_SEED = int(config.params["data"]["params"]["split_rand_seed"])
    SPLIT_DATE_COL_NAME = config.params["data"]["params"]["split_date_col_name"]
    SPLIT_CUTOFF_DATE = config.params["data"]["params"]["train_test_split_curoff_date"]
    SPLIT_DATE_FORMAT = config.params["data"]["params"]["split_date_col_format"]
    TRAIN_SET_SIZE = config.params["data"]["params"]["train_set_size"]
    PRIMARY_KEY = config.params["data"]["params"]["pk_col_name"]
    CLASS_COL_NAME = config.params["data"]["params"]["class_col_name"]
    date_col_names = config.params["data"]["params"]["date_col_names"]
    datetime_col_names = config.params["data"]["params"]["datetime_col_names"]
    num_col_names = config.params["data"]["params"]["num_col_names"]
    cat_col_names = config.params["data"]["params"]["cat_col_names"]
    train_set_file_name = config.params["files"]["params"]["train_set_file_name"]
    test_set_file_name = config.params["files"]["params"]["test_set_file_name"]
    input_split_cutoff_date = datetime.strptime(
        SPLIT_CUTOFF_DATE, SPLIT_DATE_FORMAT
    ).date()

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

    data_splitter = DataSplitter(
        dataset=preprocessed_data,
        primary_key_col_name=PRIMARY_KEY,
        class_col_name=CLASS_COL_NAME,
    )

    train_set, test_set = data_splitter.split_dataset(
        split_type=DATASET_SPLIT_TYPE,
        train_set_size=TRAIN_SET_SIZE,
        split_random_seed=DATASET_SPLIT_SEED,
        split_date_col_name=SPLIT_DATE_COL_NAME,
        split_cutoff_date=input_split_cutoff_date,
        split_date_col_format=SPLIT_DATE_FORMAT,
    )

    # Store train and test sets locally
    # Note: should be registered and tagged for reproducibility.
    train_set.to_parquet(
        data_dir / train_set_file_name,
        index=False,
    )

    test_set.to_parquet(
        data_dir / test_set_file_name,
        index=False,
    )

    print("\nTrain and test sets were created.\n")


if __name__ == "__main__":
    main(
        feast_repo_dir=sys.argv[1], config_yaml_abs_path=sys.argv[2], data_dir=DATA_DIR
    )
