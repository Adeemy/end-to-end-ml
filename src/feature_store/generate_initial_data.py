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

import sys
from pathlib import PosixPath

from ucimlrepo import fetch_ucirepo

from src.config.path import DATA_DIR
from src.feature_store.utils.config import Config
from src.feature_store.utils.prep import DataSplitter


def main(
    config_yaml_abs_path: str,
    data_dir: PosixPath,
):
    """Imports original dataset from remote source and generates the following datasets:

    raw dataset: this dataset is meant to be preprocessed, transformed, and stored in
    feature store. Think of this dataset as it's sourced from hospital EHR system.
    inference: production data used to simulate inference data scored by deployed model.

    config_yaml_abs_path (str): absolute path to config.yml file, which
        includes dataset preprocessing configuration.
    usci_dataset_id (int): UCI repo ID of dataset.
    split_random_seed (int): seed for random number generator for random split.

    Returns:
        Saves raw dataset and inference set in local path.
    """

    print(
        """\n
    ----------------------------------------------------------------
    --- Generating Initial Dataset Starts ...
    ----------------------------------------------------------------\n"""
    )

    #################################
    # Import data preprocessing config params and check inputs
    config = Config(config_path=config_yaml_abs_path)

    # Specify variable types and data source from config file
    uci_dataset_id = config.params["data"]["params"]["uci_raw_data_num"]
    PRIMARY_KEY = config.params["data"]["params"]["pk_col_name"]
    CLASS_COL_NAME = config.params["data"]["params"]["class_col_name"]
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
        [PRIMARY_KEY]
        + date_col_names
        + datetime_col_names
        + num_col_names
        + cat_col_names
        + [CLASS_COL_NAME]
    )
    raw_data = fetch_ucirepo(id=uci_dataset_id)
    raw_dataset = raw_data.data.features.copy()
    raw_dataset[PRIMARY_KEY] = raw_data.data.ids.loc[:, [PRIMARY_KEY]]
    raw_dataset[CLASS_COL_NAME] = raw_data.data.targets.loc[:, [CLASS_COL_NAME]]

    # Select relevant columns by removing irrelevant or erroneous columns (if any)
    raw_dataset = raw_dataset[required_input_col_names]

    #################################

    # Create inference set from raw dataset to simulate production data
    train_valid_splitter = DataSplitter(
        dataset=raw_dataset,
        primary_key_col_name=PRIMARY_KEY,
        class_col_name=CLASS_COL_NAME,
    )

    raw_dataset, inference_set = train_valid_splitter.split_dataset(
        split_type="random",
        train_set_size=0.95,
        split_random_seed=123,
    )

    # Save data splits in feature_repo before uploading
    # them to Hugging Face (Bena345/cdc-diabetes-health-indicators)
    raw_dataset.to_parquet(data_dir / raw_dataset_file_name, index=False)
    inference_set.to_parquet(
        data_dir / inference_set_file_name,
        index=False,
    )

    print("\nRaw dataset was generated.\n")


###########################################################
if __name__ == "__main__":
    main(
        config_yaml_abs_path=sys.argv[1],
        data_dir=DATA_DIR,
    )
