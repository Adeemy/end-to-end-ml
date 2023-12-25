"""
This code splits raw data into three portions:
1. train set (75%),
2. test set (20%),
3. inference set (5%), a data split that serves as a production
   data to be scored by the deployed model in inference pipeline.
 
This script is only used in the beginning of this project and isn't part
of feature or inference pipelines. The raw dataset is sourced from: 

Data source: this dataset was released by the CDC and it was imported
from UCI repo (https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators).

The three data splits are then stored in Hugging Face Dataset. This dataset 
was selected for the project as it's real-life dataset and represents a 
meaningful application.
"""

#################################
import os
import sys
from typing import Literal

from ucimlrepo import fetch_ucirepo

sys.path.insert(0, os.getcwd() + "/src/")
from pathlib import PosixPath

from feature_store.utils.config import Config
from feature_store.utils.prep import DataSplitter
from training.utils.path import DATA_DIR


def main(
    config_yaml_abs_path: str,
    data_dir: PosixPath,
    data_split_type: Literal["time", "random"] = "random",
    split_random_seed: int = 123,
):
    """Imports raw dataset from remote source and generates three
    splits:
    train: training set, which can be further split into train and validation sets,
    test: testing set used to assesss the generalization capability of a model.
    inference: production data used to simulate inference data scored by deployed model.

    config_yaml_abs_path (str): absolute path to config.yml file, which
        includes dataset preprocessing configuration.
    usci_dataset_id (int): UCI repo ID of dataset.
    split_random_seed (int): seed for random number generator for random split.

    Returns:
        Saves the three splits in local path.
    """

    print(
        """\n
    ----------------------------------------------------------------
    --- Generate Initial Dataset Starts ...
    ----------------------------------------------------------------\n"""
    )

    #################################
    # Import data preprocessing config params and check inputs
    config = Config(config_path=config_yaml_abs_path)

    # Specify variable types and data source from config file
    usci_dataset_id = config.params["data"]["params"]["usci_raw_data_num"]
    PRIMARY_KEY = config.params["data"]["params"]["pk_col_name"]
    CLASS_COL_NAME = config.params["data"]["params"]["class_col_name"]
    date_col_names = config.params["data"]["params"]["date_col_names"]
    datetime_col_names = config.params["data"]["params"]["datetime_col_names"]
    num_col_names = config.params["data"]["params"]["num_col_names"]
    cat_col_names = config.params["data"]["params"]["cat_col_names"]

    try:
        usci_dataset_id = int(usci_dataset_id)
    except ValueError as e:
        raise ValueError(
            f"usci_dataset_id must be integer type. Got {usci_dataset_id}"
        ) from e

    try:
        split_random_seed = int(split_random_seed)
    except ValueError as e:
        raise ValueError(
            f"split_random_seed must be integer type. Got {split_random_seed}"
        ) from e

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
    raw_data = fetch_ucirepo(id=usci_dataset_id)
    raw_dataset = raw_data.data.features.copy()
    raw_dataset[PRIMARY_KEY] = raw_data.data.ids.loc[:, [PRIMARY_KEY]]
    raw_dataset[CLASS_COL_NAME] = raw_data.data.targets.loc[:, [CLASS_COL_NAME]]

    # Select relevant columns by removing irrelevant or erroneous columns (if any)
    raw_dataset = raw_dataset[required_input_col_names]

    #################################
    # Split data into train and testing set
    raw_data_splitter = DataSplitter(
        dataset=raw_dataset,
        primary_key_col_name=PRIMARY_KEY,
        class_col_name=CLASS_COL_NAME,
    )

    print("\nTrain snd test sets split:\n")
    train_set, test_set = raw_data_splitter.split_dataset(
        split_type=data_split_type,
        train_set_size=0.8,
        split_random_seed=split_random_seed,
    )

    # Create inference set from train split to simulate production data
    train_valid_splitter = DataSplitter(
        dataset=train_set,
        primary_key_col_name=PRIMARY_KEY,
        class_col_name=CLASS_COL_NAME,
    )

    print("\nTrain snd inference sets split:\n")
    train_set, inference_set = train_valid_splitter.split_dataset(
        split_type=data_split_type,
        train_set_size=0.937,
        split_random_seed=split_random_seed,
    )

    # Drop class labels from inference set to simulate production data
    # used at inference time
    inference_set.drop(CLASS_COL_NAME, axis=1, inplace=True)

    # Save data splits in feature_repo before uploading
    # them to Hugging Face (Bena345/cdc-diabetes-health-indicators)
    train_set.to_parquet(data_dir / "train.parquet", index=False)
    test_set.to_parquet(data_dir / "test.parquet", index=False)
    inference_set.to_parquet(
        data_dir / "inference.parquet",
        index=False,
    )

    print("\nInitial dataset was generated.\n")


# python ./src/feature_store/initial_data_setup/generate_initial_data.py ./config/feature_store/config.yml 891 random 123
if __name__ == "__main__":
    main(
        config_yaml_abs_path=sys.argv[1],
        data_dir=DATA_DIR,
        data_split_type=sys.argv[2],
        split_random_seed=sys.argv[3],
    )
