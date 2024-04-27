"""
Extracts preprocessed data from feature store,i.e., features and
class labels, and creates data splits for model training. 
"""

import argparse
import logging
import logging.config
from datetime import datetime
from pathlib import PosixPath

import pandas as pd
from feast import FeatureStore
from feast.infra.offline_stores.file_source import SavedDatasetFileStorage

from src.feature_store.utils.prep import DataSplitter
from src.training.utils.config import Config
from src.training.utils.data import TrainingDataPrep
from src.utils.logger import get_console_logger
from src.utils.path import DATA_DIR, FEATURE_REPO_DIR

#################################


def main(
    feast_repo_dir: str,
    config_yaml_path: str,
    data_dir: PosixPath,
    logger: logging.Logger,
) -> None:
    """Splits dataset into train and test sets.

    Args:
        feast_repo_dir (str): path to the feature store repo.
        config_yaml_path (str): path to the config yaml file.
        data_dir (PosixPath): path to the data directory.

    Returns:
        None.
    """

    logger.info(f"Directory of training config file: {config_yaml_path}")

    feat_store = FeatureStore(repo_path=str(feast_repo_dir))
    config = Config(config_path=config_yaml_path)
    pk_col_name = config.params["data"]["pk_col_name"]
    class_column_name = config.params["data"]["class_col_name"]
    date_col_names = config.params["data"]["date_col_names"]
    datetime_col_names = config.params["data"]["datetime_col_names"]
    num_col_names = config.params["data"]["num_col_names"]
    cat_col_names = config.params["data"]["cat_col_names"]
    historical_features = config.params["data"]["historical_features"]

    preprocessed_dataset_target_file_name = config.params["files"][
        "preprocessed_dataset_target_file_name"
    ]
    historical_data_file_name = config.params["files"]["historical_data_file_name"]
    train_set_file_name = config.params["files"]["train_set_file_name"]
    valid_set_file_name = config.params["files"]["valid_set_file_name"]
    test_set_file_name = config.params["files"]["test_set_file_name"]
    dataset_split_type = config.params["data"]["split_type"]
    split_rand_seed = int(config.params["data"]["split_rand_seed"])
    train_set_ratio = config.params["data"]["train_set_size"]
    dataset_split_date_col_name = config.params["data"]["split_date_col_name"]
    train_valid_split_curoff_date = config.params["data"][
        "train_valid_split_curoff_date"
    ]
    dataset_split_date_col_format = config.params["data"]["split_date_col_format"]

    # Extract cut-off date for splitting train and test sets
    input_split_cutoff_date = None
    if dataset_split_type == "time":
        input_split_cutoff_date = datetime.strptime(
            train_valid_split_curoff_date, dataset_split_date_col_format
        ).date()

    # Get historical features and join them with target
    # Note: this join will take into account even_timestamp such that
    # a target value is joined with the latest feature values prior to
    # event_timestamp of the target. This ensures that class labels of
    # an event is attributed to the correct feature values.
    target_data = pd.read_parquet(path=data_dir / preprocessed_dataset_target_file_name)
    historical_data = feat_store.get_historical_features(
        entity_df=target_data,
        features=historical_features,
    )

    # Retrieve historical dataset into a dataframe
    # Note: this saves exact version of data used to train model for reproducibility.
    preprocessed_data = feat_store.create_saved_dataset(
        from_=historical_data,
        name="historical_data",
        storage=SavedDatasetFileStorage(f"{str(data_dir)}/{historical_data_file_name}"),
        allow_overwrite=True,
    ).to_df()

    # Select specified features
    required_input_col_names = (
        [pk_col_name]
        + date_col_names
        + datetime_col_names
        + num_col_names
        + cat_col_names
        + [class_column_name]
    )
    preprocessed_data = preprocessed_data[required_input_col_names].copy()

    # Split data into train and test sets
    data_splitter = DataSplitter(
        dataset=preprocessed_data,
        primary_key_col_name=pk_col_name,
        class_col_name=class_column_name,
    )

    training_set, testing_set = data_splitter.split_dataset(
        split_type=dataset_split_type,
        train_set_size=train_set_ratio,
        split_random_seed=split_rand_seed,
        split_date_col_name=dataset_split_date_col_name,
        split_cutoff_date=input_split_cutoff_date,
        split_date_col_format=dataset_split_date_col_format,
    )

    # Prepare data for training
    data_prep = TrainingDataPrep(
        train_set=training_set,
        test_set=testing_set,
        primary_key=pk_col_name,
        class_col_name=class_column_name,
        numerical_feature_names=num_col_names,
        categorical_feature_names=cat_col_names,
    )

    # Preprocess train and test sets by enforcing data types of numerical and categorical features
    data_prep.select_relevant_columns()
    data_prep.enforce_data_types()
    data_prep.create_validation_set(
        split_type=dataset_split_type,
        train_set_size=train_set_ratio,
        split_random_seed=split_rand_seed,
        split_date_col_name=dataset_split_date_col_name,
        split_cutoff_date=train_valid_split_curoff_date,
        split_date_col_format=dataset_split_date_col_format,
    )

    # Store train, validation, and test sets locally
    # Note: should be registered and tagged for reproducibility.
    train_set = data_prep.train_set
    train_set.to_parquet(
        data_dir / train_set_file_name,
        index=False,
    )

    valid_set = data_prep.valid_set
    valid_set.to_parquet(
        data_dir / valid_set_file_name,
        index=False,
    )

    test_set = data_prep.test_set
    test_set.to_parquet(
        data_dir / test_set_file_name,
        index=False,
    )

    logger.info("Train and test sets created.")


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
    console_logger = get_console_logger("split_data_logger")

    console_logger.info(
        "Splitting Preprocessed Data into Train and Test Sets Starts ..."
    )

    main(
        config_yaml_path=args.config_yaml_path,
        feast_repo_dir=FEATURE_REPO_DIR,
        data_dir=DATA_DIR,
        logger=console_logger,
    )
