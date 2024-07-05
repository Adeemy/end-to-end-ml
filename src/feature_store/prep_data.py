"""
Preprocesses and transforms raw data before saving it in the feature store.
"""

import argparse
import logging
import logging.config
from datetime import datetime
from pathlib import PosixPath

import pandas as pd

from src.feature_store.utils.config import Config
from src.feature_store.utils.prep import DataPreprocessor, DataTransformer
from src.utils.logger import get_console_logger
from src.utils.path import DATA_DIR


def import_data(config_yaml_path: str, data_dir: PosixPath) -> pd.DataFrame:
    """Imports raw data to be preprocessed and transformed.

    Args:
        config_yaml_path (str): path to the config yaml file.
        data_dir (PosixPath): path to the data directory.

    Returns:
        pd.DataFrame: raw dataset.
    """

    # Get configuration parameters
    config = Config(config_path=config_yaml_path)
    pk_col_name = config.params["data"]["pk_col_name"]
    class_column_name = config.params["data"]["class_col_name"]
    date_col_names = config.params["data"]["date_col_names"]
    datetime_col_names = config.params["data"]["datetime_col_names"]
    num_col_names = config.params["data"]["num_col_names"]
    cat_col_names = config.params["data"]["cat_col_names"]
    raw_dataset_file_name = config.params["files"]["raw_dataset_file_name"]

    # Import raw dataset
    raw_dataset = pd.read_parquet(path=data_dir / raw_dataset_file_name)
    required_input_col_names = (
        [pk_col_name]
        + date_col_names
        + datetime_col_names
        + num_col_names
        + cat_col_names
        + [class_column_name]
    )
    raw_dataset = raw_dataset[required_input_col_names].copy()

    return raw_dataset


def preprocess_data(raw_dataset: pd.DataFrame, config_yaml_path: str) -> pd.DataFrame:
    """Preprocesses raw data.

    Args:
        raw_dataset (pd.DataFrame): raw dataset.
        config_yaml_path (str): path to the config yaml file.

    Returns:
        preprocessed_dataset (pd.DataFrame): preprocessed dataset.
        data_preprocessor (DataPreprocessor): data preprocessor object.
    """

    # Get configuration parameters
    config = Config(config_path=config_yaml_path)
    pk_col_name = config.params["data"]["pk_col_name"]
    date_col_names = config.params["data"]["date_col_names"]
    datetime_col_names = config.params["data"]["datetime_col_names"]
    num_col_names = config.params["data"]["num_col_names"]
    cat_col_names = config.params["data"]["cat_col_names"]

    data_preprocessor = DataPreprocessor(
        input_data=raw_dataset,
        primary_key_names=[pk_col_name],
        date_cols_names=date_col_names,
        datetime_cols_names=datetime_col_names,
        num_feature_names=num_col_names,
        cat_feature_names=cat_col_names,
    )

    # Preprecess data for missing values, duplicates, and specify data types
    data_preprocessor.replace_blank_values_with_nan()
    data_preprocessor.check_duplicate_rows()
    data_preprocessor.remove_duplicates_by_primary_key()
    data_preprocessor.specify_data_types()
    preprocessed_dataset = data_preprocessor.get_preprocessed_data()

    return preprocessed_dataset, data_preprocessor


def transform_data(
    preprocessed_dataset: pd.DataFrame,
    data_preprocessor: DataPreprocessor,
    config_yaml_path: str,
) -> pd.DataFrame:
    """Transforms preprocessed data by mapping values and enriching data.

    Args:
        preprocessed_dataset (pd.DataFrame): preprocessed dataset.
        data_preprocessor (DataPreprocessor): data preprocessor class instance.
        config_yaml_path (str): path to the config yaml file.

    Returns:
        pd.DataFrame: transformed dataset.
    """

    # Get configuration parameters
    config = Config(config_path=config_yaml_path)
    class_column_name = config.params["data"]["class_col_name"]
    feature_mappings = config.params["feature_mappings"]
    class_mappings = config.params["class_mappings"]

    # Apply mapping on categorical features (e.g., general health level 3 to "Good")
    data_transformer = DataTransformer(
        preprocessed_data=preprocessed_dataset,
        primary_key_names=data_preprocessor.primary_key_names,
        date_cols_names=data_preprocessor.date_cols_names,
        datetime_cols_names=data_preprocessor.datetime_cols_names,
        num_feature_names=data_preprocessor.num_feature_names,
        cat_feature_names=data_preprocessor.cat_feature_names,
    )

    if feature_mappings is not None:
        column_names = [
            key for key in feature_mappings.keys() if key.endswith("_column")
        ]
        for i in range(0, len(column_names)):
            column_name = column_names[i].removesuffix("_column")
            data_transformer.map_categorical_features(
                col_name=column_name,
                mapping_values=feature_mappings[f"{column_name}_values"],
            )

    if class_mappings is not None:
        data_transformer.map_class_labels(
            class_col_name=class_column_name,
            mapping_values=class_mappings["class_values"],
        )

    transformed_data = data_transformer.preprocessed_data

    return transformed_data


#################################
def main(config_yaml_path: str, data_dir: PosixPath, logger: logging.Logger) -> None:
    """Imports raw data to be preprocessed and transformed.

    Args:
        config_yaml_path (str): path to the config yaml file.
        data_dir (PosixPath): path to the data directory.
        logger (logging.Logger): logger object.

    Returns:
        None.
    """

    logger.info(
        "Directory of data perprocessing and transformation config file: %s",
        config_yaml_path,
    )

    # Get configuration parameters
    config = Config(config_path=config_yaml_path)
    pk_col_name = config.params["data"]["pk_col_name"]
    class_column_name = config.params["data"]["class_col_name"]
    event_timestamp_col_name = config.params["data"]["event_timestamp_col_name"]

    preprocessed_dataset_features_file_name = config.params["files"][
        "preprocessed_data_features_file_name"
    ]
    preprocessed_dataset_target_file_name = config.params["files"][
        "preprocessed_data_target_file_name"
    ]

    # Import raw dataset
    raw_dataset = import_data(config_yaml_path=config_yaml_path, data_dir=data_dir)

    logger.info("Raw dataset was imported.")

    #################################
    # Apply required preprocessing on raw dataset
    # Note: preprocessing and transofmration stepd applied
    # here include mapping values and defining column data
    # types, i.e., doesn't cause data leakage. Hence,
    # transformed dataset can be split into train and test sets.
    preprocessed_dataset, data_preprocessor = preprocess_data(
        raw_dataset=raw_dataset, config_yaml_path=config_yaml_path
    )
    logger.info("Raw dataset was preprocessed.")

    transformed_data = transform_data(
        preprocessed_dataset=preprocessed_dataset,
        data_preprocessor=data_preprocessor,
        config_yaml_path=config_yaml_path,
    )
    logger.info("Preprocessed dataset was transformed.")

    # Save features and target in a separate parquet files
    # Note: this is meant for patient entity in feature store.
    preprocessed_features = transformed_data.drop(
        [class_column_name], axis=1, inplace=False
    )
    preprocessed_features[event_timestamp_col_name] = datetime.now()
    preprocessed_features.to_parquet(
        data_dir / preprocessed_dataset_features_file_name,
        index=False,
    )

    preprocessed_target = transformed_data[[pk_col_name] + [class_column_name]].copy()
    preprocessed_target[event_timestamp_col_name] = datetime.now()
    preprocessed_target.to_parquet(
        data_dir / preprocessed_dataset_target_file_name,
        index=False,
    )

    logger.info("Preprocessed features and target were created.")


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
    console_logger = get_console_logger("prep_data_logger")

    console_logger.info("Transforming for Feature Store Starts ...")

    main(
        config_yaml_path=args.config_yaml_path, data_dir=DATA_DIR, logger=console_logger
    )
