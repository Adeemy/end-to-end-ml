"""
Preprocesses and transforms raw data before saving it in the feature store.
"""

import argparse
import logging
from datetime import datetime
from pathlib import PosixPath
from typing import Tuple

import pandas as pd

from src.feature_store.utils.prep import DataPreprocessor, DataTransformer
from src.utils.config_loader import (
    DataConfig,
    FeatureMappingsConfig,
    FilesConfig,
    load_data_and_train_config,
)
from src.utils.logger import get_console_logger
from src.utils.path import DATA_DIR


def get_config_params(
    config_yaml_path: str,
) -> Tuple[DataConfig, FilesConfig, FeatureMappingsConfig]:
    """Load and return configuration parameters.

    Args:
        config_yaml_path: Path to the configuration YAML file.

    Returns:
        Tuple containing data, file, and feature mappings configuration parameters.
    """
    data_config, _ = load_data_and_train_config(config_yaml_path)
    return data_config.data, data_config.files, data_config.feature_mappings


def import_data(
    data_config: DataConfig, files_config: FilesConfig, data_dir: PosixPath
) -> pd.DataFrame:
    """Imports raw data to be preprocessed and transformed.

    Args:
        data_config: Data configuration parameters.
        files_config: File configuration parameters.
        data_dir: Path to the data directory.

    Returns:
        pd.DataFrame: Raw dataset.
    """
    raw_dataset = pd.read_parquet(path=data_dir / files_config.raw_dataset_file_name)
    required_columns = (
        [data_config.pk_col_name]
        + data_config.date_col_names
        + data_config.datetime_col_names
        + data_config.num_col_names
        + data_config.cat_col_names
        + [data_config.class_col_name]
    )
    return raw_dataset[required_columns].copy()


def preprocess_data(
    raw_dataset: pd.DataFrame, data_config: DataConfig
) -> Tuple[pd.DataFrame, DataPreprocessor]:
    """Preprocesses raw data by handling missing values, duplicates, and data types.

    Args:
        raw_dataset: Raw dataset.
        data_config: Data configuration parameters.

    Returns:
        Tuple containing the preprocessed dataset and the DataPreprocessor instance.
    """
    data_preprocessor = DataPreprocessor(
        input_data=raw_dataset,
        primary_key_names=[data_config.pk_col_name],
        date_cols_names=data_config.date_col_names,
        datetime_cols_names=data_config.datetime_col_names,
        num_feature_names=data_config.num_col_names,
        cat_feature_names=data_config.cat_col_names,
    )

    data_preprocessor.replace_blank_values_with_nan()
    data_preprocessor.check_duplicate_rows()
    data_preprocessor.remove_duplicates_by_primary_key()
    data_preprocessor.specify_data_types()

    return data_preprocessor.get_preprocessed_data(), data_preprocessor


def transform_data(
    preprocessed_dataset: pd.DataFrame,
    data_preprocessor: DataPreprocessor,
    data_config: DataConfig,
    feature_mappings: FeatureMappingsConfig,
) -> pd.DataFrame:
    """Transforms preprocessed data by mapping values and enriching data.

    Args:
        preprocessed_dataset: Preprocessed dataset.
        data_preprocessor: Data preprocessor instance.
        data_config: Data configuration parameters.
        feature_mappings: Feature mappings configuration.

    Returns:
        pd.DataFrame: Transformed dataset.
    """
    data_transformer = DataTransformer(
        preprocessed_data=preprocessed_dataset,
        primary_key_names=data_preprocessor.primary_key_names,
        date_cols_names=data_preprocessor.date_cols_names,
        datetime_cols_names=data_preprocessor.datetime_cols_names,
        num_feature_names=data_preprocessor.num_feature_names,
        cat_feature_names=data_preprocessor.cat_feature_names,
    )

    # Map categorical features
    if feature_mappings.mappings:
        for column, _ in feature_mappings.mappings.items():
            if column.endswith("_column"):
                col_name = column.removesuffix("_column")
                data_transformer.map_categorical_features(
                    col_name=col_name,
                    mapping_values=feature_mappings.mappings[f"{col_name}_values"],
                )

    # Map class labels
    data_transformer.map_class_labels(
        class_col_name=data_config.class_col_name,
        mapping_values=feature_mappings.mappings.get("class_values", {}),
    )

    return data_transformer.preprocessed_data


def save_transformed_data(
    transformed_data: pd.DataFrame,
    data_config: DataConfig,
    files_config: FilesConfig,
    data_dir: PosixPath,
):
    """Saves transformed data to disk to be used in the feature store.

    Args:
        transformed_data: Transformed dataset.
        data_config: Data configuration parameters.
        files_config: File configuration parameters.
        data_dir: Path to the data directory.
    """
    # Save features
    preprocessed_features = transformed_data.drop(
        [data_config.class_col_name], axis=1, inplace=False
    )
    preprocessed_features[data_config.event_timestamp_col_name] = datetime.now()
    preprocessed_features.to_parquet(
        data_dir / files_config.preprocessed_data_features_file_name, index=False
    )

    # Save target
    preprocessed_target = transformed_data[
        [data_config.pk_col_name, data_config.class_col_name]
    ].copy()
    preprocessed_target[data_config.event_timestamp_col_name] = datetime.now()
    preprocessed_target.to_parquet(
        data_dir / files_config.preprocessed_data_target_file_name, index=False
    )


def main(config_yaml_path: str, data_dir: PosixPath, logger: logging.Logger) -> None:
    """Main function to preprocess and transform raw data before saving it in the feature store through
    CI/CD pipeline.

    Args:
        config_yaml_path: Path to the configuration YAML file.
        data_dir: Path to the data directory.
        logger: Logger object.
    """
    logger.info("Starting data preprocessing and transformation...")

    # Load configuration parameters
    data_config, files_config, feature_mappings = get_config_params(config_yaml_path)

    # Import raw dataset
    raw_dataset = import_data(data_config, files_config, data_dir)
    logger.info("Raw dataset imported.")

    # Preprocess raw dataset
    preprocessed_dataset, data_preprocessor = preprocess_data(raw_dataset, data_config)
    logger.info("Raw dataset preprocessed.")

    # Transform preprocessed dataset
    transformed_data = transform_data(
        preprocessed_dataset, data_preprocessor, data_config, feature_mappings
    )
    logger.info("Preprocessed dataset transformed.")

    # Save transformed data
    save_transformed_data(transformed_data, data_config, files_config, data_dir)
    logger.info("Transformed data saved to disk.")


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

    # Get the logger object
    module_name: str = PosixPath(__file__).stem
    console_logger = get_console_logger(module_name)
    console_logger.info("Starting preprocessing for the feature store...")

    main(
        config_yaml_path=args.config_yaml_path, data_dir=DATA_DIR, logger=console_logger
    )
