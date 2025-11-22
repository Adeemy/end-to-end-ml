"""
Extracts preprocessed data from feature store, i.e., features and
class labels, and creates data splits for model training.
"""

import argparse
import logging
from datetime import datetime
from pathlib import PosixPath
from typing import Tuple

import pandas as pd
from feast import FeatureStore
from feast.infra.offline_stores.file_source import SavedDatasetFileStorage

from src.feature_store.utils.data import TrainingDataPrep
from src.feature_store.utils.prep import (
    DataSplitter,
    RandomSplitStrategy,
    TimeBasedSplitStrategy,
)
from src.training.utils.config.config import (
    Config,
    TrainFeaturesConfig,
    TrainFilesConfig,
    TrainingConfig,
    TrainPreprocessingConfig,
    build_training_config,
)
from src.utils.config_loader import load_config
from src.utils.logger import get_console_logger
from src.utils.path import DATA_DIR, FEATURE_REPO_DIR


def load_training_config(config_yaml_path: str) -> TrainingConfig:
    """Loads the training configuration from the YAML file.

    Args:
        config_yaml_path: Path to the configuration YAML file.

    Returns:
        TrainingConfig: Parsed training configuration.
    """
    config = Config(config_path=config_yaml_path)
    params = config.params

    return TrainingConfig(
        data=TrainFeaturesConfig(**params["data"]),
        preprocessing=TrainPreprocessingConfig(**params["preprocessing"]),
        files=TrainFilesConfig(**params["files"]),
    )


def import_data(
    training_config: TrainFeaturesConfig,
    files_config: TrainFilesConfig,
    data_dir: PosixPath,
    feast_repo_dir: PosixPath,
) -> pd.DataFrame:
    """Extracts preprocessed data from feature store, i.e., features and
    class labels, and creates data splits for model training.

    Args:
        training_config: Data configuration parameters.
        files_config: File configuration parameters.
        data_dir: Path to the data directory.
        feast_repo_dir: Path to the feature store repo directory.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    feat_store = FeatureStore(repo_path=str(feast_repo_dir))

    # Get historical features and join them with target
    target_data = pd.read_parquet(
        data_dir / files_config.preprocessed_dataset_target_file_name
    )
    historical_data = feat_store.get_historical_features(
        entity_df=target_data,
        features=training_config.historical_features,
    )

    # Retrieve historical dataset into a dataframe
    preprocessed_data = feat_store.create_saved_dataset(
        from_=historical_data,
        name="historical_data",
        storage=SavedDatasetFileStorage(
            f"{data_dir}/{files_config.historical_data_file_name}"
        ),
        allow_overwrite=True,
    ).to_df()

    # Select specified features
    required_input_col_names = (
        [training_config.pk_col_name]
        + training_config.date_col_names
        + training_config.datetime_col_names
        + training_config.num_col_names
        + training_config.cat_col_names
        + [training_config.class_col_name]
    )
    preprocessed_data = preprocessed_data[required_input_col_names].copy()

    return preprocessed_data


def split_data(
    preprocessed_data: pd.DataFrame, training_config: TrainingConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits preprocessed data into train and test sets.

    Args:
        preprocessed_data: Preprocessed data.
        training_config: Data configuration parameters.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Training and testing datasets.
    """

    # Extract cut-off date for splitting train and test sets
    input_split_cutoff_date = None
    if training_config.split_type == "time":
        input_split_cutoff_date = datetime.strptime(
            training_config.train_valid_split_curoff_date,
            training_config.split_date_col_format,
        ).date()

    data_splitter = DataSplitter(
        dataset=preprocessed_data,
        primary_key_col_name=training_config.pk_col_name,
        class_col_name=training_config.class_col_name,
    )

    # Select the appropriate split strategy
    if training_config.split_type == "random":
        split_strategy = RandomSplitStrategy(
            class_col_name=training_config.class_col_name,
            train_set_size=training_config.train_set_size,
            random_seed=int(training_config.split_rand_seed),
        )
    elif training_config.split_type == "time":
        split_strategy = TimeBasedSplitStrategy(
            date_col_name=training_config.split_date_col_name,
            cutoff_date=input_split_cutoff_date,
            date_format=training_config.split_date_col_format,
        )
    else:
        raise ValueError(f"Unsupported split type: {training_config.split_type}")

    training_set, testing_set = data_splitter.split_dataset(split_strategy)
    return training_set, testing_set


def prepare_data(
    training_set: pd.DataFrame,
    testing_set: pd.DataFrame,
    training_config: TrainingConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepares training, validation, and testing sets.

    Args:
        training_set (pd.DataFrame): Training set.
        testing_set (pd.DataFrame): Testing set.
        training_config (TrainingConfig): Training configuration object.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Training, validation, and testing sets.
    """

    # Prepare data for training
    data_prep = TrainingDataPrep(
        train_set=training_set,
        test_set=testing_set,
        primary_key=training_config.pk_col_name,
        class_col_name=training_config.class_col_name,
        numerical_feature_names=training_config.num_col_names,
        categorical_feature_names=training_config.cat_col_names,
    )

    # Preprocess train and test sets
    data_prep.select_relevant_columns()
    data_prep.enforce_data_types()
    data_prep.create_validation_set(
        split_type=training_config.split_type,
        train_set_size=training_config.train_set_size,
        split_random_seed=int(training_config.split_rand_seed),
        split_date_col_name=training_config.split_date_col_name,
        split_cutoff_date=training_config.train_valid_split_curoff_date,
        split_date_col_format=training_config.split_date_col_format,
    )

    return data_prep.train_set, data_prep.valid_set, data_prep.test_set


def main(
    feast_repo_dir: str,
    config_yaml_path: str,
    data_dir: PosixPath,
    logger: logging.Logger,
) -> None:
    """Splits dataset into train and test sets.

    Args:
        feast_repo_dir (str): Path to the feature store repo.
        config_yaml_path (str): Path to the config YAML file.
        data_dir (PosixPath): Path to the data directory.
        logger (logging.Logger): Logger object.
    """
    logger.info("Loading training configuration...")

    config = load_config(
        config_class=Config,
        builder_func=build_training_config,
        config_path=config_yaml_path,
    )
    training_config = config.data
    files_config = config.files

    preprocessed_data = import_data(
        training_config=training_config,
        files_config=files_config,
        data_dir=data_dir,
        feast_repo_dir=feast_repo_dir,
    )
    logger.info("Preprocessed data imported from feature store.")

    training_set, testing_set = split_data(preprocessed_data, training_config)
    logger.info("Preprocessed data split into training and testing sets.")

    train_set, valid_set, test_set = prepare_data(
        training_set, testing_set, training_config
    )
    logger.info("Training, validation, and testing sets prepared.")

    train_set.to_parquet(data_dir / files_config.train_set_file_name, index=False)
    valid_set.to_parquet(data_dir / files_config.valid_set_file_name, index=False)
    test_set.to_parquet(data_dir / files_config.test_set_file_name, index=False)
    logger.info("Train, validation, and test sets saved locally.")


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

    module_name: str = PosixPath(__file__).stem
    console_logger = get_console_logger(module_name)
    console_logger.info(
        "Splitting Preprocessed Data into Train and Test Sets Starts..."
    )

    main(
        config_yaml_path=args.config_yaml_path,
        feast_repo_dir=FEATURE_REPO_DIR,
        data_dir=DATA_DIR,
        logger=console_logger,
    )
