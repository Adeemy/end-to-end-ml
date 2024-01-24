"""
This script prepares data retrieved from feature store
for training. 
"""

import sys
from datetime import datetime
from pathlib import PosixPath

import pandas as pd
from utils.config import Config
from utils.prep import DataPreprocessor, DataTransformer

from src.config.path import DATA_DIR
from src.feature_store.utils.config import Config
from src.feature_store.utils.prep import DataPreprocessor, DataTransformer


#################################
def main(config_yaml_abs_path: str, data_dir: PosixPath):
    """Imports data from feature store to be preprocessed and transformed.
    feast_repo_dir (str): relative path to feature store repo.
    config_yaml_abs_path (str): absolute path to config.yml file, which
        includes dataset preprocessing configuration.
    """

    print(
        """\n
    ----------------------------------------------------------------
    --- Transforming for Feature Store Starts ...
    ----------------------------------------------------------------\n"""
    )

    # Get initiated FeatureStore
    # Note: repo_path is the relative path to where this script is located.
    config = Config(config_path=config_yaml_abs_path)
    PRIMARY_KEY = config.params["data"]["params"]["pk_col_name"]
    CLASS_COL_NAME = config.params["data"]["params"]["class_col_name"]
    date_col_names = config.params["data"]["params"]["date_col_names"]
    datetime_col_names = config.params["data"]["params"]["datetime_col_names"]
    num_col_names = config.params["data"]["params"]["num_col_names"]
    cat_col_names = config.params["data"]["params"]["cat_col_names"]
    event_timestamp_col_name = config.params["data"]["params"][
        "event_timestamp_col_name"
    ]
    raw_dataset_file_name = config.params["files"]["params"]["raw_dataset_file_name"]
    preprocessed_dataset_features_file_name = config.params["files"]["params"][
        "preprocessed_dataset_features_file_name"
    ]
    preprocessed_dataset_target_file_name = config.params["files"]["params"][
        "preprocessed_dataset_target_file_name"
    ]

    #################################
    # Import raw dataset
    raw_dataset = pd.read_parquet(path=data_dir / raw_dataset_file_name)

    #################################
    # Apply required preprocessing on raw dataset
    # Note: preprocessing and transofmration stepd applied
    # here include mapping values and defining column data
    # types, i.e., doesn't cause data leakage. Hence,
    # transformed dataset can be split into train and test sets.
    required_input_col_names = (
        [PRIMARY_KEY]
        + date_col_names
        + datetime_col_names
        + num_col_names
        + cat_col_names
        + [CLASS_COL_NAME]
    )
    raw_dataset = raw_dataset[required_input_col_names].copy()

    #################################
    data_preprocessor = DataPreprocessor(
        input_data=raw_dataset,
        primary_key_names=[PRIMARY_KEY],
        date_cols_names=date_col_names,
        datetime_cols_names=datetime_col_names,
        num_feature_names=num_col_names,
        cat_feature_names=cat_col_names,  # If None, cat. vars are all cols except num., date, & datetime cols.
    )

    # Preprecess data for missing values, duplicates, and specify data types
    data_preprocessor.replace_blank_values_with_nan()
    data_preprocessor.check_duplicate_rows()
    data_preprocessor.remove_duplicates_by_primary_key()
    data_preprocessor.specify_data_types()
    preprocessed_dataset = data_preprocessor.get_preprocessed_data()

    # Apply mapping on categorical features (e.g., general health level 3 to "Good")
    data_transformer = DataTransformer(
        preprocessed_data=preprocessed_dataset,
        primary_key_names=data_preprocessor.primary_key_names,
        date_cols_names=data_preprocessor.date_cols_names,
        datetime_cols_names=data_preprocessor.datetime_cols_names,
        num_feature_names=data_preprocessor.num_feature_names,
        cat_feature_names=data_preprocessor.cat_feature_names,
    )

    data_transformer.map_categorical_features()
    data_transformer.rename_class_labels(class_col_name=CLASS_COL_NAME)
    preprocessed_data = data_transformer.preprocessed_data

    # Save features and target in a separate parquet files
    # Note: this is meant for patient entity in feature store.
    preprocessed_features = raw_dataset.drop([CLASS_COL_NAME], axis=1, inplace=False)
    preprocessed_features[event_timestamp_col_name] = datetime.now()
    preprocessed_features.to_parquet(
        data_dir / preprocessed_dataset_features_file_name,
        index=False,
    )

    preprocessed_target = preprocessed_data[[PRIMARY_KEY] + [CLASS_COL_NAME]].copy()
    preprocessed_target[event_timestamp_col_name] = datetime.now()
    preprocessed_target.to_parquet(
        data_dir / preprocessed_dataset_target_file_name,
        index=False,
    )

    print("\n\nPreprocessed features and target were saved in feature store.")


###########################################################
if __name__ == "__main__":
    main(config_yaml_abs_path=sys.argv[1], data_dir=DATA_DIR)
