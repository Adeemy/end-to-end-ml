"""
This script prepares data retrieved from feature store
for training. 
"""

import os
import sys

import pandas as pd
from feast import FeatureStore
from feast.infra.offline_stores.file_source import SavedDatasetFileStorage

sys.path.insert(0, os.getcwd())
from pathlib import PosixPath

from src.feature_store.utils.config import Config
from src.feature_store.utils.prep import DataPreprocessor, DataTransformer
from src.training.utils.path import DATA_DIR


#################################
def main(feast_repo_dir: str, config_yaml_abs_path: str, data_dir: PosixPath):
    """Imports data from feature store to be preprocessed and transformed.
    feast_repo_dir (str): relative path to feature store repo.
    config_yaml_abs_path (str): absolute path to config.yml file, which
        includes dataset preprocessing configuration.
    """

    print(
        """\n
    ----------------------------------------------------------------
    --- Preparing Data Imported from Feature Store Starts ...
    ----------------------------------------------------------------\n"""
    )

    # Get initiated FeatureStore
    # Note: repo_path is the relative path to where this script is located.
    store = FeatureStore(repo_path=feast_repo_dir)
    config = Config(config_path=config_yaml_abs_path)
    PRIMARY_KEY = config.params["data"]["params"]["pk_col_name"]
    CLASS_COL_NAME = config.params["data"]["params"]["class_col_name"]
    date_col_names = config.params["data"]["params"]["date_col_names"]
    datetime_col_names = config.params["data"]["params"]["datetime_col_names"]
    num_col_names = config.params["data"]["params"]["num_col_names"]
    cat_col_names = config.params["data"]["params"]["cat_col_names"]
    raw_dataset_file_name = config.params["files"]["params"]["raw_dataset_file_name"]
    raw_dataset_target_file_name = config.params["files"]["params"][
        "raw_dataset_target_file_name"
    ]
    preprocessed_dataset_file_name = config.params["files"]["params"][
        "preprocessed_dataset_file_name"
    ]

    # Get historical features and join them with target
    # Note: this join will take into account even_timestamp such that
    # a target value is joined with the latest feature values prior to
    # event_timestamp of the target. This ensures that class labels of
    # an event is attributed to the correct feature values.
    target_data = pd.read_parquet(path=data_dir / raw_dataset_target_file_name)
    raw_data = store.get_historical_features(
        entity_df=target_data,
        features=[
            "features_view:BMI",
            "features_view:PhysHlth",
            "features_view:Age",
            "features_view:HighBP",
            "features_view:HighChol",
            "features_view:CholCheck",
            "features_view:Smoker",
            "features_view:Stroke",
            "features_view:HeartDiseaseorAttack",
            "features_view:PhysActivity",
            "features_view:Fruits",
            "features_view:Veggies",
            "features_view:HvyAlcoholConsump",
            "features_view:AnyHealthcare",
            "features_view:NoDocbcCost",
            "features_view:GenHlth",
            "features_view:MentHlth",
            "features_view:DiffWalk",
            "features_view:Sex",
            "features_view:Education",
            "features_view:Income",
        ],
    )

    # Store raw dataset with class labels as a local file
    # and register it as "raw_dataset", which will be used
    # for training.
    raw_dataset = store.create_saved_dataset(
        from_=raw_data,
        name="raw_data",
        storage=SavedDatasetFileStorage(str(data_dir) + "/" + raw_dataset_file_name),
        allow_overwrite=True,
    ).to_df()

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

    # Save preprocessed and transformed dataset to a local path
    # Note: it can be registered in a dataset versioning system in
    # a cloud platform.
    preprocessed_data.to_parquet(data_dir / preprocessed_dataset_file_name, index=False)

    print("\nPreprocessed dataset was created.\n")


# python ./src/feature_store/prep_data.py src/feature_store/feature_repo/ ./config/feature_store/config.yml
if __name__ == "__main__":
    # Preprocess and transform data
    main(
        feast_repo_dir=sys.argv[1], config_yaml_abs_path=sys.argv[2], data_dir=DATA_DIR
    )
