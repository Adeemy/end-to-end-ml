"""
This script extracts preprocessed data from feature store,
i.e., features and class labels, and creates data splits
for model training. 
"""

import os
import sys
from datetime import datetime

import pandas as pd
from feast import FeatureStore

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
    feat_store = FeatureStore(repo_path=feast_repo_dir)

    print(f"\nfeat_store: {feat_store}\n")

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
    preprocessed_dataset_target_file_name = config.params["files"]["params"][
        "preprocessed_dataset_target_file_name"
    ]
    train_set_file_name = config.params["files"]["params"]["train_set_file_name"]
    test_set_file_name = config.params["files"]["params"]["test_set_file_name"]

    # Extract cut-off date for splitting train and test sets
    input_split_cutoff_date = None
    if DATASET_SPLIT_TYPE == "time":
        input_split_cutoff_date = datetime.strptime(
            SPLIT_CUTOFF_DATE, SPLIT_DATE_FORMAT
        ).date()

    # Get historical features and join them with target
    # Note: this join will take into account even_timestamp such that
    # a target value is joined with the latest feature values prior to
    # event_timestamp of the target. This ensures that class labels of
    # an event is attributed to the correct feature values.
    target_data = pd.read_parquet(path=data_dir / preprocessed_dataset_target_file_name)
    # historical_data = feat_store.get_historical_features(
    #     entity_df=target_data,
    #     features=[
    #         "features_view:BMI",
    #         "features_view:PhysHlth",
    #         "features_view:Age",
    #         "features_view:HighBP",
    #         "features_view:HighChol",
    #         "features_view:CholCheck",
    #         "features_view:Smoker",
    #         "features_view:Stroke",
    #         "features_view:HeartDiseaseorAttack",
    #         "features_view:PhysActivity",
    #         "features_view:Fruits",
    #         "features_view:Veggies",
    #         "features_view:HvyAlcoholConsump",
    #         "features_view:AnyHealthcare",
    #         "features_view:NoDocbcCost",
    #         "features_view:GenHlth",
    #         "features_view:MentHlth",
    #         "features_view:DiffWalk",
    #         "features_view:Sex",
    #         "features_view:Education",
    #         "features_view:Income",
    #     ],
    # )

    # # Retrieve historical dataset into a dataframe
    # preprocessed_data = historical_data.to_df()

    historical_features = pd.read_parquet(
        path=data_dir
        / "./src/feature_store/feature_repo/data/preprocessed_dataset_features.parquet"
    )

    preprocessed_data = historical_features.set_index(PRIMARY_KEY).join(
        target_data.set_index(PRIMARY_KEY), how="inner"
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


###########################################################
if __name__ == "__main__":
    main(
        feast_repo_dir=sys.argv[1],
        config_yaml_abs_path=sys.argv[2],
        data_dir=DATA_DIR,
    )
