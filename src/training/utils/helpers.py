"""
Wrapper functions for loading and splitting prepared data into three data splits, and transform training
and validation sets for model training.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from azureml.core import Dataset, Workspace
from dotenv import load_dotenv

# To import modules from the parent directory in Azure compute cluster
root_dir = Path(__name__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from src.feature_store.utils.prep import DataSplitter
from src.training.utils.config import Config
from src.training.utils.data import TrainingDataPrep

load_dotenv()

##########################################################


def import_preprocessed_data(
    config_file_path: str, workspace: Workspace
) -> pd.DataFrame:
    """Imports preprocessed data from Azure Ml workspace.

    Args:
        config_file_path (str): Path to the config file.
        workspace (Workspace): Workspace object.

    Returns:
        pd.DataFrame: Preprocessed data.
    """

    config = Config(config_path=config_file_path)
    preprocessed_data_name = config.params["azure_datasets"]["preprocessed_data_name"]

    primary_key_column_name = config.params["data"]["pk_col_name"]
    class_column_name = config.params["data"]["class_col_name"]
    date_column_names = config.params["data"]["date_col_names"]
    datetime_column_names = config.params["data"]["datetime_col_names"]
    numeric_column_names = config.params["data"]["num_col_names"]
    categorical_column_names = config.params["data"]["cat_col_names"]

    # Get preprocessed data
    preprocessed_dataset = Dataset.get_by_name(
        workspace=workspace, name=preprocessed_data_name
    )
    preprocessed_data = preprocessed_dataset.to_pandas_dataframe()

    required_column_names = (
        [primary_key_column_name]
        + date_column_names
        + datetime_column_names
        + numeric_column_names
        + categorical_column_names
        + [class_column_name]
    )
    preprocessed_data = preprocessed_data[required_column_names].copy()

    return preprocessed_data


def split_data(
    preprocessed_data: pd.DataFrame,
    config_yaml_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits preprocessed data into train and test sets.

    Args:
        preprocessed_data (pd.DataFrame): Preprocessed data.
        config_yaml_path (str): Path to the config yaml file.

    Returns:
        training_set (pd.DataFrame): Training set.
        testing_set (pd.DataFrame): Testing set.
    """

    config = Config(config_path=config_yaml_path)
    primary_key_col_name = config.params["data"]["pk_col_name"]
    class_col_name = config.params["data"]["class_col_name"]
    dataset_split_type = config.params["data"]["split_type"]
    split_rand_seed = int(config.params["data"]["split_rand_seed"])
    train_set_ratio = config.params["data"]["train_set_size"]
    dataset_split_date_col_name = config.params["data"]["split_date_col_name"]
    train_valid_split_curoff_date = config.params["data"][
        "train_valid_split_curoff_date"
    ]
    dataset_split_date_col_format = config.params["data"]["split_date_col_format"]

    # Extract cut-off date for splitting train and test sets
    split_cutoff_date = None
    if dataset_split_type == "time":
        split_cutoff_date = datetime.strptime(
            train_valid_split_curoff_date, dataset_split_date_col_format
        ).date()

    # Split data into train and test sets
    data_splitter = DataSplitter(
        dataset=preprocessed_data,
        primary_key_col_name=primary_key_col_name,
        class_col_name=class_col_name,
    )

    training_set, testing_set = data_splitter.split_dataset(
        split_type=dataset_split_type,
        train_set_size=train_set_ratio,
        split_random_seed=split_rand_seed,
        split_date_col_name=dataset_split_date_col_name,
        split_cutoff_date=split_cutoff_date,
        split_date_col_format=dataset_split_date_col_format,
    )

    return training_set, testing_set


def prepare_data_splits(
    training_set: pd.DataFrame,
    testing_set: pd.DataFrame,
    config_yaml_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepares data splits for training, validation, and testing sets. It splits
    training set into training and validation sets.

    Args:
        training_set (pd.DataFrame): training set.
        testing_set (pd.DataFrame): testing set.
        config_yaml_path (str): path to the config yaml file.

    Returns:
        training_set (pd.DataFrame): training set.
        validation_set (pd.DataFrame): validation set.
        testing_set (pd.DataFrame): testing set.
    """

    # Get configuration parameters
    config = Config(config_path=config_yaml_path)
    pk_col_name = config.params["data"]["pk_col_name"]
    class_column_name = config.params["data"]["class_col_name"]
    num_col_names = config.params["data"]["num_col_names"]
    cat_col_names = config.params["data"]["cat_col_names"]
    dataset_split_type = config.params["data"]["split_type"]
    split_rand_seed = int(config.params["data"]["split_rand_seed"])
    train_set_ratio = config.params["data"]["train_set_size"]
    dataset_split_date_col_name = config.params["data"]["split_date_col_name"]
    train_valid_split_curoff_date = config.params["data"][
        "train_valid_split_curoff_date"
    ]
    dataset_split_date_col_format = config.params["data"]["split_date_col_format"]

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

    train_set = data_prep.train_set
    valid_set = data_prep.valid_set
    test_set = data_prep.test_set

    return train_set, valid_set, test_set


def prepare_training_data(
    config_yaml_path: str,
    training_set: pd.DataFrame,
    validation_set: pd.DataFrame,
    testing_set: pd.DataFrame,
) -> Tuple[
    TrainingDataPrep,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    List[str],
    List[str],
    LabelEncoder,
    int,
]:
    """Prepares data for training by preprocessing training and validation sets
    for training job. It also encodes class labels and returns the encoded positive class label.

    Args:
        config_yaml_path (str): path to config yaml file.
        training_set (pd.DataFrame): training set.
        validation_set (pd.DataFrame): validation set.
        testing_set (pd.DataFrame): testing set.

    Returns:
        data_prep (TrainingDataPrep): data preparation object.
        data_transformation_pipeline (Pipeline): data transformation pipeline.
        train_features (pd.DataFrame): preprocessed training features.
        valid_features (pd.DataFrame): preprocessed validation features.
        test_features (pd.DataFrame): preprocessed testing features.
        train_class (pd.Series): training class labels.
        valid_class (pd.Series): validation class labels.
        test_class (pd.Series): testing class labels.
        num_feature_names (List[str]): numerical feature names.
        cat_feature_names (List[str]): categorical feature names.
        class_encoder (LabelEncoder): class label encoder.
        encoded_positive_class_label (int): encoded positive class label.
    """

    config = Config(config_path=config_yaml_path)
    pk_col_name = config.params["data"]["pk_col_name"]
    class_column_name = config.params["data"]["class_col_name"]
    num_col_names = config.params["data"]["num_col_names"]
    cat_col_names = config.params["data"]["cat_col_names"]
    pos_class = config.params["data"]["pos_class"]

    num_features_imputer = config.params["preprocessing"]["num_features_imputer"]
    num_features_scaler = config.params["preprocessing"]["num_features_scaler"]
    scaler_params = config.params["preprocessing"].get("scaler_params", {})
    cat_features_imputer = config.params["preprocessing"]["cat_features_imputer"]
    cat_features_ohe_handle_unknown = config.params["preprocessing"][
        "cat_features_ohe_handle_unknown"
    ]
    cat_features_nans_replacement = config.params["preprocessing"][
        "cat_features_nans_replacement"
    ]
    var_thresh_val = config.params["preprocessing"]["var_thresh_val"]

    # Prepare data for training
    data_prep = TrainingDataPrep(
        train_set=training_set,
        test_set=testing_set,
        primary_key=pk_col_name,
        class_col_name=class_column_name,
        numerical_feature_names=num_col_names,
        categorical_feature_names=cat_col_names,
    )
    data_prep.extract_features(valid_set=validation_set)
    data_prep.enforce_data_types()

    # Encode class labels
    # Note: class encoder is fitted on train class labels and will be used
    # to transform validation and test class labels.
    (
        train_class,
        valid_class,
        encoded_positive_class_label,
        class_encoder,
    ) = data_prep.encode_class_labels(
        pos_class_label=pos_class,
    )

    # Return features
    train_features = data_prep.training_features
    valid_features = data_prep.validation_features

    # Define the mapping from strings to scaler classes
    scaler_mapping = {
        "robust": RobustScaler,
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "none": None,
    }
    scaler_class = scaler_mapping[num_features_scaler]
    scaler_params = {k: v for d in scaler_params for k, v in d.items()}

    # Create data transformation pipeline
    data_transformation_pipeline = data_prep.create_data_transformation_pipeline(
        num_features_imputer=num_features_imputer,
        num_features_scaler=scaler_class(**scaler_params),
        cat_features_imputer=cat_features_imputer,
        cat_features_ohe_handle_unknown=cat_features_ohe_handle_unknown,
        cat_features_nans_replacement=cat_features_nans_replacement,
        var_thresh_val=var_thresh_val,
    )
    data_prep.clean_up_feature_names()
    num_feature_names, cat_feature_names = data_prep.get_feature_names()

    return (
        data_prep,
        data_transformation_pipeline,
        train_features,
        valid_features,
        train_class,
        valid_class,
        num_feature_names,
        cat_feature_names,
        class_encoder,
        encoded_positive_class_label,
    )
