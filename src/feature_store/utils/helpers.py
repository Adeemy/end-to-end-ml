"""
Wrapper functions for loading and splitting UCI data, and preprocessing data for model development.
"""

import pandas as pd
from azureml.core import Dataset, Workspace
from ucimlrepo import fetch_ucirepo

from src.feature_store.utils.config import Config
from src.feature_store.utils.prep import DataPreprocessor, DataSplitter, DataTransformer

##########################################################


def import_uci_data(config_yaml_path: str) -> pd.DataFrame:
    """Imports raw data from UCI data repository. This is a wrapper function
    used in importing data from UCI data repository.

    Args:
        config_yaml_path (str): path to the config yaml file.

    Returns:
        raw_dataset (pd.DataFrame): raw dataset.
    """

    # Get configuration parameters
    config = Config(config_path=config_yaml_path)
    uci_dataset_id = config.params["data"]["uci_raw_data_num"]
    pk_col_name = config.params["data"]["pk_col_name"]
    class_column_name = config.params["data"]["class_col_name"]
    date_col_names = config.params["data"]["date_col_names"]
    datetime_col_names = config.params["data"]["datetime_col_names"]
    num_col_names = config.params["data"]["num_col_names"]
    cat_col_names = config.params["data"]["cat_col_names"]

    # Import raw data
    raw_data = fetch_ucirepo(id=uci_dataset_id)
    raw_dataset = raw_data.data.features.copy()
    raw_dataset[pk_col_name] = raw_data.data.ids.loc[:, [pk_col_name]]
    raw_dataset[class_column_name] = raw_data.data.targets.loc[:, [class_column_name]]

    # Select relevant columns
    required_input_col_names = (
        [pk_col_name]
        + date_col_names
        + datetime_col_names
        + num_col_names
        + cat_col_names
        + [class_column_name]
    )
    raw_dataset = raw_dataset[required_input_col_names]

    return raw_dataset


def split_uci_data(
    raw_dataset: pd.DataFrame, config_yaml_path: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits raw dataset into training and inference sets. This is a wrapper
    function used in splitting raw dataset into training and inference sets.

    Args:
        raw_dataset (pd.DataFrame): raw dataset.
        config_yaml_path (str): path to the config yaml file.

    Returns:
        raw_dataset (pd.DataFrame): raw dataset for model development.
        inference_set (pd.DataFrame): inference set for production data simulation.
    """

    # Get configuration parameters
    config = Config(config_path=config_yaml_path)
    inference_set_ratio = float(config.params["data"]["inference_set_ratio"])
    random_seed = int(config.params["data"]["random_seed"])
    original_split_type = config.params["data"]["original_split_type"]
    pk_col_name = config.params["data"]["pk_col_name"]
    class_column_name = config.params["data"]["class_col_name"]

    # Create inference set from raw dataset to simulate production data
    train_valid_splitter = DataSplitter(
        dataset=raw_dataset,
        primary_key_col_name=pk_col_name,
        class_col_name=class_column_name,
    )

    raw_dataset, inference_set = train_valid_splitter.split_dataset(
        split_type=original_split_type,
        train_set_size=1 - inference_set_ratio,
        split_random_seed=random_seed,
    )

    return raw_dataset, inference_set


def import_raw_data(config_yaml_path: str, ws: Workspace) -> pd.DataFrame:
    """Imports raw data to be preprocessed and transformed for model development.

    Args:
        config_yaml_path (str): path to the config yaml file.
        ws (Workspace): workspace object.

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
    raw_dataset_name = config.params["azure_datasets"]["raw_dataset_name"]

    dataset = Dataset.get_by_name(ws, name=raw_dataset_name)
    raw_dataset = dataset.to_pandas_dataframe()

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
