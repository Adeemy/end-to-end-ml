"""
This module contains helper functions used within the main 
function in train.py
"""

import os
import re
import sys
from typing import Callable, Literal, Union

import numpy as np
import pandas as pd
from datasets import load_dataset
from numpy import ravel
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (  # StandardScaler,; RobustScaler,
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
)

sys.path.insert(0, os.getcwd())
from src.feature_store.utils.prep import DataPreprocessor, DataSplitter

###########################################################


class DataPipelineCreator:
    def __init__(
        self,
        num_features_imputer: str = "median",
        num_features_scaler: Union[Callable, None] = None,
        cat_features_imputer: str = "constant",
        cat_features_ohe_handle_unknown: str = "infrequent_if_exist",
        cat_features_nans_replacement: float = np.nan,
    ):
        self.num_features_imputer = num_features_imputer
        self.num_features_scaler = num_features_scaler
        self.cat_features_imputer = cat_features_imputer
        self.cat_features_ohe_handle_unknown = cat_features_ohe_handle_unknown
        self.cat_features_nans_replacement = cat_features_nans_replacement

    def create_num_features_pipeline(
        self,
    ) -> Pipeline:
        """Creates sklearn pipeline for numerical features."""
        num_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=self.num_features_imputer)),
                ("scaler", self.num_features_scaler),
            ]
        )

        return num_transformer

    def create_cat_features_transformer(
        self,
    ) -> Pipeline:
        """Creates sklearn pipeline for categorical features."""
        cat_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy=self.cat_features_imputer,
                        fill_value=self.cat_features_nans_replacement,
                    ),
                ),
                (
                    "onehot_encoder",
                    OneHotEncoder(
                        handle_unknown=self.cat_features_ohe_handle_unknown,
                        categories="auto",
                        drop="first",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        return cat_transformer

    ###########################################################
    def create_data_pipeline(
        self,
        input_features: pd.DataFrame,
        num_feature_col_names: list = None,
        cat_feature_col_names: list = None,
        variance_threshold_val: float = 0.05,
    ) -> Union[pd.DataFrame, Pipeline]:
        """
        Creates a data pipeline to transform training set features (class column
        should not be included) and returns transformed train set features as
        pd.DataFrame and fitted pipeline to transform validation and test sets.
        """

        # Copy input data
        features_set = input_features.copy()

        # Set column names to [] if None was provided
        num_feature_col_names = (
            [] if num_feature_col_names is None else num_feature_col_names
        )
        cat_feature_col_names = (
            [] if cat_feature_col_names is None else cat_feature_col_names
        )

        # Assert that at least one numerical or categorical feature is specified
        assert (
            len(num_feature_col_names + cat_feature_col_names) > 0
        ), "At least one numerical or categorical feature name must be specified!"

        # Create a numerical features transformer
        if len(num_feature_col_names) > 0:
            numeric_transformer = self.create_num_features_pipeline()
        else:
            numeric_transformer = Pipeline(
                steps=[
                    ("numeric", "passthrough"),
                ]
            )

        # Create a categorical features transformer
        if len(cat_feature_col_names) > 0:
            cat_features_transformer = self.create_cat_features_transformer()
        else:
            cat_features_transformer = Pipeline(
                steps=[
                    ("categorical", "passthrough"),
                ]
            )

        print(f"\nNumerical features: {num_feature_col_names}")
        print(f"\nCategorical features: {cat_feature_col_names}")

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, num_feature_col_names),
                (
                    "categorical",
                    cat_features_transformer,
                    cat_feature_col_names,
                ),
            ]
        )
        selector = VarianceThreshold(threshold=variance_threshold_val)
        data_pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("selector", selector)]
        )

        # Fit and transform pipeline
        transformed_data = data_pipeline.fit_transform(features_set)
        transformed_data = pd.DataFrame(transformed_data)

        # Extract numerical and one-hot encoded features names
        col_names = []
        if len(cat_feature_col_names) > 0:
            col_names = num_feature_col_names + list(
                data_pipeline.named_steps["preprocessor"]
                .transformers_[1][1]
                .named_steps["onehot_encoder"]
                .get_feature_names_out(cat_feature_col_names)
            )
        else:
            col_names = num_feature_col_names

        # Get feature names that were selected by selector step
        col_names = [i for (i, v) in zip(col_names, list(selector.get_support())) if v]

        # Rename transformed dataframe columns
        transformed_data.columns = col_names

        return transformed_data, data_pipeline


def import_datasets(
    hf_data_source: str = None,
    train_set_path: str = None,
    test_set_path: str = None,
    is_local_source: bool = False,
) -> Union[pd.DataFrame, pd.DataFrame]:
    """Imports the training and testing sets from parquet files. It returns
    two pandas dataframes: train_set and test_set."""

    if is_local_source:
        train_set = pd.read_parquet(path=train_set_path)
        test_set = pd.read_parquet(path=test_set_path)

    else:
        # Import train and test sets
        dataset = load_dataset(hf_data_source)
        train_set = dataset["train"].to_pandas()
        test_set = dataset["test"].to_pandas()

    return train_set, test_set


def select_relevant_columns(
    primary_key: str,
    class_col_name: str,
    numerical_feature_names: list,
    categorical_feature_names: list,
    train_set: pd.DataFrame,
    test_set: pd.DataFrame,
) -> Union[pd.DataFrame, pd.DataFrame, list, list]:
    """Ensures specified numerical and categorical features exist in train and test
    set, and returns train and test sets with the selected columns."""

    # Update specified columns in case some columns were dropped from preprocessed dataset
    numerical_feature_names = [
        x
        for x in numerical_feature_names
        if x in train_set.columns and x in test_set.columns
    ]

    categorical_feature_names = [
        x
        for x in categorical_feature_names
        if x in train_set.columns and x in test_set.columns
    ]

    # Select features and class
    # Note: keep primary key for now as it's needed later to
    # create validation set.
    train_set = train_set[
        [primary_key]
        + numerical_feature_names
        + categorical_feature_names
        + [class_col_name]
    ]
    test_set = test_set[
        [primary_key]
        + numerical_feature_names
        + categorical_feature_names
        + [class_col_name]
    ]

    return train_set, test_set, numerical_feature_names, categorical_feature_names


def preprocess_datasets(
    train_set: pd.DataFrame,
    test_set: pd.DataFrame,
    numerical_feature_names: list = None,
    categorical_feature_names: list = None,
) -> Union[pd.DataFrame, pd.DataFrame]:
    """Enforces data types of numerical and categorical features.
    Note: if only numerical feature names are provided, all other features will
    be considered categorical and will be converted to string.
    """

    # Assert that at least one feature data type was passed
    assert (
        len(numerical_feature_names + categorical_feature_names) > 0
    ), "Name of numerical or categorical features must be provided. None was provided!"

    train_set_processor = DataPreprocessor(
        input_data=train_set,
        num_feature_names=numerical_feature_names,
        cat_feature_names=categorical_feature_names,
    )
    train_set_processor.specify_data_types()
    train_set = train_set_processor.get_preprocessed_data()

    test_set_processor = DataPreprocessor(
        input_data=test_set,
        num_feature_names=numerical_feature_names,
        cat_feature_names=categorical_feature_names,
    )
    test_set_processor.specify_data_types()
    test_set = test_set_processor.get_preprocessed_data()

    return train_set, test_set


def replace_nans_in_cat_features(
    categorical_feature_names: list,
    train_set: pd.DataFrame,
    test_set: pd.DataFrame,
    nan_replacement: str = "Unspecified",
) -> Union[pd.DataFrame, pd.DataFrame]:
    """Replaces missing values with NaNs to allow converting them from float to integer.
    Note: the error (AttributeError: 'bool' object has no attribute 'transpose') is raised
    when transforming train set possibly because of pd.NA."""

    train_set[categorical_feature_names] = train_set[categorical_feature_names].replace(
        {pd.NA: nan_replacement}
    )

    test_set[categorical_feature_names] = test_set[categorical_feature_names].replace(
        {pd.NA: nan_replacement}
    )

    return train_set, test_set


def create_validation_set(
    primary_key: str,
    class_col_name: str,
    train_set: pd.DataFrame,
    split_type: Literal["time", "random"] = "random",
    train_set_size: float = 0.8,
    split_random_seed: int = None,
    split_date_col_name: str = None,
    split_cutoff_date: str = None,
    split_date_col_format: str = "%Y-%m-%d %H:%M:%S",
) -> Union[pd.DataFrame, pd.DataFrame]:
    """Creates a validation set by splitting training set into training and
    validation sets randomly or based on time.
    Note: validation set will be used to select the best model.
    """
    data_splitter = DataSplitter(
        dataset=train_set,
        primary_key_col_name=primary_key,
        class_col_name=class_col_name,
    )

    train_set, valid_set = data_splitter.split_dataset(
        split_type=split_type,
        train_set_size=train_set_size,
        split_random_seed=split_random_seed,
        split_date_col_name=split_date_col_name,
        split_cutoff_date=split_cutoff_date,
        split_date_col_format=split_date_col_format,
    )

    return train_set, valid_set


def drop_primary_key(
    primary_key: str,
    train_set: pd.DataFrame,
    valid_set: pd.DataFrame,
    test_set: pd.DataFrame,
) -> Union[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Drop primary key column (not needed in training)."""

    train_set.drop([primary_key], axis=1, inplace=True)
    valid_set.drop([primary_key], axis=1, inplace=True)
    test_set.drop([primary_key], axis=1, inplace=True)

    return train_set, valid_set, test_set


def seperate_features_from_class_labels(
    class_col_name: str,
    dataset: pd.DataFrame,
    numerical_feature_names: list = None,
    categorical_feature_names: list = None,
) -> Union[pd.DataFrame, pd.DataFrame]:
    """Separate features and class column of testing set."""

    selected_features = numerical_feature_names + categorical_feature_names
    dataset_features = dataset.loc[:, selected_features]
    dataset_class = dataset[[class_col_name]]

    return dataset_features, dataset_class


def encode_class_labels(
    class_labels: pd.DataFrame,
    pos_class_label: str,
    fitted_class_encoder: LabelEncoder = None,
) -> Union[pd.DataFrame, np.ndarray, LabelEncoder]:
    """Encode class labels into integers and returns encoded class labels
    of training, validation, and testing sets in addition to encoded positive
    class label."""

    # Accept class encoder if provided, e.g., already fitted on training class
    # labels, otherwise, create it.
    if fitted_class_encoder is None:
        fitted_class_encoder = LabelEncoder()

    # Encode class labels
    encoded_class = fitted_class_encoder.fit_transform(ravel(class_labels))

    # Get the encoded value of the positive class label
    enc_pos_class_label = fitted_class_encoder.transform([pos_class_label])[0]

    return (encoded_class, enc_pos_class_label, fitted_class_encoder)


def create_data_transformation_pipeline(
    numerical_feature_names: list,
    categorical_feature_names: list,
    training_features: pd.DataFrame,
    validation_features: pd.DataFrame,
) -> Union[pd.DataFrame, pd.DataFrame, Pipeline]:
    """Creates a data transformation pipeline and fit it on training set."""

    train_set_transformer = DataPipelineCreator(
        num_features_imputer="median",
        num_features_scaler=MinMaxScaler(),
        cat_features_imputer="constant",
        cat_features_ohe_handle_unknown="infrequent_if_exist",
        cat_features_nans_replacement=np.nan,
    )
    (
        train_features_preprocessed,
        data_transformation_pipeline,
    ) = train_set_transformer.create_data_pipeline(
        input_features=training_features,
        num_feature_col_names=numerical_feature_names,
        cat_feature_col_names=categorical_feature_names,
        variance_threshold_val=0.05,
    )

    # Transform validation set using the same training set transformation
    # Note: these transfomred versions of train and validation sets will be
    # used inside the objective function to avoid applying data transformation
    # at each function call during optimization procedure.
    valid_features_preprocessed = data_transformation_pipeline.transform(
        validation_features
    )
    valid_features_preprocessed = pd.DataFrame(valid_features_preprocessed)
    valid_features_preprocessed.columns = list(train_features_preprocessed.columns)

    return (
        train_features_preprocessed,
        valid_features_preprocessed,
        data_transformation_pipeline,
    )


def clean_up_feature_names(
    dataset_features: pd.DataFrame,
) -> list:
    """Cleans up feature names of datasets to prevent errors that may arise
    because of special characters and returns clean feature names. This issue
    can be ebcountered after one-hot encoding where some category values with
    special characters can be become problematic column names.
    """

    # Remove special characters from column name to avoid error that LightGBM does
    # not support feature names with special characters
    dataset_features.rename(
        columns=lambda x: re.sub("[^A-Za-z0-9]+", "_", x), inplace=True
    )

    return list(dataset_features.columns)
