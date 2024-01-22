"""
This module contains helper functions used within the main 
function in train.py
"""

import re
import sys
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import numpy as np
import pandas as pd
from numpy import ravel
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (  # StandardScaler, RobustScaler,
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
)

sys.path.append(str(Path(__file__).parent.resolve().parent.parent))

from feature_store.utils.prep import DataPreprocessor, DataSplitter

###########################################################


class DataPipelineCreator:
    """A class to create a data preprocessing pipeline using sklearn."""

    def __init__(
        self,
        num_features_imputer: str = "median",
        num_features_scaler: Optional[Union[Callable, None]] = None,
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
        num_feature_col_names: Optional[list] = None,
        cat_feature_col_names: Optional[list] = None,
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


class PrepTrainingData:
    """A class to prep data for training during data split. It can be used to create the
    three data splits: training, validation, and testing, and apply minimal preprocessing
    steps that doesn't cause data leakage. It can also be used in training script to ensure
    all data splits have proper data types espcially when importing data from csv files. The
    class can also be used to fit label encoder on training class and use the fitted encoder
    to transform the validation and testing set class labels. Although there is no dependencies
    between methods, this class can be further improved in future refactoring efforts.
    """

    def __init__(
        self,
        train_set: pd.DataFrame,
        test_set: pd.DataFrame,
        primary_key: str,
        class_col_name: str,
        numerical_feature_names: Optional[list] = None,
        categorical_feature_names: Optional[list] = None,
    ) -> None:
        self.primary_key = primary_key
        self.class_col_name = class_col_name
        self.numerical_feature_names = numerical_feature_names
        self.categorical_feature_names = categorical_feature_names
        self.train_set = train_set
        self.valid_set = None
        self.test_set = test_set
        self.training_features = None
        self.validation_features = None
        self.testing_features = None
        self.train_features_preprocessed = None
        self.valid_features_preprocessed = None

        # Assert that at least one feature data type was passed
        assert (
            self.numerical_feature_names is not None
            and self.categorical_feature_names is not None
        ), "Names of numerical and/or categorical features must be provided. None was provided!"

    def select_relevant_columns(self) -> None:
        """Ensures specified numerical and categorical features exist in train and test
        set, and returns train and test sets with the selected columns."""

        # Update specified columns in case some columns were dropped from preprocessed dataset
        self.numerical_feature_names = [
            x
            for x in self.numerical_feature_names
            if x in self.train_set.columns and x in self.test_set.columns
        ]

        self.categorical_feature_names = [
            x
            for x in self.categorical_feature_names
            if x in self.train_set.columns and x in self.test_set.columns
        ]

        # Select features and class
        # Note: keep primary key for now as it's needed later to
        # create validation set.
        self.train_set = self.train_set[
            [self.primary_key]
            + self.numerical_feature_names
            + self.categorical_feature_names
            + [self.class_col_name]
        ]
        self.test_set = self.test_set[
            [self.primary_key]
            + self.numerical_feature_names
            + self.categorical_feature_names
            + [self.class_col_name]
        ]

    def enforce_data_types(self) -> None:
        """Enforces data types of numerical and categorical features.
        Note: if only numerical feature names are provided, all other features will
        be considered categorical and will be converted to string.
        """

        train_set_processor = DataPreprocessor(
            input_data=self.train_set,
            num_feature_names=self.numerical_feature_names,
            cat_feature_names=self.categorical_feature_names,
        )
        train_set_processor.specify_data_types()
        self.train_set = train_set_processor.get_preprocessed_data()

        if self.valid_set is not None:
            valid_set_processor = DataPreprocessor(
                input_data=self.valid_set,
                num_feature_names=self.numerical_feature_names,
                cat_feature_names=self.categorical_feature_names,
            )
            valid_set_processor.specify_data_types()
            self.valid_set = valid_set_processor.get_preprocessed_data()

        test_set_processor = DataPreprocessor(
            input_data=self.test_set,
            num_feature_names=self.numerical_feature_names,
            cat_feature_names=self.categorical_feature_names,
        )
        test_set_processor.specify_data_types()
        self.test_set = test_set_processor.get_preprocessed_data()

    def replace_nans_in_cat_features(
        self,
        nan_replacement: str = "Unspecified",
    ) -> None:
        """Replaces missing values with NaNs to allow converting them from
        float to integer.
        Note: the error (AttributeError: 'bool' object has no attribute 'transpose')
        is raised when transforming train set possibly because of pd.NA."""

        self.train_set[self.categorical_feature_names] = self.train_set[
            self.categorical_feature_names
        ].replace({pd.NA: nan_replacement})

        if self.valid_set is not None:
            self.valid_set[self.categorical_feature_names] = self.valid_set[
                self.categorical_feature_names
            ].replace({pd.NA: nan_replacement})

        self.test_set[self.categorical_feature_names] = self.test_set[
            self.categorical_feature_names
        ].replace({pd.NA: nan_replacement})

    def create_validation_set(
        self,
        split_type: Literal["time", "random"] = "random",
        train_set_size: float = 0.8,
        split_random_seed: Optional[int] = None,
        split_date_col_name: Optional[str] = None,
        split_cutoff_date: Optional[str] = None,
        split_date_col_format: str = "%Y-%m-%d %H:%M:%S",
    ) -> None:
        """Creates a validation set by splitting training set into training and
        validation sets randomly or based on time.
        Note: validation set will be used to select the best model.
        """

        if self.valid_set is not None:
            raise ValueError("Validation set already exists!")

        data_splitter = DataSplitter(
            dataset=self.train_set,
            primary_key_col_name=self.primary_key,
            class_col_name=self.class_col_name,
        )

        self.train_set, self.valid_set = data_splitter.split_dataset(
            split_type=split_type,
            train_set_size=train_set_size,
            split_random_seed=split_random_seed,
            split_date_col_name=split_date_col_name,
            split_cutoff_date=split_cutoff_date,
            split_date_col_format=split_date_col_format,
        )

    def extract_features(self, valid_set: Optional[pd.DataFrame] = None) -> None:
        """Separate features and class column of testing set. The validation
        set (valid_set) can be provided in this method if to wasn't provided
        already. If validation set is provided here, it will overwrite the
        validation set created by create_validation_set"""

        if valid_set is not None and self.valid_set is None:
            self.valid_set = valid_set
        elif valid_set is not None and self.valid_set is not None:
            raise ValueError(
                "Validation set was provided although it was already created!"
            )
        elif valid_set is None and self.valid_set is None:
            raise ValueError(
                "Validation set neither provided nor created using create_validation_set method!"
            )

        selected_features = (
            self.numerical_feature_names + self.categorical_feature_names
        )
        self.training_features = self.train_set[selected_features]
        self.validation_features = self.valid_set[selected_features]
        self.testing_features = self.test_set[selected_features]

    def encode_class_labels(
        self,
        pos_class_label: str,
    ) -> Union[pd.DataFrame, np.ndarray, LabelEncoder, int]:
        """Encode class labels into integers and returns encoded class labels
        of training, validation, and testing sets in addition to encoded positive
        class label and fitted encoder."""

        train_class = self.train_set[[self.class_col_name]]
        valid_class = self.valid_set[[self.class_col_name]]
        test_class = self.test_set[[self.class_col_name]]

        # Encode class labels
        fitted_class_encoder = LabelEncoder()
        encoded_train_class = fitted_class_encoder.fit_transform(ravel(train_class))
        encoded_valid_class = fitted_class_encoder.transform(ravel(valid_class))
        encoded_test_class = fitted_class_encoder.transform(ravel(test_class))

        # Get the encoded value of the positive class label
        enc_pos_class_label = fitted_class_encoder.transform([pos_class_label])[0]

        return (
            encoded_train_class,
            encoded_valid_class,
            encoded_test_class,
            enc_pos_class_label,
            fitted_class_encoder,
        )

    def create_data_transformation_pipeline(
        self,
        var_thresh_val: float = 0.05,
    ) -> Pipeline:
        """Creates a data transformation pipeline and fit it on training set."""

        train_set_transformer = DataPipelineCreator(
            num_features_imputer="median",
            num_features_scaler=MinMaxScaler(),
            cat_features_imputer="constant",
            cat_features_ohe_handle_unknown="infrequent_if_exist",
            cat_features_nans_replacement=np.nan,
        )
        (
            self.train_features_preprocessed,
            data_transformation_pipeline,
        ) = train_set_transformer.create_data_pipeline(
            input_features=self.training_features,
            num_feature_col_names=self.numerical_feature_names,
            cat_feature_col_names=self.categorical_feature_names,
            variance_threshold_val=var_thresh_val,
        )

        # Transform validation set using the same training set transformation
        # Note: these transfomred versions of train and validation sets will be
        # used inside the objective function to avoid applying data transformation
        # at each function call during optimization procedure.
        self.valid_features_preprocessed = data_transformation_pipeline.transform(
            self.validation_features
        )
        self.valid_features_preprocessed = pd.DataFrame(
            self.valid_features_preprocessed
        )
        self.valid_features_preprocessed.columns = list(
            self.train_features_preprocessed.columns
        )

        return data_transformation_pipeline

    def clean_up_feature_names(self) -> None:
        """Cleans up feature names of datasets to prevent errors that may arise
        because of special characters and returns clean feature names. This issue
        can be ebcountered after one-hot encoding where some category values with
        special characters can be become problematic column names.
        """

        # Remove special characters from column name to avoid error that LightGBM does
        # not support feature names with special characters
        self.training_features = self.training_features.rename(
            columns=lambda x: re.sub("[^A-Za-z0-9]+", "_", x), inplace=False
        )

        self.validation_features = self.validation_features.rename(
            columns=lambda x: re.sub("[^A-Za-z0-9]+", "_", x), inplace=False
        )

        self.testing_features = self.testing_features.rename(
            columns=lambda x: re.sub("[^A-Za-z0-9]+", "_", x), inplace=False
        )

        self.train_features_preprocessed = self.train_features_preprocessed.rename(
            columns=lambda x: re.sub("[^A-Za-z0-9]+", "_", x), inplace=False
        )
        self.valid_features_preprocessed = self.valid_features_preprocessed.rename(
            columns=lambda x: re.sub("[^A-Za-z0-9]+", "_", x), inplace=False
        )

    def get_train_set(self):
        """Returns the training set (features and class) when invoked."""
        return self.train_set.copy()

    def get_validation_set(self):
        """Returns the validation set (features and class) when invoked."""
        return self.valid_set.copy()

    def get_test_set(self):
        """Returns the testing set (features and class) when invoked."""
        return self.test_set.copy()

    def get_training_features(self):
        """Returns the training features when invoked."""
        return self.training_features.copy()

    def get_validation_features(self):
        """Returns the validation features when invoked."""
        return self.validation_features.copy()

    def get_testing_features(self):
        """Returns the testing features when invoked."""
        return self.testing_features.copy()

    def get_train_features_preprocessed(self):
        """Returns the transformed training set features when invoked."""
        return self.train_features_preprocessed.copy()

    def get_valid_features_preprocessed(self):
        """Returns the transformed validation set features when invoked."""
        return self.valid_features_preprocessed.copy()

    def get_feature_names(self) -> Union[list, list]:
        """Returns the numerical and categorical feature names of all data
        splits. During preprocessing some features might be dropped in they
        are near-zero variance. So this methods ensures updated feature
        names are returned."""
        return self.numerical_feature_names, self.categorical_feature_names
