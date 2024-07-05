"""
Contains helper functions for data preprocessing and data split used
in the training script (train.py)
"""

import re
from typing import Callable, Literal, Optional, Union

import numpy as np
import pandas as pd
from numpy import ravel
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from src.feature_store.utils.prep import DataPreprocessor, DataSplitter
from src.utils.logger import get_console_logger

##########################################################
# Get the logger objects by name
logger = get_console_logger("data_logger")


class DataPipelineCreator:
    """Creates a data preprocessing pipeline using sklearn.

    Attributes:
        num_features_imputer (str): strategy to impute missing values in numerical features. Defaults to "median".
        num_features_scaler (Optional[Union[RobustScaler, StandardScaler, MinMaxScaler]]): scaler to scale
            numerical features. Defaults to RobustScaler().
        cat_features_imputer (str): strategy to impute missing values in categorical features. Defaults to "constant".
        cat_features_ohe_handle_unknown (str): strategy to handle unknown categories in categorical features. Defaults
            to "infrequent_if_exist".
        cat_features_nans_replacement (str): value to replace NaNs in categorical features. Defaults to np.nan.
    """

    def __init__(
        self,
        num_features_imputer: Literal[
            "mean", "median", "most_frequent", "constant"
        ] = "median",
        num_features_scaler: Optional[
            Union[RobustScaler, StandardScaler, MinMaxScaler]
        ] = RobustScaler(),
        cat_features_imputer: Literal["most_frequent", "constant"] = "constant",
        cat_features_ohe_handle_unknown: Literal[
            "error", "ignore", "infrequent_if_exist"
        ] = "infrequent_if_exist",
        cat_features_nans_replacement: str = np.nan,
    ):
        """Initializes an instance for data preprocessing pipeline using sklearn."""

        self.num_features_imputer = num_features_imputer
        self.num_features_scaler = num_features_scaler
        self.cat_features_imputer = cat_features_imputer
        self.cat_features_ohe_handle_unknown = cat_features_ohe_handle_unknown
        self.cat_features_nans_replacement = cat_features_nans_replacement

    def create_num_features_transformer(
        self,
    ) -> Pipeline:
        """Creates sklearn pipeline for numerical features.

        Returns:
            num_transformer (Pipeline): sklearn pipeline for numerical features.
        """

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
        """Creates sklearn pipeline for categorical features.

        Attributes:
            cat_transformer (Pipeline): sklearn pipeline for categorical features.
        """
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

    def extract_col_names_after_preprocessing(
        self,
        num_feat_col_names: list,
        cat_feat_col_names: list,
        selector: VarianceThreshold,
        data_pipeline: Pipeline,
    ) -> list:
        """Extracts one-hot encoded feature names from the data transformation pipeline.
        If no categorical features are provided, it returns numerical feature names.

        Note: if categorical features are provided the data transformation pipeline must
        have a 'preprocessor' step that includes a 'onehot_encoder' step.

        Args:
            num_feat_col_names (list): list of numerical feature names.
            cat_feat_col_names (list): list of categorical feature names.
            selector (VarianceThreshold): variance threshold selector.
            data_pipeline (Pipeline): data transformation pipeline.

        Returns:
            col_names: list of one-hot encoded feature names.
        """

        # The transformers_ attribute is only available after fitting
        # ColumnTransformer
        col_names = []
        if len(cat_feat_col_names) > 0:
            col_names = num_feat_col_names + list(
                data_pipeline.named_steps["preprocessor"]
                .transformers_[1][1]
                .named_steps["onehot_encoder"]
                .get_feature_names_out(cat_feat_col_names)
            )
        else:
            col_names = num_feat_col_names

        # Get feature names that were selected by selector step
        col_names = [i for (i, v) in zip(col_names, list(selector.get_support())) if v]

        return col_names

    ###########################################################
    def create_data_pipeline(
        self,
        input_features: pd.DataFrame,
        num_feature_col_names: Optional[list] = None,
        cat_feature_col_names: Optional[list] = None,
        variance_threshold_val: float = 0.05,
    ) -> Union[pd.DataFrame, Pipeline]:
        """Creates a data transformation pipeline using sklearn and returns
        transformed data and the pipeline.

        Args:
            input_features (pd.DataFrame): training set features.
            num_feature_col_names (Optional[list], optional): numerical feature names. Defaults to None.
            cat_feature_col_names (Optional[list], optional): categorical feature names. Defaults to None.
            variance_threshold_val (float, optional): variance threshold value. Defaults to 0.05.

        Raises:
            AssertionError: if no numerical or categorical features are specified.
            ValueError: if an error occurs while extracting feature names after preprocessing.
        """

        features_set = input_features.copy()

        # Set column names to [] if None was provided
        num_feature_col_names = (
            [] if num_feature_col_names is None else num_feature_col_names
        )
        cat_feature_col_names = (
            [] if cat_feature_col_names is None else cat_feature_col_names
        )

        assert (
            len(num_feature_col_names + cat_feature_col_names) > 0
        ), "At least one numerical or categorical feature name must be specified!"

        # Create a numerical features transformer
        if len(num_feature_col_names) > 0:
            numeric_transformer = self.create_num_features_transformer()
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
        try:
            transformed_data.columns = self.extract_col_names_after_preprocessing(
                num_feat_col_names=num_feature_col_names,
                cat_feat_col_names=cat_feature_col_names,
                selector=selector,
                data_pipeline=data_pipeline,
            )
        except ValueError as e:
            raise ValueError(
                f"An error occurred while extracting feature names after preprocessing: {e}"
            ) from e

        logger.info("Data transformation pipeline created successfully.")

        return transformed_data, data_pipeline


class TrainingDataPrep:
    """Prepares data for training during data split. It can be used to create the
    three data splits: training, validation, and testing, and apply minimal preprocessing
    steps that doesn't cause data leakage. It can also be used in training script to ensure
    all data splits have proper data types espcially when importing data from csv files. The
    class can also be used to fit label encoder on training class and use the fitted encoder
    to transform the validation and testing set class labels. Although there is no dependencies
    between methods, this class can be further improved in future refactoring efforts.

    Attributes:
        train_set (pd.DataFrame): training set.
        valid_set (Optional[pd.DataFrame]): validation set.
        test_set (pd.DataFrame): testing set.
        primary_key (str): name of primary key column.
        class_col_name (str): name of class column.
        numerical_feature_names (Optional[list]): list of numerical feature names.
        categorical_feature_names (Optional[list]): list of categorical feature names.
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
        """Creates a TrainingDataPrep instance.

        Args:
            train_set (pd.DataFrame): training set.
            test_set (pd.DataFrame): testing set.
            primary_key (str): name of primary key column.
            class_col_name (str): name of class column.
            numerical_feature_names (Optional[list], optional): list of numerical feature names. Defaults to None.
            categorical_feature_names (Optional[list], optional): list of categorical feature names. Defaults to None.

            The following are attributes that are set to None by default and can be set later:

            training_features (Optional[pd.DataFrame], optional): training set features. Defaults to None.
            valid_set (Optional[pd.DataFrame], optional): validation set. Defaults to None but can be created using
                create_validation_set method.
            validation_features (Optional[pd.DataFrame], optional): validation set features. Defaults to None.
            testing_features (Optional[pd.DataFrame], optional): testing set features. Defaults to None.
            train_features_preprocessed (Optional[pd.DataFrame], optional): transformed training set features. Defaults to None.
            valid_features_preprocessed (Optional[pd.DataFrame], optional): transformed validation set features. Defaults to None.

        Raises:
            AssertionError: if no numerical or categorical features are specified.
        """

        self.train_set = train_set
        self.test_set = test_set
        self.primary_key = primary_key
        self.class_col_name = class_col_name
        self.numerical_feature_names = numerical_feature_names
        self.categorical_feature_names = categorical_feature_names
        self.training_features = None
        self.valid_set = None
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
        set, and returns train and test sets with the selected columns.
        """

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

        # This allows this method to be independent of create_validation_set method
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

        Note: validation set will be used to select the best model. An error should be
        raised if this method is called when validation set is already provided to
        prevent overwritting the provided validation set unintentionally.

        Args:
            split_type (Literal["time", "random"], optional): type of split. Defaults to "random".
            train_set_size (float, optional): size of training set. Defaults to 0.8.
            split_random_seed (Optional[int], optional): random seed for reproducibility. Defaults to None.
            split_date_col_name (Optional[str], optional): name of date column. Defaults to None.
            split_cutoff_date (Optional[str], optional): date to split on. Defaults to None.
            split_date_col_format (str, optional): date column format. Defaults to "%Y-%m-%d %H:%M:%S".


        Raises:
            ValueError: if validation set already exists.
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
        """Separates features and class column of testing set. The validation
        set (valid_set) can be provided in this method if to wasn't provided
        already. If validation set is provided here, it will overwrite the
        validation set created by create_validation_set.

        Args:
            valid_set (Optional[pd.DataFrame], optional): validation set. Defaults to None.

        Raises:
            ValueError: if validation set is provided although it was already created.
            ValueError: if validation set is neither provided nor created using create_validation_set method.
        """

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
        class label and fitted encoder.

        Args:
            pos_class_label (str): positive class label.

        Returns:
            tuple: tuple containing:
            encoded_train_class (np.ndarray): encoded training set class labels.
            encoded_valid_class (np.ndarray): encoded validation set class labels.
            encoded_test_class (np.ndarray): encoded testing set class labels.
            enc_pos_class_label (int): encoded positive class label.
            fitted_class_encoder (LabelEncoder): fitted class encoder.
        """

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
        num_features_imputer: Literal[
            "mean", "median", "most_frequent", "constant"
        ] = "median",
        num_features_scaler: Optional[Union[Callable, None]] = RobustScaler(),
        cat_features_imputer: Literal["most_frequent", "constant"] = "constant",
        cat_features_ohe_handle_unknown: Literal[
            "error", "ignore", "infrequent_if_exist"
        ] = "infrequent_if_exist",
        cat_features_nans_replacement: str = np.nan,
        var_thresh_val: float = 0.05,
    ) -> Pipeline:
        """Creates a data transformation pipeline and fit it on training set.

        Args:
            num_features_imputer (Literal["mean", "median", "most_frequent", "constant"], optional): strategy to
                impute missing values in numerical features. Defaults to "median".
            num_features_scaler (Optional[Union[Callable, None]], optional): scaler to scale numerical features.
                Defaults to RobustScaler().
            cat_features_imputer (Literal["most_frequent", "constant"], optional): strategy to impute missing values
                in categorical features. Defaults to "constant".
            cat_features_ohe_handle_unknown (Literal["error", "ignore", "infrequent_if_exist"], optional): strategy
                to handle unknown categories in categorical features. Defaults to "infrequent_if_exist".
            cat_features_nans_replacement (str, optional): value to replace NaNs in categorical features. Defaults to np.nan.
            var_thresh_val (float, optional): variance threshold value. Defaults to 0.05.

        Returns:
            data_transformation_pipeline (Pipeline): data transformation pipeline.
        """

        train_set_transformer = DataPipelineCreator(
            num_features_imputer=num_features_imputer,
            num_features_scaler=num_features_scaler,
            cat_features_imputer=cat_features_imputer,
            cat_features_ohe_handle_unknown=cat_features_ohe_handle_unknown,
            cat_features_nans_replacement=cat_features_nans_replacement,
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

        Note: this method should be called after data transformation pipeline is
        created and applied on training, validation, and testing sets.
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

    def get_feature_names(self) -> Union[list, list]:
        """Returns the numerical and categorical feature names of all data
        splits. During preprocessing some features might be dropped in they
        are near-zero variance. So this methods ensures updated feature
        names are returned.

        Returns:
            tuple: tuple containing:
                - numerical_feature_names (list): list of numerical feature names.
                - categorical_feature_names (list): list of categorical feature names.
        """
        return self.numerical_feature_names, self.categorical_feature_names
