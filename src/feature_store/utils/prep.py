""" 
Data preprocessing and transformation classes.
"""

from datetime import datetime
from typing import Literal, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataSplitter:
    """
    Splits dataset into train and test sets either randomly or based on time. It
    also verifies there is no overlapping samples (data leakage) in case the
    primary key column was not unique.

    Args:
        dataset (pd.Dataframe): dataset features without class.
        class_col_name (str): name of class column.
        split_type (str): type of split and it can be either "random" or "time".
        train_set_size (float): % of the training set (default: 0.8).
        split_random_seed (int): seed for random number generator for random split.
        split_date_col_name (str): name of date column to split dataset based on time.
        split_cutoff_date (date): cut-off date (data after this date are test set). For
            example, datetime.strptime("2023-15-11", "%Y-%d-%m").date().
        split_date_col_format (str): date format of date column used for split.

    Returns:
        train_set (pd.Dataframe): train set with class column.
        test_set (pd.Dataframe): test set with class column.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        primary_key_col_name: str,
        class_col_name: str,
    ):
        self._dataset = dataset.copy()
        self.primary_key_col_name = primary_key_col_name
        self.class_col_name = class_col_name

    def split_dataset(
        self,
        split_type: Literal["time", "random"] = "random",
        train_set_size: float = 0.8,
        split_random_seed: int = 123,
        split_date_col_name: str = None,
        split_cutoff_date: datetime.date = None,
        split_date_col_format: str = "%Y-%m-%d %H:%M:%S",
    ) -> Union[pd.DataFrame, pd.DataFrame]:
        if split_type == "random":
            (
                training_features,
                testing_features,
                training_class,
                testing_class,
            ) = train_test_split(
                self._dataset.drop([self.class_col_name], axis=1),
                self._dataset[[self.class_col_name]],
                train_size=train_set_size,
                stratify=self._dataset[self.class_col_name],
                random_state=split_random_seed,
            )

            # Add class labels to train and test sets
            train_set = pd.concat([training_features, training_class], axis=1)
            test_set = pd.concat([testing_features, testing_class], axis=1)

            print(
                f"Dataset was split randomly using {train_set_size} as train set size."
            )

        elif split_type == "time":
            train_set = self._dataset[
                pd.to_datetime(
                    self._dataset[split_date_col_name],
                    errors="coerce",
                    format=split_date_col_format,
                )
                < pd.to_datetime(split_cutoff_date, format=split_date_col_format)
            ]
            test_set = self._dataset[
                pd.to_datetime(
                    self._dataset[split_date_col_name],
                    errors="coerce",
                    format=split_date_col_format,
                )
                >= pd.to_datetime(split_cutoff_date, format=split_date_col_format)
            ]

            print(
                f"Dataset was split based on time using {split_cutoff_date} as cut-off date."
            )

        else:
            raise ValueError(
                f"split_type must be 'random' or 'time'. Got {split_type} instead!"
            )

        self.check_datasets_overlap(train_set, test_set)
        self.print_class_dist()

        return train_set, test_set

    def check_datasets_overlap(
        self, first_dataset: pd.DataFrame, second_dataset: pd.DataFrame
    ) -> None:
        """
        Checks if there is overlapping between two sets (e.g., train and test sets)
        based on a common primary key. It prints a message indicating whether there are
        samples that exist in both sets (i.e., data leakage) or not.

        Args:
            first_dataset (dataframe): it can be either training or testing set.
            second_dataset (dataframe): it can be either training or testing set.
            primary_key_col_name (str): name of the shared primary key column(s).
        """

        left_dataset = first_dataset.copy()
        right_dataset = second_dataset.copy()

        if left_dataset.shape[0] == 0 or right_dataset.shape[0] == 0:
            raise ValueError("\nEither dataset has a sample size of zero!\n")

        # Join datasets (inner join) on primary key to get overlapping samples
        left_dataset.set_index([self.primary_key_col_name], inplace=True)
        right_dataset.set_index([self.primary_key_col_name], inplace=True)
        overlap_samples = left_dataset.join(
            right_dataset, how="inner", lsuffix="_left", rsuffix="_right"
        )

        if len(overlap_samples) > 0:
            raise ValueError(
                f"\n{len(overlap_samples)} overlapping samples between train and test sets.\n"
            )
        else:
            print("\nNo overlapping samples between train and test sets.\n")

    def print_class_dist(
        self,
    ) -> Union[pd.Series, pd.Series]:
        """
        Prints class distribution (counts and percentages).
        """

        # Calculate class labels counts and percentages
        n_class_labels = self._dataset[self.class_col_name].value_counts()
        class_labels_proportions = round(
            100 * n_class_labels / self._dataset.shape[0], 2
        )

        # Print class proportions as dictionaries
        n_classes = n_class_labels.to_dict()
        class_proportions = class_labels_proportions.to_dict()

        print(f"Class label counts: {n_classes}\n")
        print(f"Class label proportions (%): {class_proportions}\n")

        return n_class_labels, class_labels_proportions


class DataPreprocessor:
    def __init__(
        self,
        input_data: pd.DataFrame,
        primary_key_names: list = None,
        date_cols_names: list = None,
        datetime_cols_names: list = None,
        num_feature_names: list = None,
        cat_feature_names: list = None,
    ):
        self._data = input_data.copy()
        self.primary_key_names = primary_key_names
        self.date_cols_names = date_cols_names
        self.datetime_cols_names = datetime_cols_names
        self.num_feature_names = num_feature_names
        self.cat_feature_names = cat_feature_names

        if self.date_cols_names is None:
            self.date_cols_names = []

        if self.datetime_cols_names is None:
            self.datetime_cols_names = []

        if self.num_feature_names is None:
            self.num_feature_names = []

        if self.cat_feature_names is None:
            self.cat_feature_names = []

        # Assert that at least one feature data type was passed
        assert (
            len(
                self.date_cols_names
                + self.datetime_cols_names
                + self.num_feature_names
                + self.cat_feature_names
            )
            > 0
        ), "At least one feature data type must be provided. None was provided!"

    def replace_blank_values_with_nan(self) -> pd.DataFrame:
        """Replaces blank cells like "" and white space with np.nan.
        This is particularly useful when preprocessing data sourced
        from csv files."""

        self._data.replace(r"^\s*$", np.nan, regex=True, inplace=True)

    def check_duplicate_rows(self) -> pd.DataFrame:
        """Checks if there are duplicate rows in dataset."""

        duplicates_count = self._data.duplicated().sum()
        if duplicates_count > 0:
            print(f"\nThere are {duplicates_count} duplicate rows in input data.")
        else:
            print("\nThere are no duplicate rows in input data.")

        # Check if there is duplicated primary_key_names
        if len(self.primary_key_names) > 0:
            duplicates_by_id_count = self._data.duplicated(
                subset=self.primary_key_names
            ).sum()
            if duplicates_by_id_count > 0:
                print(
                    f"\n{duplicates_by_id_count} rows with non-unique {self.primary_key_names} in input data."
                )
            else:
                print(f"\nNo duplicate rows by {self.primary_key_names} in input data.")

    def remove_duplicates_by_primary_key(self) -> pd.DataFrame:
        """Checks if there are duplicates in dataset by primary key column(s), which could
        be multiple columns."""

        # Check if there is duplicated primary_key_names and remove duplicate rows if any
        if len(self.primary_key_names) == 0:
            print("No primary key column(s) provided!")
        else:
            duplicates_by_id_count = self._data.duplicated(
                subset=self.primary_key_names
            ).sum()
            if duplicates_by_id_count > 0:
                print(
                    f"\n{duplicates_by_id_count} rows with the non-unique {self.primary_key_names} in input data."
                )
                self._data.drop_duplicates(
                    subset=self.primary_key_names, keep="last", inplace=True
                )
            else:
                print(
                    f"\nThere are no duplicate rows by {self.primary_key_names} in input data."
                )

    def specify_data_types(
        self,
        desired_date_format: str = "%Y-%d-%m",
        desired_datetime_format: str = "%Y-%d-%m %H:%M:%S",
    ) -> pd.DataFrame:
        """Enforces the specified data types of input dataset columns with
        their proper missing value indicator. If date, datetime, and numerical
        columns are not provided, all columns will be converted to categorical
        data type (categorical type).
        """

        # Categorical variables are all veriables that are not numerical or date
        input_data_vars_names = self._data.columns.tolist()
        non_cat_col_names = (
            self.date_cols_names + self.datetime_cols_names + self.num_feature_names
        )

        # Replace common missing values representations with with np.nan
        self._data = self._data.replace(
            {
                "": np.nan,
                "<NA>": np.nan,
                "null": np.nan,
                "?": np.nan,
                None: np.nan,
                "N/A": np.nan,
                "NAN": np.nan,
                "nan": np.nan,
                pd.NA: np.nan,
            }
        )

        # Identify categorical variables if not provided
        if len(self.cat_feature_names) == 0:
            self.cat_feature_names = [
                col for col in input_data_vars_names if col not in non_cat_col_names
            ]

        # Cast date columns
        if len(self.date_cols_names) > 0:
            self._data[self.date_cols_names] = self._data[self.date_cols_names].apply(
                pd.to_datetime, format=desired_date_format, errors="coerce"
            )

            self._data[self.date_cols_names] = self._data[self.date_cols_names].replace(
                {np.nan: pd.NaT}
            )

            print(f"Date columns:\n{self.date_cols_names}\n\n")

        # Cast datetime columns
        if len(self.datetime_cols_names) > 0:
            self._data[self.datetime_cols_names] = self._data[
                self.datetime_cols_names
            ].apply(pd.to_datetime, format=desired_datetime_format, errors="coerce")

            self._data[self.datetime_cols_names] = self._data[
                self.datetime_cols_names
            ].replace({np.nan: pd.NaT})

            print(f"Datetime columns:\n{self.datetime_cols_names}\n\n")

        # Cast numerical as float type
        if len(self.num_feature_names) > 0:
            self._data[self.num_feature_names] = self._data[
                self.num_feature_names
            ].astype("float32")

            print(f"Numerical columns:\n{self.num_feature_names}\n\n")

        # Cast categorical columns to object stype
        # Note: replacing NaNs after casting to object will convert columns to object data type.
        if len(self.cat_feature_names) > 0:
            print(
                "The following (categorical) columns will be converted to 'string' type.\n",
                self.cat_feature_names,
            )

            self._data[self.cat_feature_names] = self._data[
                self.cat_feature_names
            ].astype("string")

    def identify_cols_with_high_nans(
        self,
        cols_to_exclude: list = None,
        high_nans_percent_threshold: float = 0.3,
        update_cols_types: bool = True,
    ) -> list:
        """Identifies columns with missing values higher than high_nans_percent_threshold
        (default: 0.3). It will update list of categorical, continuous, date and datetime
        column names. It keeps columns specified col_names_to_exclude from exclusion if
        provided. update_cols_types was added to pass tests where updating column names
        by data types is not required."""

        # Identify columns with % of missing values > high_nans_percent_threshold
        nans_count = self._data.isna().sum().sort_values(ascending=False)
        nans_count = nans_count / self._data.shape[0]
        nans_count = nans_count.drop(
            nans_count[nans_count < high_nans_percent_threshold].index
        )

        if len(nans_count) > 0:
            cols_to_drop = list(nans_count.index)
        else:
            cols_to_drop = None

        # Exclude columns from removal due to high NaNs
        if cols_to_exclude is not None and cols_to_drop is not None:
            cols_to_drop = [col for col in cols_to_drop if col not in cols_to_exclude]

        # Update column names by data type
        if update_cols_types and cols_to_drop is not None:
            input_data_col_names = [
                col_name
                for col_name in self._data.columns.tolist()
                if col_name not in cols_to_drop
            ]

            self.date_cols_names = [
                col_name
                for col_name in self.date_cols_names
                if col_name not in cols_to_drop
            ]

            self.datetime_cols_names = [
                col_name
                for col_name in self.datetime_cols_names
                if col_name not in cols_to_drop
            ]

            self.num_feature_names = [
                col_name
                for col_name in self.num_feature_names
                if col_name not in cols_to_drop
            ]

            non_categorical_vars_names = (
                self.date_cols_names + self.datetime_cols_names + self.num_feature_names
            )
            self.cat_feature_names = [
                col
                for col in input_data_col_names
                if col not in non_categorical_vars_names
            ]

        return cols_to_drop

    def get_preprocessed_data(self):
        """Returns the preprocessed data when invoked."""
        return self._data.copy()


class DataTransformer:
    def __init__(
        self,
        preprocessed_data: pd.DataFrame,
        primary_key_names: list = None,
        date_cols_names: list = None,
        datetime_cols_names: list = None,
        num_feature_names: list = None,
        cat_feature_names: list = None,
    ):
        self.preprocessed_data = preprocessed_data.copy()
        self.primary_key_names = primary_key_names
        self.date_cols_names = date_cols_names
        self.datetime_cols_names = datetime_cols_names
        self.num_feature_names = num_feature_names
        self.cat_feature_names = cat_feature_names

        if self.date_cols_names is None:
            self.date_cols_names = []

        if self.datetime_cols_names is None:
            self.datetime_cols_names = []

        if self.num_feature_names is None:
            self.num_feature_names = []

        if self.cat_feature_names is None:
            self.cat_feature_names = []

    def map_categorical_features(self):
        """Maps categorical features to expressive values."""

        if "GenHlth" in self.preprocessed_data.columns:
            self.preprocessed_data.loc[:, "GenHlth"] = self.preprocessed_data.loc[
                :, "GenHlth"
            ].replace(
                {
                    "1": "Poor",
                    "2": "Fair",
                    "3": "Good",
                    "4": "Very Good",
                    "5": "Excellent",
                }
            )

        if "Education" in self.preprocessed_data.columns:
            self.preprocessed_data.loc[:, "Education"] = self.preprocessed_data.loc[
                :, "Education"
            ].replace(
                {
                    "1": "Never Attended School",
                    "2": "Elementary",
                    "3": "High School",
                    "4": "Some College Degree",
                    "5": "Advanced Degree",
                }
            )

        if "Age" in self.preprocessed_data.columns:
            self.preprocessed_data.loc[:, "Age"] = self.preprocessed_data.loc[
                :, "Age"
            ].replace(
                {
                    "1": "18 to 24",
                    "2": "25 to 29",
                    "3": "30 to 34",
                    "4": "35 to 39",
                    "5": "40 to 44",
                    "6": "45 to 49",
                    "7": "50 to 54",
                    "8": "55 to 59",
                    "9": "60 to 64",
                    "10": "65 to 69",
                    "11": "70 to 74",
                    "12": "75 to 79",
                    "13": "80 or older",
                }
            )

    def rename_class_labels(self, class_col_name: str):
        """Rename class labels to 'Diabetic' or 'Non-Diabetic'."""

        if class_col_name in self.preprocessed_data.columns:
            self.preprocessed_data.loc[:, class_col_name] = (
                self.preprocessed_data.loc[:, class_col_name]
                .astype("string")
                .replace({"0": "Non-Diabetic", "1": "Diabetic"})
            )
        else:
            raise ValueError(f"Class column {class_col_name} doesn't exist in data.")
