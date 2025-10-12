"""
Data preprocessing and transformation classes.
"""

import warnings
from datetime import datetime
from pathlib import PosixPath
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_console_logger

module_name: str = PosixPath(__file__).stem
logger = get_console_logger(module_name)


class SplitStrategy:
    """Abstract base class for dataset splitting strategies. Subclasses must implement
    the split method."""

    def split(
        self,
        dataset: pd.DataFrame,  # pylint: disable=unused-argument
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits the dataset into train and test sets.

        Args:
            dataset: The dataset to split.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Train and test datasets.
        """
        raise NotImplementedError("Subclasses must implement the split method.")


class DataSplitter:
    """Splits dataset into two disjoint sets, train and test sets, either randomly
    or based on time (a cut-off date must be provided).

    Attributes:
        dataset (pd.DataFrame): input dataset.
        primary_key_col_name (str): name of the primary key column(s).
        class_col_name (str): name of the class column.

    Usages:
        data_splitter = DataSplitter(
            dataset=raw_dataset,
            primary_key_col_name=data_config.pk_col_name,
            class_col_name=data_config.class_col_name,
        )

        # Random split
        split_strategy = RandomSplitStrategy(
            class_col_name="class",
            train_set_size=0.8,
            random_seed=data_config.random_seed,
        )

        # Time-based split
        split_strategy = TimeBasedSplitStrategy(
            date_col_name=data_config.event_timestamp_col_name,
            cutoff_date=datetime.date(2020, 1, 1),
            date_format="%Y-%m-%d %H:%M:%S",
        )

        train_set, test_set = data_splitter.split_dataset(split_strategy)
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
        split_strategy: SplitStrategy,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits dataset into two disjoint sets using the provided strategy.

        Args:
            split_strategy: Strategy to use for splitting the dataset.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Train and test datasets.
        """
        train_set, test_set = split_strategy.split(self._dataset)
        self._check_datasets_overlap(train_set, test_set)
        self._log_class_distribution()
        return train_set, test_set

    def _check_datasets_overlap(
        self, first_dataset: pd.DataFrame, second_dataset: pd.DataFrame
    ) -> None:
        """Raises an error if there are overlapping samples in two sets using
        primary key column, i.e., two sets are not disjoint causing data leakage.

        Args:
            first_dataset: Training or testing set.
            second_dataset: Training or testing set.

        Raises:
            ValueError: If there are overlapping samples in both sets.
        """
        left_dataset = first_dataset.set_index(self.primary_key_col_name)
        right_dataset = second_dataset.set_index(self.primary_key_col_name)
        overlap_samples = left_dataset.index.intersection(right_dataset.index)

        if not overlap_samples.empty:
            raise ValueError(
                f"{len(overlap_samples)} overlapping samples found in both sets."
            )
        logger.info("The provided datasets are disjoint.")

    def _log_class_distribution(self) -> None:
        """Logs class distribution (counts and percentages) of the dataset."""
        n_class_labels = self._dataset[self.class_col_name].value_counts()
        class_labels_proportions = round(
            100 * n_class_labels / self._dataset.shape[0], 2
        )

        logger.info("Class label counts: %s", n_class_labels.to_dict())
        logger.info(
            "Class label proportions (%%): %s", class_labels_proportions.to_dict()
        )


class RandomSplitStrategy(SplitStrategy):
    """Random splitting strategy. It splits the dataset randomly into train and test sets.
    It uses stratified sampling to ensure that the class distribution is preserved
    in both sets.

    Attributes:
        train_set_size (float): proportion of the dataset to include in the train set.
        random_seed (int): random seed for reproducibility.
    """

    def __init__(
        self, class_col_name: str, train_set_size: float = 0.8, random_seed: int = 123
    ):
        self.class_col_name = class_col_name
        self.train_set_size = train_set_size
        self.random_seed = random_seed

    def validate_inputs(self, dataset: pd.DataFrame) -> None:
        """Validates the inputs for the random split strategy.

        Args:
            dataset: The dataset to validate.

        Raises:
            ValueError: if class_col_name doesn't exist in dataset.
            ValueError: if class_col_name has only one class label.
            ValueError: if class_col_name contains missing values.
            ValueError: if train_set_size is not between 0 and 1.
            ValueError: if random_seed is not an integer.
        """

        if self.class_col_name not in dataset.columns:
            raise ValueError(f"{self.class_col_name} doesn't exist in dataset.")

        if dataset[self.class_col_name].nunique() < 2:
            raise ValueError(f"{self.class_col_name} must have at least two classes.")

        if dataset[self.class_col_name].isna().sum() > 0:
            raise ValueError(f"{self.class_col_name} contains missing values.")

        if not (0 < self.train_set_size < 1):
            raise ValueError("train_set_size must be between 0 and 1.")

        if not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be an integer.")

    def split(
        self,
        dataset: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits the dataset randomly into train and test sets. It validates the inputs before
        splitting the dataset.

        Args:
            dataset: The dataset to split.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Train and test datasets.
        """

        # Validate inputs
        self.validate_inputs(dataset)

        train_features, test_features, train_labels, test_labels = train_test_split(
            dataset.drop(columns=[self.class_col_name]),
            dataset[[self.class_col_name]],
            train_size=self.train_set_size,
            stratify=dataset[self.class_col_name],
            random_state=self.random_seed,
        )
        train_set = pd.concat([train_features, train_labels], axis=1)
        test_set = pd.concat([test_features, test_labels], axis=1)

        logger.info(
            "Dataset split randomly with train set size: %.2f", self.train_set_size
        )
        return train_set, test_set


class TimeBasedSplitStrategy(SplitStrategy):
    """Time-based splitting strategy. It splits the dataset into train and test sets
    based on a cut-off date. All samples with a date before the cut-off date are
    assigned to the train set, and all samples with a date on or after the cut-off
    date are assigned to the test set.

    Attributes:
        date_col_name: name of the date column.
        cutoff_date: cut-off date for splitting the dataset.
        date_format: format of the date column.
    """

    def __init__(
        self,
        date_col_name: str,
        cutoff_date: datetime.date,
        date_format: str = "%Y-%m-%d %H:%M:%S",
    ):
        self.date_col_name = date_col_name
        self.cutoff_date = cutoff_date
        self.date_format = date_format

    def _validate_inputs(self, dataset: pd.DataFrame) -> None:
        """Validates the inputs for the time-based split strategy.

        Args:
            dataset: The dataset to validate.

        Raises:
            ValueError: if date_col_name doesn't exist in dataset.
            ValueError: if date_col_name is not in datetime format.
            ValueError: if cutoff_date is not in the range of date_col_name.
            ValueError: if date_col_name contains missing values.
        """
        if self.date_col_name not in dataset.columns:
            raise ValueError(f"{self.date_col_name} doesn't exist in dataset.")

        if not np.issubdtype(dataset[self.date_col_name].dtype, np.datetime64):
            raise ValueError(f"{self.date_col_name} must be in datetime format.")

        if dataset[self.date_col_name].isna().sum() > 0:
            raise ValueError(f"{self.date_col_name} contains missing values.")

        min_date = dataset[self.date_col_name].min()
        max_date = dataset[self.date_col_name].max()

        if not (min_date <= pd.to_datetime(self.cutoff_date) <= max_date):
            raise ValueError(f"cutoff_date must be between {min_date} and {max_date}.")

    def split(
        self,
        dataset: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits the dataset based on a cutoff date. It validates the inputs before
        splitting the dataset.

        Args:
            dataset: The dataset to split based on date column.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Train and test datasets.
        """

        # Validate inputs
        self._validate_inputs(dataset)

        train_set = dataset[
            pd.to_datetime(dataset[self.date_col_name], format=self.date_format)
            < pd.to_datetime(self.cutoff_date, format=self.date_format)
        ]
        test_set = dataset[
            pd.to_datetime(dataset[self.date_col_name], format=self.date_format)
            >= pd.to_datetime(self.cutoff_date, format=self.date_format)
        ]

        logger.info(
            "Dataset split based on time with cutoff date: %s", self.cutoff_date
        )
        return train_set, test_set


class DataPreprocessor:
    """Preprocesses input data by replacing blank values with np.nan, checking
    for duplicate rows, removing duplicate rows by primary key, specifying data types,
    identifying columns with high % of missing values, and returning the preprocessed
    data when invoked.

    Attributes:
        input_data (pd.DataFrame): input data.
        primary_key_names (list): list of primary key column names.
        date_cols_names (list): list of date column names.
        datetime_cols_names (list): list of datetime column names.
        num_feature_names (list): list of numerical column names.
        cat_feature_names (list): list of categorical column names.
    """

    def __init__(
        self,
        input_data: pd.DataFrame,
        primary_key_names: Optional[list] = None,
        date_cols_names: Optional[list] = None,
        datetime_cols_names: Optional[list] = None,
        num_feature_names: Optional[list] = None,
        cat_feature_names: Optional[list] = None,
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

    def replace_blank_values_with_nan(self) -> None:
        """Replaces blank values with np.nan. It is useful when reading data from
        csv files where blank values are represented by empty strings.
        """

        self._data.replace(r"^\s*$", np.nan, regex=True, inplace=True)

    def check_duplicate_rows(self) -> None:
        """Checks if there are duplicate rows in dataset. It returns the number of
        duplicate rows if any.
        """

        duplicates_count = self._data.duplicated().sum()
        if self.primary_key_names is None and duplicates_count > 0:
            raise ValueError(f"\n{duplicates_count} duplicate rows.")

        # Check if there is duplicated primary_key_names
        if self.primary_key_names is not None:
            duplicates_by_id_count = self._data.duplicated(
                subset=self.primary_key_names
            ).sum()
            if duplicates_by_id_count > 0:
                raise ValueError(
                    f"\n{duplicates_by_id_count} rows with duplicate {self.primary_key_names}."
                )

    def remove_duplicates_by_primary_key(self) -> None:
        """Removes duplicate rows by primary key. It returns the number of duplicate
        rows and keeps the last duplicate row if any. It is useful when there are
        duplicate rows by primary key in the dataset.
        """

        # Check if there is duplicated primary_key_names and remove duplicate rows if any
        if len(self.primary_key_names) == 0:
            raise ValueError("No primary key column(s) provided!")
        else:
            duplicates_by_id_count = self._data.duplicated(
                subset=self.primary_key_names
            ).sum()
            if duplicates_by_id_count > 0:
                logger.info(
                    """\n{duplicates_by_id_count} rows with the non-unique
                    %s in input data.""",
                    self.primary_key_names,
                )
                self._data.drop_duplicates(
                    subset=self.primary_key_names, keep="last", inplace=True
                )
            else:
                logger.info(
                    "\nNo duplicate rows by %s in input data.", self.primary_key_names
                )

    def replace_common_missing_values(self) -> None:
        """Replaces common missing values with np.nan. It is useful when reading data
        from csv files where common missing values are represented by different strings.
        """

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

    def specify_data_types(
        self,
        desired_date_format: str = "%Y-%d-%m",
        desired_datetime_format: str = "%Y-%d-%m %H:%M:%S",
    ) -> None:
        """Specifies data types for date, datetime, numerical and categorical columns
        with their proper missing value indicator. If date, datetime, and numerical
        columns are not provided, all columns will be converted to categorical
        data type (categorical type).

        Args:
            desired_date_format (str): desired date format.
            desired_datetime_format (str): desired datetime format.
        """

        # Categorical variables are all veriables that are not numerical or date
        input_data_vars_names = self._data.columns.tolist()
        non_cat_col_names = (
            self.date_cols_names + self.datetime_cols_names + self.num_feature_names
        )

        # Replace common missing values with np.nan
        self.replace_common_missing_values()

        # Identify categorical variables if not provided
        if len(self.cat_feature_names) == 0:
            self.cat_feature_names = [
                col for col in input_data_vars_names if col not in non_cat_col_names
            ]

        # Cast columns to proper data types
        if len(self.date_cols_names) > 0:
            self._data[self.date_cols_names] = self._data[self.date_cols_names].apply(
                pd.to_datetime, format=desired_date_format, errors="coerce"
            )

            self._data[self.date_cols_names] = self._data[self.date_cols_names].replace(
                {np.nan: pd.NaT}
            )

        if len(self.datetime_cols_names) > 0:
            self._data[self.datetime_cols_names] = self._data[
                self.datetime_cols_names
            ].apply(pd.to_datetime, format=desired_datetime_format, errors="coerce")

            self._data[self.datetime_cols_names] = self._data[
                self.datetime_cols_names
            ].replace({np.nan: pd.NaT})

        if len(self.num_feature_names) > 0:
            self._data[self.num_feature_names] = self._data[
                self.num_feature_names
            ].astype("float32")

        if len(self.cat_feature_names) > 0:
            logger.info(
                """The following categorical columns will be converted to 'string'
                type.\n %s""",
                self.cat_feature_names,
            )

            self._data[self.cat_feature_names] = self._data[
                self.cat_feature_names
            ].astype("string")

    def identify_cols_with_high_nans(
        self,
        cols_to_exclude: Optional[list] = None,
        high_nans_percent_threshold: float = 0.3,
        update_cols_types: bool = True,
    ) -> list:
        """Identifies columns with missing values higher than high_nans_percent_threshold
        (default: 0.3). It will update list of categorical, continuous, date and datetime
        column names. It keeps columns specified col_names_to_exclude from exclusion if
        provided. update_cols_types was added to pass tests where updating column names
        by data types is not required.

        Args:
            cols_to_exclude (list): list of columns to exclude from removal.
            high_nans_percent_threshold (float): threshold for high % of missing values.
            update_cols_types (bool): whether to update column names by data type.

        Returns:
            cols_to_drop (list): list of columns with high % of missing values.
        """

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

    def get_preprocessed_data(self) -> pd.DataFrame:
        """Returns the preprocessed data when invoked.

        Returns:
            preprocessed_data (pd.DataFrame): preprocessed data.
        """
        return self._data.copy()


class DataTransformer:
    """Transforms input data by mapping categorical features to expressive values
    and renaming class labels to 'Diabetic' or 'Non-Diabetic'. It returns the
    transformed data when invoked. It is useful when the input data is not
    in the same format as the training data.

    Attributes:
        preprocessed_data (pd.DataFrame): preprocessed data.
        primary_key_names (list): list of primary key column names.
        date_cols_names (list): list of date column names.
        datetime_cols_names (list): list of datetime column names.
        num_feature_names (list): list of numerical column names.
        cat_feature_names (list): list of categorical column names.
    """

    def __init__(
        self,
        preprocessed_data: pd.DataFrame,
        primary_key_names: Optional[list] = None,
        date_cols_names: Optional[list] = None,
        datetime_cols_names: Optional[list] = None,
        num_feature_names: Optional[list] = None,
        cat_feature_names: Optional[list] = None,
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

    def map_categorical_features(self, col_name: str, mapping_values: dict) -> None:
        """Maps categorical features to expressive values. It applies only to
        columns that exists in the preprocessed data. Otherwise, it return a
        warning message.

        Args:
            col_name (str): name of the categorical column.
            mapping_values (dict): dictionary of mapping values.
        """

        if col_name in self.preprocessed_data.columns:
            self.preprocessed_data.loc[:, col_name] = self.preprocessed_data.loc[
                :, col_name
            ].replace(mapping_values)
        else:
            warnings.warn(f"Column {col_name} doesn't exist in data.")

    def map_class_labels(self, class_col_name: str, mapping_values: dict) -> None:
        """Maps class labels to expressive names: 'Diabetic' or 'Non-Diabetic'.

        Args:
            class_col_name (str): name of the class column.
            mapping_values (dict): dictionary of mapping values.

        Raises:
            ValueError: if class column doesn't exist in data.
        """

        if class_col_name in self.preprocessed_data.columns:
            self.preprocessed_data[class_col_name] = (
                self.preprocessed_data[class_col_name]
                .astype("string")
                .replace(mapping_values)
            )
        else:
            raise ValueError(f"Class column {class_col_name} doesn't exist in data.")
