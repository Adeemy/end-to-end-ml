"""
Data preprocessing and transformation classes.
"""

from datetime import datetime
from pathlib import PosixPath
from typing import Callable, Optional, Tuple

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
    """Stateless orchestrator for preprocessing steps.

    - Holds configuration (column lists, formats) but does NOT store dataset.
    - Each step is a method that accepts a DataFrame and returns a DataFrame.
    - Callers can extend pipeline with add_step (no class modification required).

    Attributes:
        primary_key_names (list): list of primary key column names.
        date_cols_names (list): list of date column names.
        datetime_cols_names (list): list of datetime column names.
        num_feature_names (list): list of numerical column names.
        cat_feature_names (list): list of categorical column names.
        desired_date_format (str): desired date format.
        desired_datetime_format (str): desired datetime format.
        _steps (list): list of preprocessing steps (callables).

    Usage examples:

    1) Default pipeline (convenient, minimal code):
        dp = DataPreprocessor(
            primary_key_names=['ID'],
            date_cols_names=['date_col'],
            datetime_cols_names=['dt_col'],
            num_feature_names=['BMI', 'Age'],
            cat_feature_names=['Sex', 'Smoker'],
        )
        processed = dp.run_preprocessing_pipeline(raw_df)

    2) Explicit pipeline composition (inject steps without modifying class):
        dp = DataPreprocessor(
            primary_key_names=['ID'],
            date_cols_names=['date_col'],
            num_feature_names=['BMI'],
            cat_feature_names=['Sex'],
            steps=[],  # start empty and inject desired steps
        )
        dp.add_step(dp.replace_blank_values_with_nan)
        dp.add_step(dp.replace_common_missing_values)
        dp.add_step(dp.specify_data_types)
        processed = dp.run(raw_df)

    3) Prepend/append steps (augment default pipeline):
        dp = DataPreprocessor(prepend_steps=[custom_pre_step], append_steps=[custom_post_step])

    4) Inspecting metadata produced by steps:
        # identify_cols_with_high_nans stores results in DataFrame attrs
        processed = dp.run_preprocessing_pipeline(raw_df)
        cols_to_drop = processed.attrs.get('cols_to_drop_due_to_nans')
        updated_cols = processed.attrs.get('updated_column_lists')

    Notes:
        - The instance stores only configuration; it never keeps the dataset.
        - Steps are pure-ish: accept a DataFrame and return a DataFrame, enabling easy testing.
    """

    def __init__(
        self,
        primary_key_names: Optional[list] = None,
        date_cols_names: Optional[list] = None,
        datetime_cols_names: Optional[list] = None,
        num_feature_names: Optional[list] = None,
        cat_feature_names: Optional[list] = None,
        desired_date_format: str = "%Y-%d-%m",
        desired_datetime_format: str = "%Y-%d-%m %H:%M:%S",
        steps: Optional[list] = None,
        prepend_steps: Optional[list] = None,
        append_steps: Optional[list] = None,
    ):
        # store configuration (treated as immutable by methods)
        self.primary_key_names = list(primary_key_names) if primary_key_names else []
        self.date_cols_names = list(date_cols_names) if date_cols_names else []
        self.datetime_cols_names = (
            list(datetime_cols_names) if datetime_cols_names else []
        )
        self.num_feature_names = list(num_feature_names) if num_feature_names else []
        self.cat_feature_names = list(cat_feature_names) if cat_feature_names else []
        self.desired_date_format = desired_date_format
        self.desired_datetime_format = desired_datetime_format

        # build pipeline: caller-provided steps override default; allow prepend/append
        if steps is not None:
            self._steps = list(steps)
        else:
            self._steps = [
                self.replace_blank_values_with_nan,
                self.replace_common_missing_values,
                self.check_duplicate_rows,
                self.remove_duplicates_by_primary_key,
                self.specify_data_types,
                self.identify_cols_with_high_nans,
            ]

        if prepend_steps:
            self._steps = list(prepend_steps) + self._steps
        if append_steps:
            self._steps = self._steps + list(append_steps)

    def add_step(self, step: Callable) -> None:
        """Appends a preprocessing step (callable df -> df).

        Args:
            step: callable that accepts a DataFrame and returns a DataFrame.
        """
        self._steps.append(step)

    def replace_blank_values_with_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replaces blank strings with np.nan.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with blank strings replaced by np.nan.
        """
        return df.replace(r"^\s*$", np.nan, regex=True)

    def replace_common_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replaces common missing-value tokens with np.nan.
        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with common missing-value tokens replaced by np.nan.
        """
        return df.replace(
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

    def check_duplicate_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validates duplicates; raises on failures (no mutation).
        Args:
            df: Input DataFrame.

        Returns:
            Original DataFrame if no duplicates found.

        Raises:
            ValueError: if duplicate rows found.
            ValueError: if duplicate primary key rows found.
        """

        duplicates_count = df.duplicated().sum()

        if not self.primary_key_names and duplicates_count > 0:
            raise ValueError(f"{duplicates_count} duplicate rows.")

        if self.primary_key_names:
            duplicates_by_id_count = df.duplicated(subset=self.primary_key_names).sum()
            if duplicates_by_id_count > 0:
                raise ValueError(
                    f"{duplicates_by_id_count} rows with duplicate {self.primary_key_names}."
                )

        return df

    def remove_duplicates_by_primary_key(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns DataFrame with duplicates removed by primary key (no mutation).

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with duplicates removed by primary key.

        Raises:
            ValueError: if no primary key columns provided.
        """

        if not self.primary_key_names:
            raise ValueError("No primary key column(s) provided!")
        dup_count = df.duplicated(subset=self.primary_key_names).sum()
        if dup_count > 0:
            logger.info(
                "%d rows with non-unique %s in input data.",
                dup_count,
                self.primary_key_names,
            )
            return df.drop_duplicates(subset=self.primary_key_names, keep="last")
        logger.info("No duplicate rows by %s in input data.", self.primary_key_names)
        return df

    def specify_data_types(
        self,
        df: pd.DataFrame,
        desired_date_format: Optional[str] = None,
        desired_datetime_format: Optional[str] = None,
    ) -> pd.DataFrame:
        """Casts columns to appropriate dtypes. Does not mutate class-level lists.

        Args:
            df: Input DataFrame.
            desired_date_format: Optional date format to use (overrides instance value).
            desired_datetime_format: Optional datetime format to use (overrides instance value).

        Returns:
            DataFrame with specified data types.
        """

        date_fmt = desired_date_format or self.desired_date_format
        datetime_fmt = desired_datetime_format or self.desired_datetime_format

        input_cols = df.columns.tolist()
        non_cat = (
            (self.date_cols_names or [])
            + (self.datetime_cols_names or [])
            + (self.num_feature_names or [])
        )
        cat_cols_local = self.cat_feature_names or [
            c for c in input_cols if c not in non_cat
        ]

        if self.date_cols_names:
            df = df.copy()
            df[self.date_cols_names] = df[self.date_cols_names].apply(
                pd.to_datetime, format=date_fmt, errors="coerce"
            )
            df[self.date_cols_names] = df[self.date_cols_names].replace(
                {np.nan: pd.NaT}
            )

        if self.datetime_cols_names:
            df[self.datetime_cols_names] = df[self.datetime_cols_names].apply(
                pd.to_datetime, format=datetime_fmt, errors="coerce"
            )
            df[self.datetime_cols_names] = df[self.datetime_cols_names].replace(
                {np.nan: pd.NaT}
            )

        if self.num_feature_names:
            df[self.num_feature_names] = df[self.num_feature_names].astype("float32")

        if cat_cols_local:
            logger.info(
                "Converting categorical columns to 'string' type: %s", cat_cols_local
            )
            df[cat_cols_local] = df[cat_cols_local].astype("string")

        return df

    def identify_cols_with_high_nans(
        self,
        df: pd.DataFrame,
        cols_to_exclude: Optional[list] = None,
        high_nans_percent_threshold: float = 0.3,
        update_cols_types: bool = True,
    ) -> pd.DataFrame:
        """Identifies high-NaN cols, attach metadata to df.attrs, and return df.

        - Does not mutate any instance-level column lists.
        - Puts 'cols_to_drop_due_to_nans' and optionally 'updated_column_lists' in df.attrs.

        Args:
            df: Input DataFrame.
            cols_to_exclude: Optional list of columns to exclude from dropping.
            high_nans_percent_threshold: Threshold proportion of NaNs to flag a column.
            update_cols_types: Whether to update column lists in df.attrs.

        Returns:
            DataFrame with metadata about columns to drop due to high NaNs.
        """

        nans_frac = (df.isna().sum() / df.shape[0]).sort_values(ascending=False)
        cols_to_drop = (
            list(nans_frac[nans_frac >= high_nans_percent_threshold].index)
            if not nans_frac.empty
            else []
        )

        if cols_to_exclude:
            cols_to_drop = [c for c in cols_to_drop if c not in cols_to_exclude]

        df.attrs["cols_to_drop_due_to_nans"] = cols_to_drop

        if update_cols_types:
            input_cols = [c for c in df.columns.tolist() if c not in cols_to_drop]
            date_cols = [c for c in (self.date_cols_names or []) if c in input_cols]
            datetime_cols = [
                c for c in (self.datetime_cols_names or []) if c in input_cols
            ]
            num_cols = [c for c in (self.num_feature_names or []) if c in input_cols]
            non_cat = date_cols + datetime_cols + num_cols
            cat_cols = [c for c in input_cols if c not in non_cat]
            df.attrs["updated_column_lists"] = {
                "date_cols": date_cols,
                "datetime_cols": datetime_cols,
                "num_cols": num_cols,
                "cat_cols": cat_cols,
            }

        return df

    def run_preprocessing_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executes preprocessing pipeline steps on a copy of df and return preprocessed DataFrame.

        Args:
            df: Input DataFrame.

        Returns:
            Preprocessed DataFrame.
        """
        processed = df.copy()
        for step in self._steps:
            # step may be a bound method or callable accepting df
            processed = step(processed)
        return processed


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
            logger.warning("Column %s doesn't exist in data.", col_name)

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
