"""
Test functions for the DataSplitter class in the feature store
module src/feature/utils/prep.py.
"""

import re
from datetime import date

import pandas as pd
import pytest

from src.feature.utils.prep import (
    DataSplitter,
    RandomSplitStrategy,
    SplitStrategy,
    TimeBasedSplitStrategy,
)


@pytest.fixture
def split_dataset_data():
    """Fixture to provide sample data for testing dataset splitting."""
    return (
        "class_col",
        pd.DataFrame(
            {
                "feature_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature_2": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
                "feature_3": [
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                ],
                "date_col": [
                    "2021-01-01",
                    "2021-01-02",
                    "2021-01-03",
                    "2021-01-04",
                    "2021-01-05",
                    "2021-01-06",
                    "2021-01-07",
                    "2021-01-08",
                    "2021-01-09",
                    "2021-01-10",
                ],
                "class_col": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }
        ),
    )


def test_split_on_abstract_base_raises_not_implemented():
    """Calling SplitStrategy.split on the abstract base should raise NotImplementedError."""
    df = pd.DataFrame({"x": [1, 2, 3]})
    base = SplitStrategy()
    with pytest.raises(
        NotImplementedError, match="Subclasses must implement the split method."
    ):
        base.split(df)


def test_validate_inputs_raises_when_class_col_missing():
    df = pd.DataFrame({"a": [1, 2, 3]})
    strat = RandomSplitStrategy(class_col_name="label")
    with pytest.raises(ValueError, match=re.escape("label doesn't exist in dataset.")):
        strat.validate_inputs(df)


def test_validate_inputs_raises_when_single_class():
    df = pd.DataFrame({"label": [0, 0, 0]})
    strat = RandomSplitStrategy(class_col_name="label")
    with pytest.raises(
        ValueError, match=re.escape("label must have at least two classes.")
    ):
        strat.validate_inputs(df)


def test_validate_inputs_raises_when_class_contains_missing_values():
    df = pd.DataFrame({"label": [0, 1, None]})
    strat = RandomSplitStrategy(class_col_name="label")
    with pytest.raises(ValueError, match=re.escape("label contains missing values.")):
        strat.validate_inputs(df)


def test_validate_inputs_raises_for_invalid_train_size():
    df = pd.DataFrame({"label": [0, 1, 0, 1]})
    strat = RandomSplitStrategy(
        class_col_name="label", train_set_size=1.0
    )  # invalid boundary
    with pytest.raises(
        ValueError, match=re.escape("train_set_size must be between 0 and 1.")
    ):
        strat.validate_inputs(df)


def test_validate_inputs_raises_for_non_int_random_seed():
    df = pd.DataFrame({"label": [0, 1, 0, 1]})
    strat = RandomSplitStrategy(class_col_name="label", random_seed="not-an-int")
    with pytest.raises(ValueError, match=re.escape("random_seed must be an integer.")):
        strat.validate_inputs(df)


def test_split_dataset_random(
    split_dataset_data,
):  # pylint: disable=redefined-outer-name
    """Test random splitting of dataset."""

    # Extract class column and test data
    class_col_name, input_data = split_dataset_data

    data_splitter = DataSplitter(
        dataset=input_data,
        primary_key_col_name="feature_1",
        class_col_name=class_col_name,
    )

    # Use RandomSplitStrategy
    random_strategy = RandomSplitStrategy(
        class_col_name=class_col_name,
        train_set_size=0.8,
        random_seed=42,
    )
    train_set_random, test_set_random = data_splitter.split_dataset(random_strategy)

    # Assert that the train and test sets are dataframes
    assert isinstance(train_set_random, pd.DataFrame)
    assert isinstance(test_set_random, pd.DataFrame)

    # Assert that the train and test sets have the expected sizes
    assert train_set_random.shape[0] == 0.8 * input_data.shape[0]
    assert test_set_random.shape[0] == 0.2 * input_data.shape[0]

    # Assert that the train and test sets have the same columns as the input data
    assert train_set_random.columns.tolist() == input_data.columns.tolist()
    assert test_set_random.columns.tolist() == input_data.columns.tolist()

    # Assert that the train and test sets have the same distribution of the class column as the input data
    assert (
        train_set_random[class_col_name]
        .value_counts(normalize=True)
        .sort_index(ascending=True)
        .equals(
            input_data[class_col_name]
            .value_counts(normalize=True)
            .sort_index(ascending=True)
        )
    )
    assert (
        test_set_random[class_col_name]
        .value_counts(normalize=True)
        .sort_index(ascending=True)
        .equals(
            input_data[class_col_name]
            .value_counts(normalize=True)
            .sort_index(ascending=True)
        )
    )


def test_validate_inputs_raises_when_date_column_missing():
    df = pd.DataFrame({"other": [1, 2, 3]})
    strat = TimeBasedSplitStrategy(date_col_name="date", cutoff_date=date(2020, 1, 1))
    with pytest.raises(ValueError, match=re.escape("date doesn't exist in dataset.")):
        strat._validate_inputs(df)  # pylint: disable=protected-access


def test_validate_inputs_raises_when_date_column_not_datetime():
    df = pd.DataFrame({"date": ["2020-01-01", "2020-02-01"]})  # object / string dtype
    strat = TimeBasedSplitStrategy(date_col_name="date", cutoff_date=date(2020, 1, 1))
    with pytest.raises(ValueError, match=re.escape("date must be in datetime format.")):
        strat._validate_inputs(df)  # pylint: disable=protected-access


def test_validate_inputs_raises_when_date_column_contains_missing_values():
    df = pd.DataFrame(
        {"date": pd.to_datetime(["2020-01-01", None])}
    )  # datetime dtype with NaT
    strat = TimeBasedSplitStrategy(date_col_name="date", cutoff_date=date(2020, 1, 1))
    with pytest.raises(ValueError, match=re.escape("date contains missing values.")):
        strat._validate_inputs(df)  # pylint: disable=protected-access


def test_validate_inputs_raises_when_cutoff_out_of_range():
    """Raises if cutoff_date is outside the min/max range of the date column."""

    df = pd.DataFrame({"date": pd.to_datetime(["2021-01-01", "2021-06-01"])})
    strat = TimeBasedSplitStrategy(date_col_name="date", cutoff_date=date(2020, 1, 1))

    with pytest.raises(ValueError, match="cutoff_date must be between"):
        strat._validate_inputs(df)  # pylint: disable=protected-access


def test_split_dataset_time(split_dataset_data):  # pylint: disable=redefined-outer-name
    """Test time-based splitting of dataset."""

    # Extract class column and test data
    class_col_name, input_data = split_dataset_data

    # Convert the date_col to datetime format
    input_data["date_col"] = pd.to_datetime(input_data["date_col"])

    data_splitter = DataSplitter(
        dataset=input_data,
        primary_key_col_name="feature_1",
        class_col_name=class_col_name,
    )

    # Use TimeBasedSplitStrategy
    time_strategy = TimeBasedSplitStrategy(
        date_col_name="date_col",
        cutoff_date=date(2021, 1, 7),
        date_format="%Y-%m-%d",
    )
    train_set_time, test_set_time = data_splitter.split_dataset(time_strategy)

    # Assert that the train and test sets are dataframes
    assert isinstance(train_set_time, pd.DataFrame)
    assert isinstance(test_set_time, pd.DataFrame)

    # Assert that the train and test sets have the expected sizes
    assert train_set_time.shape[0] == 6
    assert test_set_time.shape[0] == 4

    # Assert that the train and test sets have the same columns as the input data
    assert train_set_time.columns.tolist() == input_data.columns.tolist()
    assert test_set_time.columns.tolist() == input_data.columns.tolist()

    # Assert that the train and test sets have the expected date ranges
    assert train_set_time["date_col"].min() == pd.Timestamp("2021-01-01")
    assert train_set_time["date_col"].max() == pd.Timestamp("2021-01-06")
    assert test_set_time["date_col"].min() == pd.Timestamp("2021-01-07")
    assert test_set_time["date_col"].max() == pd.Timestamp("2021-01-10")


def test_check_datasets_overlap():
    """Test checking for overlapping datasets."""

    dataset1 = pd.DataFrame({"id": [1, 2, 3]})
    dataset2 = pd.DataFrame({"id": [3, 4, 5]})

    data_splitter = DataSplitter(
        dataset=dataset1,
        primary_key_col_name="id",
        class_col_name=None,
    )

    with pytest.raises(ValueError):
        data_splitter._check_datasets_overlap(  # pylint: disable=protected-access
            dataset1, dataset2
        )


def test_log_class_distribution(
    split_dataset_data,
):  # pylint: disable=redefined-outer-name, disable=protected-access
    """Test logging of class distribution."""
    class_col_name, input_data = split_dataset_data

    data_splitter = DataSplitter(
        dataset=input_data,
        primary_key_col_name="feature_1",
        class_col_name=class_col_name,
    )

    # Call the method and assert the return types
    data_splitter._log_class_distribution()
