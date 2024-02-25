"""
Tests helper functions of feature pipeline.
"""

from datetime import date

import pandas as pd
import pytest

from src.feature_store.utils.prep import DataPreprocessor, DataSplitter


@pytest.fixture
def split_dataset_data():
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


def test_split_dataset_random(split_dataset_data):
    # Extract class column and test data
    class_col_name, input_data = split_dataset_data

    data_splitter_class = DataSplitter(
        dataset=input_data,
        primary_key_col_name="feature_1",
        class_col_name=class_col_name,
    )

    # Call the function with the random split type and the default parameters
    train_set_random, test_set_random = data_splitter_class.split_dataset(
        split_type="random"
    )

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


def test_split_dataset_time(split_dataset_data):
    # Extract class column and test data
    class_col_name, input_data = split_dataset_data

    data_splitter_class = DataSplitter(
        dataset=input_data,
        primary_key_col_name="feature_1",
        class_col_name=class_col_name,
    )

    train_set_time, test_set_time = data_splitter_class.split_dataset(
        split_type="time",
        split_date_col_name="date_col",
        split_cutoff_date=date(2021, 1, 7),
        split_date_col_format="%Y-%m-%d",
    )

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
    assert train_set_time["date_col"].min() == "2021-01-01"
    assert train_set_time["date_col"].max() == "2021-01-06"
    assert test_set_time["date_col"].min() == "2021-01-07"
    assert test_set_time["date_col"].max() == "2021-01-10"


def test_check_datasets_overlap():
    dataset1 = pd.DataFrame({"id": [1, 2, 3]})
    dataset2 = pd.DataFrame({"id": [3, 4, 5]})

    data_splitter_class = DataSplitter(
        dataset=dataset1,
        primary_key_col_name="id",
        class_col_name=None,
    )

    with pytest.raises(ValueError):
        data_splitter_class.check_datasets_overlap(dataset1, dataset2)


def test_print_class_dist(split_dataset_data):
    class_col_name, input_data = split_dataset_data

    data_splitter_class = DataSplitter(
        dataset=input_data,
        primary_key_col_name="feature_1",
        class_col_name=class_col_name,
    )

    n_class_labels, class_labels_proportions = data_splitter_class.print_class_dist()

    assert isinstance(n_class_labels, pd.Series)
    assert isinstance(class_labels_proportions, pd.Series)
