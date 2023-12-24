"""
Tests helper functions of feature pipeline.
"""

import sys
from datetime import date

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import pytest

sys.path.insert(0, "src/feature_store/")
# sys.path.insert(1, "src/feature_store/")
from utils.prep import DataPreprocessor, DataSplitter


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


def test_split_dataset(split_dataset_data):
    # Extract class column and test data
    class_col_name, input_data = split_dataset_data

    # Create an instance of the class that contains the split_dataset function
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


#################################
@pytest.fixture
def remove_duplicates_by_unique_id_data():
    return (
        ["id"],
        pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "name": [
                    "Alice",
                    "Bob",
                    "Charlie",
                    "David",
                    "Eve",
                    "Alice",
                    "Bob",
                    "Charlie",
                    "David",
                    "Eve",
                ],
                "age": [20, 21, 22, 23, 24, 20, 21, 22, 23, 24],
                "gender": ["F", "M", "M", "M", "F", "F", "M", "M", "M", "F"],
            }
        ),
        pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 9],
                "name": [
                    "Alice",
                    "Bob",
                    "Charlie",
                    "David",
                    "Eve",
                    "Alice",
                    "Bob",
                    "Charlie",
                    "David",
                    "Eve",
                ],
                "age": [20, 21, 22, 23, 24, 20, 21, 22, 23, 24],
                "gender": ["F", "M", "M", "M", "F", "F", "M", "M", "M", "F"],
            }
        ),
    )


def test_remove_duplicates_by_unique_id(remove_duplicates_by_unique_id_data):
    (
        input_data_id,
        test_data_with_unique_id,
        test_data_with_non_unique_id,
    ) = remove_duplicates_by_unique_id_data

    test_data_with_unique_id_class = DataPreprocessor(
        input_data=test_data_with_unique_id,
        primary_key_names=input_data_id,
        date_cols_names=None,
        datetime_cols_names=None,
        num_feature_names=None,
        cat_feature_names=input_data_id,  # specified to pass assertion that at least one feature must be specified.
    )
    test_data_with_non_unique_id_class = DataPreprocessor(
        input_data=test_data_with_non_unique_id,
        primary_key_names=input_data_id,
        date_cols_names=None,
        datetime_cols_names=None,
        num_feature_names=None,
        cat_feature_names=input_data_id,  # specified to pass assertion that at least one feature must be specified.
    )

    test_data_with_unique_id_class.remove_duplicates_by_primary_key()
    output_data_with_unique_id = test_data_with_unique_id_class.get_preprocessed_data()

    test_data_with_non_unique_id_class.remove_duplicates_by_primary_key()
    output_data_with_non_unique_id = (
        test_data_with_non_unique_id_class.get_preprocessed_data()
    )

    # Assert that the output data is a dataframe
    assert isinstance(output_data_with_unique_id, pd.DataFrame)
    assert isinstance(output_data_with_non_unique_id, pd.DataFrame)

    # Assert that the output data has the expected shape
    assert output_data_with_unique_id.shape == test_data_with_unique_id.shape
    assert output_data_with_non_unique_id.shape == (9, 4)

    # Assert that the output data has no duplicate rows
    assert output_data_with_unique_id.duplicated().sum() == 0
    assert output_data_with_non_unique_id.duplicated().sum() == 0

    # Assert that the output data has no duplicate rows by the id column
    assert output_data_with_unique_id.duplicated(subset=input_data_id).sum() == 0
    assert output_data_with_non_unique_id.duplicated(subset=input_data_id).sum() == 0


#################################
@pytest.fixture(name="replace_blank_values_with_nan_data")
def replace_blank_values_with_nan_data_fixture():
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "   ", pd.NA, "", "Eve"],
            "patient": ["Patient 1", " Patient 2", None, pd.NA, " ", "Patient 3 "],
            "payment": [255.2, 30528.90, None, 9.76, 40, np.nan],
            "date": ["2023-22-08", "2023-16-09", pd.NA, "", None, "2023-03-01"],
            "datetime": [
                pd.NaT,
                "2022-03-02 12:00:55.1567",
                pd.NA,
                " ",
                None,
                "2021-20-05 00:14:00.568",
            ],
        }
    )


def test_replace_blank_values_with_nan(replace_blank_values_with_nan_data):
    """
    Tests if blank cells in string or object columns are replaced
    with pd.nan. Columns with other data types should be kept unchanged.
    """

    # Create a class instance
    replace_blanks_class = DataPreprocessor(
        input_data=replace_blank_values_with_nan_data,
        primary_key_names=None,
        date_cols_names=None,
        datetime_cols_names=None,
        num_feature_names=None,
        cat_feature_names=[
            "name"
        ],  # specified to pass assertion that at least one feature must be specified.
    )
    replace_blanks_class.replace_blank_values_with_nan()
    sample_data_without_blank_cells = replace_blanks_class.get_preprocessed_data()

    # Assert the output is dataframe and has same shape as input data
    assert isinstance(sample_data_without_blank_cells, pd.DataFrame)
    assert (
        sample_data_without_blank_cells.shape
        == replace_blank_values_with_nan_data.shape
    )

    # Assert that the output data has the expected NaNs count in each column
    assert sample_data_without_blank_cells[["name"]].isna().sum().iloc[0] == 3
    assert sample_data_without_blank_cells[["patient"]].isna().sum().iloc[0] == 3
    assert sample_data_without_blank_cells[["payment"]].isna().sum().iloc[0] == 2
    assert sample_data_without_blank_cells[["date"]].isna().sum().iloc[0] == 3
    assert sample_data_without_blank_cells[["datetime"]].isna().sum().iloc[0] == 4


#################################
@pytest.fixture(name="specify_data_types_data")
def specify_data_types_data_fixture():
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", None, pd.NA, "", "Eve"],
            "age": [25, 30, None, 9, 40, np.nan],
            "payment": [255.2, 30528.90, None, 9.76, 40, np.nan],
            "date": ["2023-22-08", "2023-16-09", pd.NA, "", None, "2023-03-01"],
            "datetime": [
                pd.NaT,
                "2022-03-02 12:00:55",
                pd.NA,
                "",
                None,
                "2021-20-05 00:14:00",
            ],
        }
    )


def test_specify_data_types(specify_data_types_data):
    """
    Tests missing values indicators of float, string, date, and datetime
    data type columns. Note that specify_data_types converts numerical
    columns, including int, to float 32.
    """

    # Create a class instance
    specify_data_types_class = DataPreprocessor(
        input_data=specify_data_types_data,
        primary_key_names=None,
        date_cols_names=["date"],
        datetime_cols_names=["datetime"],
        num_feature_names=["age", "payment"],
        cat_feature_names=None,
    )
    specify_data_types_class.specify_data_types()
    data_with_nan_indicator = specify_data_types_class.get_preprocessed_data()

    # Assert the output is dataframe and has same shape as input data
    assert isinstance(data_with_nan_indicator, pd.DataFrame)
    assert data_with_nan_indicator.shape == specify_data_types_data.shape

    # Assert that each column has the expected data type
    assert ptypes.is_string_dtype(data_with_nan_indicator["name"])
    assert ptypes.is_float_dtype(data_with_nan_indicator["age"])
    assert ptypes.is_float_dtype(data_with_nan_indicator["payment"])
    assert ptypes.is_datetime64_ns_dtype(data_with_nan_indicator["date"])
    assert ptypes.is_datetime64_ns_dtype(data_with_nan_indicator["datetime"])

    # Assert that the output data has the expected NA indicators
    assert data_with_nan_indicator[["name"]].isna().sum().iloc[0] == 3
    assert data_with_nan_indicator[["age"]].isna().sum().iloc[0] == 2
    assert data_with_nan_indicator[["payment"]].isna().sum().iloc[0] == 2
    assert data_with_nan_indicator[["date"]].isna().sum().iloc[0] == 3
    assert data_with_nan_indicator[["datetime"]].isna().sum().iloc[0] == 4


#################################
# Create data to test identify_cols_with_high_nans function
identify_cols_with_high_nans_data = pd.DataFrame(
    np.random.randn(10, 5), columns=list("ABCDE")
)

identify_cols_with_high_nans_data["A"] = identify_cols_with_high_nans_data["A"].astype(
    int
)  # integer
identify_cols_with_high_nans_data["B"] = identify_cols_with_high_nans_data["B"].astype(
    float
)  # float
identify_cols_with_high_nans_data["C"] = identify_cols_with_high_nans_data["C"].astype(
    "string"
)  # string
identify_cols_with_high_nans_data["D"] = pd.to_datetime(
    identify_cols_with_high_nans_data["D"]
)  # datetime
identify_cols_with_high_nans_data["E"] = pd.date_range(
    "2023-10-13", periods=10, freq="D"
)  # date


identify_cols_with_high_nans_data.loc[
    :2, "A"
] = np.nan  # 40% missing values in column A
identify_cols_with_high_nans_data.loc[3:4, "A"] = None  # 40% missing values in column A
identify_cols_with_high_nans_data.loc[
    1:4, "B"
] = np.nan  # 40% missing values in column B
identify_cols_with_high_nans_data.loc[3, "C"] = np.nan  # 40% missing values in column C
identify_cols_with_high_nans_data.loc[4, "C"] = pd.NA  # 40% missing values in column C
identify_cols_with_high_nans_data.loc[5, "C"] = None  # 40% missing values in column C
identify_cols_with_high_nans_data.loc[
    [0, 3], "D"
] = np.nan  # 30% missing values in column D
identify_cols_with_high_nans_data.loc[
    [3], "E"
] = np.nan  # 20% missing values in column E


@pytest.mark.parametrize(
    "data, threshold_val, expected",
    [
        (
            identify_cols_with_high_nans_data,
            0.1,
            [
                "A",
                "B",
                "C",
                "D",
                "E",
            ],
        ),
        (
            identify_cols_with_high_nans_data,
            0.2,
            [
                "A",
                "B",
                "C",
                "D",
            ],
        ),
        (
            identify_cols_with_high_nans_data,
            0.3,
            [
                "A",
                "B",
                "C",
            ],
        ),
        (
            identify_cols_with_high_nans_data,
            0.4,
            [
                "A",
                "B",
            ],
        ),
    ],
)
def test_identify_cols_with_high_nans(data, threshold_val, expected):
    """
    Tests columns with high missing values above a specific threshold value.
    """

    identify_high_nans_class = DataPreprocessor(
        input_data=data,
        primary_key_names=None,
        date_cols_names=None,
        datetime_cols_names=None,
        num_feature_names=None,
        cat_feature_names=["C"],
    )

    # Assert that function identifies the expect columns with high NaNs for each test input
    assert (
        identify_high_nans_class.identify_cols_with_high_nans(
            high_nans_percent_threshold=threshold_val, update_cols_types=False
        )
        == expected
    )
