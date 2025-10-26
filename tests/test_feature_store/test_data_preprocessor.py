"""
Test functions for the DataPreprocessor class in the
src/feature_store/utils/prep.py file.
"""

import re

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import pytest

from src.feature_store.utils.prep import DataPreprocessor


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
                    "David",
                ],
                "age": [20, 21, 22, 23, 24, 20, 21, 22, 24, 24],
                "gender": ["F", "M", "M", "M", "F", "F", "M", "M", "F", "F"],
            }
        ),
    )


def test_prepend_and_append_steps_are_applied_in_order():
    """Verify prepend_steps are placed before and append_steps after the base steps."""

    calls = []

    def pre_step(df: pd.DataFrame) -> pd.DataFrame:
        calls.append("pre")
        return df.copy()

    def base_step(df: pd.DataFrame) -> pd.DataFrame:
        calls.append("base")
        return df.copy()

    def post_step(df: pd.DataFrame) -> pd.DataFrame:
        calls.append("post")
        return df.copy()

    def added_step(df: pd.DataFrame) -> pd.DataFrame:
        calls.append("added")
        return df.copy()

    dp = DataPreprocessor(
        steps=[base_step], prepend_steps=[pre_step], append_steps=[post_step]
    )
    dp.add_step(added_step)

    input_df = pd.DataFrame({"x": [1, 2, 3]})
    out = dp.run_preprocessing_pipeline(input_df)

    assert calls == ["pre", "base", "post", "added"]

    # Ensure pipeline returned a DataFrame and original input was not mutated
    assert isinstance(out, pd.DataFrame)
    assert "x" in input_df.columns and input_df.shape == (3, 1)


def test_check_duplicate_rows(
    remove_duplicates_by_unique_id_data,  # pylint: disable=redefined-outer-name
):
    # Extract primary key column name(s) and test data with a duplicate row
    input_data_id, _, test_data_with_non_unique_id = remove_duplicates_by_unique_id_data

    # Case: no primary key provided -> any duplicate row should raise
    preprocessor_no_pk = DataPreprocessor(primary_key_names=None)
    with pytest.raises(ValueError, match=re.escape("1 duplicate rows.")):
        preprocessor_no_pk.check_duplicate_rows(test_data_with_non_unique_id)

    # Case: primary key provided -> duplicate by primary key should raise with PK mentioned
    preprocessor_with_pk = DataPreprocessor(primary_key_names=input_data_id)
    expected_msg = f"1 rows with duplicate {input_data_id}."
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        preprocessor_with_pk.check_duplicate_rows(test_data_with_non_unique_id)


def test_check_duplicate_rows_returns_df_when_no_duplicates():
    """Returns the original DataFrame when no duplicates are present."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "value": [10, 20, 30],
        }
    )

    pre_no_pk = DataPreprocessor(primary_key_names=None)
    out = pre_no_pk.check_duplicate_rows(df)
    assert out is df  # same object returned when no duplicates

    # also when primary key provided and rows are unique by that key
    pre_with_pk = DataPreprocessor(primary_key_names=["id"])
    out2 = pre_with_pk.check_duplicate_rows(df)
    assert out2 is df


def test_remove_duplicates_by_primary_key(
    remove_duplicates_by_unique_id_data,  # pylint: disable=redefined-outer-name
):
    (
        input_data_id,
        test_data_with_unique_id,
        test_data_with_non_unique_id,
    ) = remove_duplicates_by_unique_id_data

    # Use stateless DataPreprocessor: methods accept a DataFrame and return a DataFrame
    preprocessor_unique = DataPreprocessor(primary_key_names=input_data_id)
    output_data_with_unique_id = preprocessor_unique.remove_duplicates_by_primary_key(
        test_data_with_unique_id
    )

    preprocessor_non_unique = DataPreprocessor(primary_key_names=input_data_id)
    output_data_with_non_unique_id = (
        preprocessor_non_unique.remove_duplicates_by_primary_key(
            test_data_with_non_unique_id
        )
    )

    # Assert that the function raises an error if no primary key is provided
    preprocessor_no_pk = DataPreprocessor(primary_key_names=[])
    with pytest.raises(
        ValueError, match=re.escape("No primary key column(s) provided!")
    ):
        preprocessor_no_pk.remove_duplicates_by_primary_key(test_data_with_unique_id)

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

    preprocessor = DataPreprocessor(
        primary_key_names=None,
        date_cols_names=None,
        datetime_cols_names=None,
        num_feature_names=None,
        cat_feature_names=["name"],  # ensure categorical column list is provided
    )

    sample_data_without_blank_cells = preprocessor.replace_blank_values_with_nan(
        replace_blank_values_with_nan_data
    )

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


def test_replace_common_missing_values_replaces_tokens_with_nan_and_is_non_mutating():
    df = pd.DataFrame(
        {
            "col": [
                "",  # empty string
                "<NA>",  # literal token
                "null",  # literal token
                "?",  # literal token
                None,  # None
                "N/A",  # literal token
                "NAN",  # literal token
                "nan",  # literal token
                pd.NA,  # pandas NA
                "keep",  # value that should remain unchanged
            ]
        }
    )

    pre = DataPreprocessor()
    out = pre.replace_common_missing_values(df)

    # output: nine tokens should be converted to np.nan, one value unchanged
    assert isinstance(out, pd.DataFrame)

    assert out["col"].isna().sum() == 9  # pylint: disable=unsubscriptable-object
    assert out["col"].iloc[-1] == "keep"  # pylint: disable=unsubscriptable-object

    # original DataFrame should not be mutated for string tokens
    assert df["col"].iloc[0] == ""
    assert df["col"].iloc[1] == "<NA>"
    assert df["col"].iloc[-1] == "keep"


#################################
@pytest.fixture(name="specify_data_types_data")
def specify_data_types_data_fixture():
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", None, pd.NA, "", np.nan],
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
    columns, including int, to float32.
    """

    preprocessor = DataPreprocessor(
        primary_key_names=None,
        date_cols_names=["date"],
        datetime_cols_names=["datetime"],
        num_feature_names=["age", "payment"],
        cat_feature_names=None,
    )

    # specify_data_types is now stateless: it accepts a DataFrame and returns a DataFrame
    data_with_nan_indicator = preprocessor.specify_data_types(specify_data_types_data)

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


identify_cols_with_high_nans_data.loc[:2, "A"] = (
    np.nan
)  # 40% missing values in column A
identify_cols_with_high_nans_data.loc[3:4, "A"] = None  # 40% missing values in column A
identify_cols_with_high_nans_data.loc[1:4, "B"] = (
    np.nan
)  # 40% missing values in column B
identify_cols_with_high_nans_data.loc[3, "C"] = np.nan  # 40% missing values in column C
identify_cols_with_high_nans_data.loc[4, "C"] = pd.NA  # 40% missing values in column C
identify_cols_with_high_nans_data.loc[5, "C"] = None  # 40% missing values in column C
identify_cols_with_high_nans_data.loc[[0, 3], "D"] = (
    np.nan
)  # 30% missing values in column D
identify_cols_with_high_nans_data.loc[[3], "E"] = (
    np.nan
)  # 20% missing values in column E


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

    preprocessor = DataPreprocessor(
        primary_key_names=None,
        date_cols_names=None,
        datetime_cols_names=None,
        num_feature_names=None,
        cat_feature_names=["C"],
    )

    processed = preprocessor.identify_cols_with_high_nans(
        data, high_nans_percent_threshold=threshold_val, update_cols_types=False
    )

    cols_to_drop = processed.attrs.get("cols_to_drop_due_to_nans", [])
    assert cols_to_drop == expected


def test_identify_cols_with_high_nans_respects_cols_to_exclude():
    # A: 3/4 NaNs => 0.75, B: 2/4 => 0.5, C: 0/4 => 0.0
    df = pd.DataFrame(
        {
            "A": [np.nan, 1, np.nan, np.nan],
            "B": [np.nan, np.nan, 1, 2],
            "C": [1, 2, 3, 4],
        }
    )

    pre = DataPreprocessor()
    processed = pre.identify_cols_with_high_nans(
        df,
        cols_to_exclude=["A"],
        high_nans_percent_threshold=0.5,
        update_cols_types=False,
    )

    # 'A' would be flagged but is excluded; 'B' remains flagged
    assert processed.attrs["cols_to_drop_due_to_nans"] == ["B"]


def test_identify_cols_with_high_nans_updates_column_lists_when_requested():
    # Prepare data: num1 -> 3/4 NaNs (0.75), cat1 -> 2/4 NaNs (0.5)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2021-01-01", "2021-02-01", "2021-03-01", "2021-04-01"]
            ),
            "datetime": pd.to_datetime(
                ["2021-01-01 00:00", "2021-02-01 00:00", None, "2021-04-01 00:00"]
            ),
            "num1": [np.nan, np.nan, np.nan, 1.0],
            "num2": [1.0, 2.0, 3.0, 4.0],
            "cat1": [None, None, "a", "b"],
            "cat2": ["x", "y", "z", "w"],
        }
    )

    pre = DataPreprocessor(
        date_cols_names=["date"],
        datetime_cols_names=["datetime"],
        num_feature_names=["num1", "num2"],
        cat_feature_names=None,
    )

    processed = pre.identify_cols_with_high_nans(
        df, high_nans_percent_threshold=0.5, update_cols_types=True
    )

    # cols with high nans (>= 0.5) should be num1 and cat1
    assert processed.attrs["cols_to_drop_due_to_nans"] == ["num1", "cat1"]

    # updated_column_lists should reflect removal of num1 and cat1
    expected = {
        "date_cols": ["date"],
        "datetime_cols": ["datetime"],
        "num_cols": ["num2"],
        "cat_cols": ["cat2"],
    }
    assert processed.attrs["updated_column_lists"] == expected


def test_run_preprocessing_pipeline_applies_steps_in_order_and_does_not_mutate_input():
    # input dataframe
    df = pd.DataFrame({"x": [1, 2]})

    # step 1: add column 'y' = x + 1
    def add_y(df_in: pd.DataFrame) -> pd.DataFrame:
        out = df_in.copy()
        out["y"] = out["x"] + 1
        return out

    # step 2: scale 'y' by 10
    def scale_y(df_in: pd.DataFrame) -> pd.DataFrame:
        out = df_in.copy()
        out["y"] = out["y"] * 10
        return out

    preprocessor = DataPreprocessor(steps=[add_y, scale_y])

    processed = preprocessor.run_preprocessing_pipeline(df)

    # processed has new column and expected values
    assert "y" in processed.columns
    assert processed["y"].tolist() == [20, 30]

    # original input is unchanged (stateless behavior)
    assert "y" not in df.columns
    assert df.shape == (2, 1)

    # processed is a different object than the input copy
    assert processed is not df
