"""
Test functions for the DataTransformer class in the feature store
module src/feature_store/utils/prep.py.
"""

import pandas as pd
import pytest

from src.feature_store.utils.prep import DataTransformer


@pytest.fixture
def data_transformer_input_data():
    return pd.DataFrame(
        {
            "GenHlth": [
                "1",
                "2",
                "3",
                "4",
                "5",
                "1",
                "2",
                "3",
                "4",
                "5",
                "1",
                "2",
                "3",
            ],
            "Education": [
                "1",
                "2",
                "3",
                "4",
                "5",
                "1",
                "2",
                "3",
                "4",
                "5",
                "1",
                "2",
                "3",
            ],
            "Age": [
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
                "11",
                "12",
                "13",
            ],
            "Class": ["0", "1", "0", "1", "0", "0", "1", "0", "1", "0", "0", "1", "0"],
        }
    )


def test_map_categorical_features(data_transformer_input_data):
    data_transformer = DataTransformer(data_transformer_input_data)
    data_transformer.map_categorical_features(
        col_name="GenHlth",
        mapping_values={
            "1": "Poor",
            "2": "Fair",
            "3": "Good",
            "4": "Very Good",
            "5": "Excellent",
        },
    )
    data_transformer.map_categorical_features(
        col_name="Education",
        mapping_values={
            "1": "Never Attended School",
            "2": "Elementary",
            "3": "High School",
            "4": "Some College Degree",
            "5": "Advanced Degree",
            "6": "Advanced Degree",
        },
    )

    data_transformer.map_categorical_features(
        col_name="Age",
        mapping_values={
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
        },
    )

    # Check if categorical features are mapped correctly
    assert data_transformer.preprocessed_data["GenHlth"].tolist() == [
        "Poor",
        "Fair",
        "Good",
        "Very Good",
        "Excellent",
        "Poor",
        "Fair",
        "Good",
        "Very Good",
        "Excellent",
        "Poor",
        "Fair",
        "Good",
    ]
    assert data_transformer.preprocessed_data["Education"].tolist() == [
        "Never Attended School",
        "Elementary",
        "High School",
        "Some College Degree",
        "Advanced Degree",
        "Never Attended School",
        "Elementary",
        "High School",
        "Some College Degree",
        "Advanced Degree",
        "Never Attended School",
        "Elementary",
        "High School",
    ]
    assert data_transformer.preprocessed_data["Age"].tolist() == [
        "18 to 24",
        "25 to 29",
        "30 to 34",
        "35 to 39",
        "40 to 44",
        "45 to 49",
        "50 to 54",
        "55 to 59",
        "60 to 64",
        "65 to 69",
        "70 to 74",
        "75 to 79",
        "80 or older",
    ]

    # Check if raises warning when column name is not found
    with pytest.warns(UserWarning):
        data_transformer.map_categorical_features(
            col_name="NonExistentColumn",
            mapping_values={
                "1": "First",
                "2": "Second",
            },
        )


def test_map_class_labels(data_transformer_input_data):
    data_transformer = DataTransformer(data_transformer_input_data)
    data_transformer.map_class_labels(
        "Class", mapping_values={"0": "Non-Diabetic", "1": "Diabetic"}
    )

    # Check if class labels are mapped correctly
    assert data_transformer.preprocessed_data["Class"].tolist() == [
        "Non-Diabetic",
        "Diabetic",
        "Non-Diabetic",
        "Diabetic",
        "Non-Diabetic",
        "Non-Diabetic",
        "Diabetic",
        "Non-Diabetic",
        "Diabetic",
        "Non-Diabetic",
        "Non-Diabetic",
        "Diabetic",
        "Non-Diabetic",
    ]

    # Check if raises ValueError when class column name is not found
    with pytest.raises(ValueError):
        data_transformer.map_class_labels(
            "NonExistentClass",
            mapping_values={"0": "Non-Diabetic", "1": "Diabetic", "2": "Pre-Diabetic"},
        )
