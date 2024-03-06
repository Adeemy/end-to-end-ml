"""
Test functions for data preprocessing and splitting in the 
training module src/training/utils/data.py.
"""

import re

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler

from src.feature_store.utils.prep import DataSplitter
from src.training.utils.data import DataPipelineCreator, TrainingDataPrep


@pytest.fixture
def test_df():
    """Creates a test dataset with numerical and categorical features. It also
    includes a near-zero variance feature to test selector step."""

    # Set seed for reproducibility
    np.random.seed(0)
    df = pd.DataFrame(
        {
            "num_feature1": np.random.randn(100),
            "num_feature2": np.random.randn(100),
            "near_zero_var_num_feature": np.append(
                np.full(97, 0), np.random.randn(3)
            ),  # 97% zeros, 3% random values
            "cat_feature1": np.random.choice(["A", "B", "C"], 100),
            "cat_feature2": np.random.choice(["X", "Y", "Z"], 100),
            "near_zero_var_cat_feature": np.append(
                np.full(97, "A"), np.random.choice(["B", "C"], 3)
            ),  # 97% 'A', 3% 'B' or 'C'
        }
    )

    # Define the names of the numerical and categorical features
    num_feat_col_names = ["num_feature1", "num_feature2", "near_zero_var_num_feature"]
    cat_feat_col_names = ["cat_feature1", "cat_feature2", "near_zero_var_cat_feature"]

    return df, num_feat_col_names, cat_feat_col_names


def test_create_num_features_transformer():
    """Tests the creation of a pipeline for numerical features. The pipeline should include
    an imputer and a scaler. The imputer should be a SimpleImputer with the mean strategy,
    and the scaler should be a StandardScaler. The pipeline should have two steps.
    """

    pipeline_creator = DataPipelineCreator()
    pipeline = pipeline_creator.create_num_features_transformer()

    # Test output type (must be a pipeline)
    assert isinstance(pipeline, Pipeline)
    # num_features_transformer must have two steps: imputer and scaler
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0][0] == "imputer"
    assert pipeline.steps[1][0] == "scaler"
    # If this fails, it means that the default values have changed and the tests need to be updated.
    assert isinstance(pipeline.steps[0][1], SimpleImputer)
    assert isinstance(pipeline.steps[1][1], RobustScaler)


def test_create_cat_features_transformer():
    """Tests the creation of a pipeline for categorical features. The pipeline should include
    an imputer and a one-hot encoder. The imputer should be a SimpleImputer with the constant
    strategy and the one-hot encoder should be a OneHotEncoder. The pipeline should have two steps.
    """

    pipeline_creator = DataPipelineCreator()
    transformer = pipeline_creator.create_cat_features_transformer()

    # Test output type (must be a pipeline)
    assert isinstance(transformer, Pipeline)
    assert len(transformer.steps) == 2

    # cat_features_transformer must have two steps: imputer and one-hot encoder
    assert transformer.steps[0][0] == "imputer"
    assert transformer.steps[1][0] == "onehot_encoder"

    # If this fails, it means that the default values have changed and the tests need to be updated.
    assert isinstance(transformer.steps[0][1], SimpleImputer)
    assert isinstance(transformer.steps[1][1], OneHotEncoder)


def test_extract_col_names_after_preprocessing(test_df):
    """Tests the extraction of column names after preprocessing. The test dataset
    includes numerical and categorical features, and a near-zero variance feature
    to test the selector step. The expected output is a list with the names of the
    numerical and categorical features. The near-zero variance feature should be removed.
    """

    df, num_feat_col_names, cat_feat_col_names = test_df

    pipeline_creator = DataPipelineCreator()

    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value=np.nan,
                ),
            ),
            (
                "onehot_encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    categories="auto",
                    drop="first",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", num_transformer, num_feat_col_names),
            (
                "categorical",
                cat_transformer,
                cat_feat_col_names,
            ),
        ]
    )

    selector = VarianceThreshold(threshold=0.05)
    data_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("selector", selector)]
    )
    transformed_df = data_pipeline.fit_transform(df)
    transformed_df = pd.DataFrame(transformed_df)

    col_names = pipeline_creator.extract_col_names_after_preprocessing(
        num_feat_col_names=num_feat_col_names,
        cat_feat_col_names=cat_feat_col_names,
        selector=selector,
        data_pipeline=data_pipeline,
    )

    # Check the new column names are as expected
    assert col_names[0] == "num_feature1"
    assert col_names[1] == "num_feature2"
    assert col_names[2] == "cat_feature1_B"
    assert col_names[3] == "cat_feature1_C"
    assert col_names[4] == "cat_feature2_Y"
    assert col_names[5] == "cat_feature2_Z"


def test_create_data_pipeline(test_df):
    """Tests the creation of a data pipeline. The test dataset includes numerical and
    categorical features, and a near-zero variance feature to test the selector step.
    The expected output is a dataframe with the preprocessed features and a pipeline
    with the preprocessing steps.
    """

    df, num_feat_col_names, cat_feat_col_names = test_df

    pipeline_creator = DataPipelineCreator(
        num_features_imputer="median",
        num_features_scaler=RobustScaler(),
        cat_features_imputer="constant",
        cat_features_ohe_handle_unknown="infrequent_if_exist",
        cat_features_nans_replacement=np.nan,
    )
    transformed_df, data_pipeline = pipeline_creator.create_data_pipeline(
        input_features=df,
        num_feature_col_names=num_feat_col_names,
        cat_feature_col_names=cat_feat_col_names,
        variance_threshold_val=0.05,
    )
    col_names = list(transformed_df.columns)

    assert isinstance(transformed_df, pd.DataFrame)
    assert isinstance(data_pipeline, Pipeline)

    # Check the new column names are as expected
    assert col_names[0] == "num_feature1"
    assert col_names[1] == "num_feature2"
    assert col_names[2] == "cat_feature1_B"
    assert col_names[3] == "cat_feature1_C"
    assert col_names[4] == "cat_feature2_Y"
    assert col_names[5] == "cat_feature2_Z"

    # Check the near-zero variance features are removed
    assert ["cat_feature1_A", "cat_feature2_X"] not in col_names

    # Check the drop='first' columns are removed
    assert ["near_zero_var_num_feature", "near_zero_var_cat_feature"] not in col_names


@pytest.fixture
def training_data_prep():
    # Create a sample train set and test set
    train_set = pd.DataFrame(
        {
            "id": list(range(1, 11)),
            "age": [25, 30, np.nan, 35, 40, 45, 50, np.nan, 55, 60],
            "gender": [
                "Male",
                "Female",
                np.nan,
                "",
                pd.NA,
                "Male",
                "Female",
                "Male",
                "Male",
                "Female",
            ],
            "zip_code": [
                "29875",
                "37540",
                "37535",
                "12345",
                "67890",
                "11111",
                "22222",
                "33333",
                "44444",
                "55555",
            ],
            "class": list("AABBBABAAA"),
        }
    )

    test_set = pd.DataFrame(
        {
            "id": list(range(10, 15)),
            "age": [20, 70, np.nan, 75, 31],
            "gender": [
                "Female",
                "Male",
                np.nan,
                np.nan,
                "Female",
            ],
            "zip_code": [
                "66666",
                "77777",
                "11111",
                "33333",
                "44444",
            ],
            "class": list("BBAAA"),
        }
    )

    # Create an instance of TrainingDataPrep
    data_prep = TrainingDataPrep(
        train_set=train_set,
        test_set=test_set,
        primary_key="id",
        class_col_name="class",
        numerical_feature_names=["age"],
        categorical_feature_names=["gender"],
    )

    return data_prep


def test_select_relevant_columns(training_data_prep):
    """Tests the select_relevant_columns method of the TrainingDataPrep class. The method
    should select the relevant columns from the train and test sets. The relevant columns
    are the primary key, numerical features, categorical features, and the target variable
    specified in TrainingDataPrep initialization. The method should remove any other
    columns from the train and test sets. The expected output is the train and test sets
    with only the relevant columns.
    """

    # Call the select_relevant_columns method
    training_data_prep.select_relevant_columns()

    # Check if the selected columns are correct in the train and test set
    assert list(training_data_prep.train_set.columns) == [
        "id",
        "age",
        "gender",
        "class",
    ]

    # Check if the selected columns are correct in the test set
    assert list(training_data_prep.test_set.columns) == ["id", "age", "gender", "class"]


def test_enforce_data_types(training_data_prep):
    """Tests the enforce_data_types method of the TrainingDataPrep class. The method
    should enforce the specified data types for the numerical and categorical features
    in the train and test sets. The method should also replace common missing values in
    categorical features np.nan because it calls replace_common_missing_values method
    before specifying data types.
    """

    # Call the enforce_data_types method
    training_data_prep.enforce_data_types()

    # Check if the data types are enforced correctly in the train set
    assert training_data_prep.train_set["age"].dtype == "float32"
    assert training_data_prep.train_set["gender"].dtype == "string"

    # Check if the data types are enforced correctly in the test set
    assert training_data_prep.test_set["age"].dtype == "float32"
    assert training_data_prep.test_set["gender"].dtype == "string"

    # Check the number of NaNs are in train set
    assert training_data_prep.train_set["age"].isna().sum() == 2
    assert training_data_prep.train_set["gender"].isna().sum() == 3

    # Check the number of NaNs are in test set
    assert training_data_prep.test_set["age"].isna().sum() == 1
    assert training_data_prep.test_set["gender"].isna().sum() == 2


def test_create_validation_set(training_data_prep):
    # Default validation set is None
    assert training_data_prep.valid_set is None

    # Call the create_validation_set method
    training_data_prep.create_validation_set(split_random_seed=100)

    # Check if the validation set is created
    assert training_data_prep.valid_set is not None


def test_extract_features(training_data_prep):
    # Apply all steps before extract_features to ensure this test is independent of other tests
    training_data_prep.select_relevant_columns()
    training_data_prep.create_validation_set(split_random_seed=100)
    training_data_prep.enforce_data_types()

    # Call the extract_features method
    training_data_prep.extract_features()

    # Check if the training, testing, and validation features are extracted correctly,
    # which should not include the primary key and target variable
    assert list(training_data_prep.training_features.columns) == ["age", "gender"]
    assert list(training_data_prep.validation_features.columns) == ["age", "gender"]
    assert list(training_data_prep.testing_features.columns) == ["age", "gender"]


def test_encode_class_labels(training_data_prep):
    """Tests the encode_class_labels method of the TrainingDataPrep class. The method
    should encode the class labels in the train, validation, and test sets. The method
    should also return the encoded class labels, the encoded positive class label, and
    the fitted class encoder.
    """

    # Apply all steps before encoding class labels to ensure this test is independent of other tests
    training_data_prep.select_relevant_columns()
    training_data_prep.enforce_data_types()
    training_data_prep.create_validation_set(split_random_seed=100)
    training_data_prep.extract_features()

    (
        encoded_train_class,
        encoded_valid_class,
        encoded_test_class,
        enc_pos_class_label,
        fitted_class_encoder,
    ) = training_data_prep.encode_class_labels(pos_class_label="B")

    # Check if encoded classes are correct (A=1 and B=0)
    assert np.array_equal(encoded_train_class, np.array([0, 0, 0, 0, 1, 1, 0, 1]))
    assert np.array_equal(encoded_valid_class, np.array([0, 1]))
    assert np.array_equal(encoded_test_class, np.array([1, 1, 0, 0, 0]))

    assert (
        enc_pos_class_label == 1
    )  # positive class label is 'B' and should be encoded as 1
    assert isinstance(fitted_class_encoder, LabelEncoder)


def test_create_data_transformation_pipeline(test_df):
    """Tests if create_data_transformation_pipeline method returns a data transformation
    pipeline and creates preprocessed train and validation sets used for hyperparameter tuning.
    """

    df, num_feat_col_names, cat_feat_col_names = test_df

    # Add primary key and class columns
    df = df.reset_index(names="id")
    df["class"] = np.random.choice(["A", "B"], 100)

    # Split data into train and test sets
    data_splitter = DataSplitter(
        dataset=df,
        primary_key_col_name="id",
        class_col_name="class",
    )
    train_set, test_set = data_splitter.split_dataset()

    # Apply all steps before encoding class labels to ensure this test is independent of other tests
    data_prep = TrainingDataPrep(
        train_set=train_set,
        test_set=test_set,
        primary_key="id",
        class_col_name="class",
        numerical_feature_names=num_feat_col_names,
        categorical_feature_names=cat_feat_col_names,
    )

    data_prep.select_relevant_columns()
    data_prep.enforce_data_types()
    data_prep.create_validation_set(split_random_seed=100)
    data_prep.extract_features()

    pipeline = data_prep.create_data_transformation_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert isinstance(data_prep.train_features_preprocessed, pd.DataFrame)
    assert isinstance(data_prep.valid_features_preprocessed, pd.DataFrame)

    # Check if the preprocessed train and validation sets have the same columns
    assert list(data_prep.train_features_preprocessed.columns) == list(
        data_prep.valid_features_preprocessed.columns
    )


def test_clean_up_feature_names(training_data_prep):
    """Tests the clean_up_feature_names method of the TrainingDataPrep class. The method
    should clean up the column names of the training, validation, and testing features
    by removing special characters and replacing them with underscores. The method should
    also clean up the column names of the preprocessed training and validation features.
    """

    # Create small test data with special characters in column names
    training_data_prep.training_features = pd.DataFrame(
        {"num@": [1, 2, 3], "ca$t#": ["A", "B", "A"]}
    )
    training_data_prep.validation_features = pd.DataFrame(
        {"num@": [2, 3, 4], "cat#": ["B", "A", "B"]}
    )
    training_data_prep.testing_features = pd.DataFrame(
        {"num1%": [3, 4, 5], "cat#": ["C", "A", "B"]}
    )
    training_data_prep.train_features_preprocessed = pd.DataFrame(
        {"num^": [1, 2, 3], "c1at#": ["A", "B", "A"]}
    )
    training_data_prep.valid_features_preprocessed = pd.DataFrame(
        {"nu&u8": [2, 3, 4], "cat#": ["B", "A", "B"]}
    )

    training_data_prep.clean_up_feature_names()

    # Check if column names are cleaned up
    for df in [
        training_data_prep.training_features,
        training_data_prep.validation_features,
        training_data_prep.testing_features,
        training_data_prep.train_features_preprocessed,
        training_data_prep.valid_features_preprocessed,
    ]:
        for column in df.columns:
            assert re.match("^[A-Za-z0-9_]+$", column) is not None


def test_get_feature_names(training_data_prep):
    """Tests the get_feature_names method of the TrainingDataPrep class. The method
    should return the names of the numerical and categorical features in the training
    set.
    """

    num_features, cat_features = training_data_prep.get_feature_names()

    assert ["age"] == num_features
    assert ["gender"] == cat_features
