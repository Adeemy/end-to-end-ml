import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.training.utils.data import DataPipelineCreator, PrepTrainingData


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

    # Create an instance of PrepTrainingData
    training_data_prep = PrepTrainingData(
        train_set=train_set,
        test_set=test_set,
        primary_key="id",
        class_col_name="class",
        numerical_feature_names=["age"],
        categorical_feature_names=["gender"],
    )

    return training_data_prep


def test_select_relevant_columns(training_data_prep):
    """Tests the select_relevant_columns method of the PrepTrainingData class. The method
    should select the relevant columns from the train and test sets. The relevant columns
    are the primary key, numerical features, categorical features, and the target variable
    specified in PrepTrainingData initialization. The method should remove any other
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
    """Tests the enforce_data_types method of the PrepTrainingData class. The method
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
    training_data_prep.create_validation_set()

    # Check if the validation set is created
    assert training_data_prep.valid_set is not None


def test_extract_features(training_data_prep):
    # Create validation set
    training_data_prep.create_validation_set()

    # Call the extract_features method
    training_data_prep.extract_features()

    # Check if the training, testing, and validation features are extracted correctly,
    # which should not include the primary key and target variable
    assert list(training_data_prep.training_features.columns) == ["age", "gender"]
    assert list(training_data_prep.validation_features.columns) == ["age", "gender"]
    assert list(training_data_prep.testing_features.columns) == ["age", "gender"]


@pytest.fixture
def test_dataset():
    """Creates a test dataset with numerical and categorical features. It also includes a near-zero variance feature
    to test selector step."""

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


def test_extract_col_names_after_preprocessing(test_dataset):
    """Tests the extraction of column names after preprocessing. The test dataset
    includes numerical and categorical features, and a near-zero variance feature
    to test the selector step. The expected output is a list with the names of the
    numerical and categorical features. The near-zero variance feature should be removed.
    """

    df, num_feat_col_names, cat_feat_col_names = test_dataset

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


def test_create_data_pipeline(test_dataset):
    """Tests the creation of a data pipeline. The test dataset includes numerical and
    categorical features, and a near-zero variance feature to test the selector step.
    The expected output is a dataframe with the preprocessed features and a pipeline
    with the preprocessing steps.
    """

    df, num_feat_col_names, cat_feat_col_names = test_dataset

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
