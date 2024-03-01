import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.training.utils.data import DataPipelineCreator


@pytest.fixture
def test_dataset():
    """Create a test dataset with numerical and categorical features. It also includes a near-zero variance feature
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
