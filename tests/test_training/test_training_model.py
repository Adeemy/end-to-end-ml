import numpy as np
import optuna
import optuna_distributed
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.training.utils.model import ModelOptimizer


@pytest.fixture
def model_optimizer(mocker):
    """Create a ModelOptimizer instance for testing. This fixture is used by all the tests
    in this module.
    """

    # Create mock objects for the required attributes
    comet_exp = mocker.MagicMock()
    train_features_preprocessed = mocker.create_autospec(pd.DataFrame)
    train_class = mocker.create_autospec(np.ndarray)
    valid_features_preprocessed = mocker.create_autospec(pd.DataFrame)
    valid_class = mocker.create_autospec(np.ndarray)
    n_features = 10
    model = mocker.create_autospec(LogisticRegression)
    search_space_params = {"C": [1, 10], "l1_ratio": [0.1, 0.9, True]}
    fbeta_score_beta = 1.0
    encoded_pos_class_label = 1
    is_voting_ensemble = False

    model_optimizer = ModelOptimizer(
        comet_exp,
        train_features_preprocessed,
        train_class,
        valid_features_preprocessed,
        valid_class,
        n_features,
        model,
        search_space_params,
        fbeta_score_beta,
        encoded_pos_class_label,
        is_voting_ensemble,
    )

    return model_optimizer


def test_model_optimizer_init(model_optimizer):
    """Tests that the ModelOptimizer instance is created with the expected attributes.
    This test also checks that the attributes are not None.
    """

    assert model_optimizer.comet_exp is not None
    assert model_optimizer.train_features_preprocessed is not None
    assert model_optimizer.train_class is not None
    assert model_optimizer.valid_features_preprocessed is not None
    assert model_optimizer.valid_class is not None
    assert model_optimizer.n_features == 10
    assert model_optimizer.model is not None
    assert model_optimizer.search_space_params == {
        "C": [1, 10],
        "l1_ratio": [0.1, 0.9, True],
    }
    assert model_optimizer.fbeta_score_beta == 1.0
    assert model_optimizer.encoded_pos_class_label == 1
    assert model_optimizer.is_voting_ensemble == False


def test_model_optimizer_generate_trial_params(mocker, model_optimizer):
    """Tests that the generate_trial_params method returns a non-empty dictionary."""

    # Create a mock optuna trial object
    trial = mocker.create_autospec(optuna.trial.Trial)

    # Call the generate_trial_params method
    params = model_optimizer.generate_trial_params(trial)

    # Check that the returned params dictionary is not empty
    assert params
    assert isinstance(params, dict)


def test_model_optimizer_calc_perf_metrics(model_optimizer):
    """Tests that the calc_perf_metrics method returns a non-empty dataframe."""

    # Create mock arrays for true_class and pred_class
    true_class = [0, 1, 0, 1, 0]
    pred_class = [0, 1, 1, 1, 0]

    performance_metrics = model_optimizer.calc_perf_metrics(true_class, pred_class)

    # Check that the returned performance_metrics dataframe is not empty
    assert not performance_metrics.empty
    assert isinstance(performance_metrics, pd.DataFrame)

    # Check that the returned performance_metrics dataframe has the expected metrics
    assert performance_metrics["Metric"].to_list() == [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f_1.0_score",
        "roc_auc",
    ]


def test_model_optimizer_objective_function(mocker, model_optimizer):
    """Tests that the objective_function method returns a float value.
    This test also mocks the suggest_float method of the optuna trial object.
    """

    # Create a mock optuna trial object
    trial = mocker.create_autospec(optuna.trial.Trial)

    # Configure the suggest_float method of the trial mock object
    trial.suggest_float.return_value = 0.5

    # Mock the generate_trial_params method
    model_optimizer.generate_trial_params = mocker.MagicMock()
    model_optimizer.generate_trial_params.return_value = {
        "C": 1,
        "l1_ratio": 2,
    }  # replace with actual params

    # Set train_features_preprocessed and train_class attributes
    model_optimizer.train_features_preprocessed = pd.DataFrame(
        {"feat1": [5, 0], "feat2": [8, 7]}
    )
    model_optimizer.train_class = np.array([0, 1])

    model_optimizer.valid_features_preprocessed = pd.DataFrame(
        {"feat1": [3, 4], "feat2": [6, 5]}
    )
    model_optimizer.valid_class = np.array([1, 0])

    # Mock the predict method of the model to return predictions of the same length as the true labels
    model_optimizer.model.predict = mocker.MagicMock(return_value=np.array([0, 1]))

    # Call the objective_function method
    valid_score = float(model_optimizer.objective_function(trial))

    # Check that the returned valid_score is a float
    assert isinstance(valid_score, float)


def test_model_optimizer_tune_model(mocker, model_optimizer):
    """Tests that the tune_model method returns a non-empty optuna study object. This test also
    mocks the predict method of the model to return predictions of the same length as the true labels.
    """

    # Mock required attributes for tune_model method
    model_optimizer.train_class = np.array([0, 1])
    model_optimizer.valid_class = np.array([1, 0])

    # Mock the predict method of the model to return predictions of the same length as the true labels
    model_optimizer.model.predict = mocker.MagicMock(return_value=np.array([0, 1]))

    # Call the tune_model method
    study = model_optimizer.tune_model()

    # Check that the returned study object is not None
    assert study
    assert isinstance(study, optuna.study.Study)


def test_model_optimizer_tune_model_in_parallel(mocker, model_optimizer):
    """Tests that the tune_model method returns a non-empty optuna study object. This test also
    mocks the predict method of the model to return predictions of the same length as the true labels.
    """

    # Mock required attributes for tune_model_in_parallel method
    model_optimizer.train_class = np.array([0, 1])
    model_optimizer.valid_class = np.array([1, 0])

    # Mock the predict method of the model to return predictions of the same length as the true labels
    model_optimizer.model.predict = mocker.MagicMock(return_value=np.array([0, 1]))

    # Call the tune_model method
    study = model_optimizer.tune_model_in_parallel()

    # Check that the returned study object is not None
    assert study
    assert isinstance(study, optuna_distributed.study.DistributedStudy)


def test_create_pipeline():
    """Tests that the create_pipeline method returns a pipeline with the expected steps."""

    # Create mock preprocessor, selector, and model
    preprocessor_step = ColumnTransformer(transformers=[])
    selector_step = VarianceThreshold()
    model = LogisticRegression()

    pipeline = ModelOptimizer.create_pipeline(preprocessor_step, selector_step, model)

    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 3
    assert pipeline.steps[0][0] == "preprocessor"
    assert isinstance(pipeline.steps[0][1], ColumnTransformer)
    assert pipeline.steps[1][0] == "selector"
    assert isinstance(pipeline.steps[1][1], VarianceThreshold)
    assert pipeline.steps[2][0] == "classifier"
    assert isinstance(pipeline.steps[2][1], LogisticRegression)


def test_fit_pipeline(mocker, model_optimizer):
    """Tests that the fit_pipeline method returns a pipeline with the expected steps.
    This test also mocks the fit method of the pipeline to track its calls, i.e.,
    interaction testing.
    """

    # Set the required attributes
    model_optimizer.train_class = np.random.randint(0, 2, size=10)
    model_optimizer.n_features = 5

    # Create mock train_features, preprocessor, selector, and model
    train_features = pd.DataFrame(
        {
            "feat1": np.random.rand(10),
            "feat2": np.random.rand(10),
            "feat3": np.random.rand(10),
            "feat4": np.random.rand(10),
            "feat5": np.random.rand(10),
        }
    )
    preprocessor_step = ColumnTransformer(
        transformers=[
            (
                "passthrough",
                FunctionTransformer(),
                ["feat1", "feat2", "feat3", "feat4", "feat5"],
            )
        ]
    )
    selector_step = VarianceThreshold(
        threshold=0
    )  # threshold=0 means no features will be removed
    model = LogisticRegression()

    # Create a spy on the fit_pipeline method to track its calls
    spy = mocker.spy(model_optimizer, "fit_pipeline")

    # Call the fit_pipeline method
    pipeline = model_optimizer.fit_pipeline(
        train_features, preprocessor_step, selector_step, model
    )

    # Check that the fit method of the pipeline was called with the correct arguments
    called_args = spy.call_args

    # Check that the returned pipeline is not None
    assert isinstance(pipeline, Pipeline)

    # Check if fit_pipeline was called with the expected arguments (interaction testing assertions)
    assert isinstance(called_args[0][0], pd.DataFrame)
    assert called_args[0][0].equals(train_features)
    assert isinstance(called_args[0][1], ColumnTransformer)
    assert isinstance(called_args[0][2], VarianceThreshold)
    assert isinstance(called_args[0][3], LogisticRegression)

    # Check if fit_pipeline fits the logistic regression model
    assert pipeline.named_steps["classifier"].coef_ is not None
