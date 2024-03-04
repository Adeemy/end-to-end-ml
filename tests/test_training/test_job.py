from datetime import datetime, timedelta

import numpy as np
import optuna
import optuna_distributed
import pandas as pd
import pytest
from comet_ml import Experiment
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)

from src.training.utils.job import ModelTrainer, VotingEnsembleCreator
from src.training.utils.model import ModelEvaluator, ModelOptimizer


@pytest.fixture
def model_trainer():
    train_features = pd.DataFrame(np.random.rand(10, 5))
    train_class = np.random.randint(2, size=10)
    valid_features = pd.DataFrame(np.random.rand(10, 5))
    valid_class = np.random.randint(2, size=10)
    train_features_preprocessed = pd.DataFrame(np.random.rand(10, 5))
    valid_features_preprocessed = pd.DataFrame(np.random.rand(10, 5))
    n_features = 5
    class_encoder = LabelEncoder().fit(train_class)
    preprocessor_step = ColumnTransformer(transformers=[])
    selector_step = VarianceThreshold()
    artifacts_path = "/path/to/artifacts"
    num_feature_names = ["f3", "f4", "f5"]
    cat_feature_names = ["f1", "f2"]
    fbeta_score_beta = 0.9888
    encoded_pos_class_label = 1

    return ModelTrainer(
        train_features=train_features,
        train_class=train_class,
        valid_features=valid_features,
        valid_class=valid_class,
        train_features_preprocessed=train_features_preprocessed,
        valid_features_preprocessed=valid_features_preprocessed,
        n_features=n_features,
        class_encoder=class_encoder,
        preprocessor_step=preprocessor_step,
        selector_step=selector_step,
        artifacts_path=artifacts_path,
        num_feature_names=num_feature_names,
        cat_feature_names=cat_feature_names,
        fbeta_score_beta=fbeta_score_beta,
        encoded_pos_class_label=encoded_pos_class_label,
    )


@pytest.fixture
def model_evaluator():
    """Create a ModelEvaluator instance for testing. This fixture is used by all the tests
    related to ModelEvaluator in this module.
    """

    comet_exp = Experiment(
        api_key="dummy_key", project_name="general", workspace="test"
    )
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        ("num", "passthrough", ["num_feature"]),
                        ("cat", OneHotEncoder(), ["cat_feature"]),
                    ]
                ),
            ),
            ("selector", SelectKBest(chi2, k=2)),
            ("classifier", LogisticRegression()),
        ]
    )
    train_features = pd.DataFrame(
        {"num_feature": [1, 2, 3], "cat_feature": ["a", "b", "a"]}
    )
    train_class = np.array([0, 1, 0])
    valid_features = pd.DataFrame(
        {"num_feature": [2, 3, 1], "cat_feature": ["b", "a", "b"]}
    )
    valid_class = np.array([1, 0, 1])
    return ModelEvaluator(
        comet_exp, pipeline, train_features, train_class, valid_features, valid_class
    )


def test_create_comet_experiment(model_trainer):
    """Tests if the comet experiment is created correctly. It creates a comet experiment
    locally because api key is incorrect. It checks if the experiment is created and
    the api key, project name and experiment name are set correctly.
    """

    comet_api_key = "test_api_key"
    comet_project_name = "test_project_name"
    comet_exp_name = "test_exp_name"

    result = model_trainer._create_comet_experiment(
        comet_api_key, comet_project_name, comet_exp_name
    )

    assert isinstance(result, Experiment)
    assert result.api_key == comet_api_key
    assert result.project_name == comet_project_name
    assert result.name == comet_exp_name


def test_optimize_model_in_serial_mode(mocker, model_trainer):
    """Tests if the _optimize_model method returns optuna study and optimizer when running in
    serial mode as expected. It mocks the experiment and model (to mock its predictions) and
    checks if the returned types are as expected. It also mocks the ModelOptimizer to prevent
    actual optimization that would require the model to be trained and evaluated (which is not
    the purpose of this test).
    """

    mock_experiment = mocker.MagicMock(spec=Experiment)
    mock_model = mocker.MagicMock(spec=LogisticRegression)

    search_space_params = {"C": [1, 10], "l1_ratio": [0.1, 0.9, True]}
    max_search_iters = 2
    n_parallel_jobs = 2
    model_opt_timeout_secs = 600
    is_voting_ensemble = False

    # Mock model predictions to avoid error when calculating metrics
    mock_model.predict.return_value = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])

    # Mock ModelOptimizer to return the mock study and prevent actual optimization that
    # would require the model to be trained and evaluated (which is not the purpose of this test)
    mock_optimizer = mocker.MagicMock(spec=ModelOptimizer)
    mock_study = mocker.MagicMock(spec=optuna.study.Study)
    mock_optimizer.tune_model_in_parallel.return_value = mock_study

    optimize_in_parallel = False
    result_study, result_optimizer = model_trainer._optimize_model(
        comet_exp=mock_experiment,
        model=mock_model,
        search_space_params=search_space_params,
        max_search_iters=max_search_iters,
        optimize_in_parallel=optimize_in_parallel,
        n_parallel_jobs=n_parallel_jobs,
        model_opt_timeout_secs=model_opt_timeout_secs,
        is_voting_ensemble=is_voting_ensemble,
    )
    trials_df = result_study.trials_dataframe()

    # Assert the returned types are as expected
    assert isinstance(result_study, optuna.study.Study)
    assert isinstance(result_optimizer, ModelOptimizer)
    with pytest.raises(AssertionError):
        assert result_study.trials_dataframe().shape[0] == max_search_iters + 1
    assert result_study.trials_dataframe().shape[0] == max_search_iters

    # Assert the trials dataframe has the expected columns
    assert all(
        "params_" + key in trials_df.columns for key in search_space_params.keys()
    )

    # Assert all trials has status COMPLETE and values are not NaN
    assert all(trials_df["state"] == "COMPLETE")
    assert not any(trials_df["value"].isna())


def test_optimize_model_in_parallel_mode(mocker, model_trainer):
    """Tests if the _optimize_model method returns optuna study and optimizer when running in
    parallel mode as expected. It mocks the experiment and model (to mock its predictions) and
    checks if the returned types are as expected. It also mocks the ModelOptimizer to prevent
    actual optimization that would require the model to be trained and evaluated (which is not
    the purpose of this test).
    """

    mock_experiment = mocker.MagicMock(spec=Experiment)
    mock_model = mocker.MagicMock(spec=LogisticRegression)

    search_space_params = {"C": [1, 10], "l1_ratio": [0.1, 0.9, True]}
    max_search_iters = 2
    n_parallel_jobs = 2
    model_opt_timeout_secs = 600
    is_voting_ensemble = False

    # Mock model predictions to avoid error when calculating metrics
    mock_model.predict.return_value = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])

    # Mock ModelOptimizer to return the mock study and prevent actual optimization that
    # would require the model to be trained and evaluated (which is not the purpose of this test)
    mock_optimizer = mocker.MagicMock(spec=ModelOptimizer)
    mock_study = mocker.MagicMock(spec=optuna_distributed.study.DistributedStudy)
    mock_optimizer.tune_model_in_parallel.return_value = mock_study

    # Inject the mock optimizer into the model trainer
    model_trainer.optimizer = mock_optimizer

    optimize_in_parallel = True
    result_study, result_optimizer = model_trainer._optimize_model(
        comet_exp=mock_experiment,
        model=mock_model,
        search_space_params=search_space_params,
        max_search_iters=max_search_iters,
        optimize_in_parallel=optimize_in_parallel,
        n_parallel_jobs=n_parallel_jobs,
        model_opt_timeout_secs=model_opt_timeout_secs,
        is_voting_ensemble=is_voting_ensemble,
    )
    trials_df = result_study.trials_dataframe()

    # Assert the returned types are as expected
    assert isinstance(result_study, optuna_distributed.study.DistributedStudy)
    assert isinstance(result_optimizer, ModelOptimizer)
    with pytest.raises(AssertionError):
        assert result_study.trials_dataframe().shape[0] == max_search_iters + 1
    assert result_study.trials_dataframe().shape[0] == max_search_iters

    # Assert the trials dataframe has the expected columns
    assert all(
        "params_" + key in trials_df.columns for key in search_space_params.keys()
    )

    # Assert all trials has status COMPLETE and values are not NaN
    assert all(trials_df["state"] == "COMPLETE")
    assert not any(trials_df["value"].isna())


def test_log_study_trials(mocker, model_trainer):
    """Tests if the _log_study_trials method logs the study trials to Comet as expected.
    It mocks the Experiment and Study objects and checks if the log_asset method was called
    with the expected arguments.
    """

    # Create required mock objects
    mock_comet_exp = mocker.MagicMock(spec=Experiment)
    mock_study = mocker.MagicMock(spec=optuna.study.Study)

    # Create a mock trials dataframe and save it in mock local path
    mock_trials_df = pd.DataFrame(
        {
            "number": np.arange(2),
            "value": np.random.rand(2),
            "datetime_start": [datetime.now() - timedelta(days=i) for i in range(2)],
            "datetime_complete": [
                datetime.now() - timedelta(days=i, hours=2) for i in range(2)
            ],
            "duration": [timedelta(hours=i) for i in range(2)],
            "params_C": np.random.rand(2),
            "params_l1_ratio": np.random.rand(2),
            "state": ["COMPLETE" if i % 2 == 0 else "INCOMPLETE" for i in range(2)],
        }
    )
    mock_study.trials_dataframe.return_value = mock_trials_df
    mock_trials_df.to_csv = mocker.MagicMock()  # Mock the to_csv method
    model_trainer.artifacts_path = "/path/to/artifacts"

    classifier_name = "classifier_name"
    model_trainer._log_study_trials(mock_comet_exp, mock_study, classifier_name)

    # Check that the log_asset method was called with the expected arguments
    mock_comet_exp.log_asset.assert_called_once_with(
        file_data=f"{model_trainer.artifacts_path}/study_{classifier_name}.csv",
        file_name=f"study_{classifier_name}",
    )


def test_fit_best_model(mocker, model_trainer):
    """Tests if the _fit_best_model method fits the pipeline as expected. It mocks
    the Study, ModelOptimizer, and Pipeline objects and checks if the fit_pipeline
    method was called with the expected arguments.
    """

    # Create required mock objects
    mock_optimizer = mocker.MagicMock(spec=ModelOptimizer)
    mock_study = mocker.MagicMock(spec=optuna.study.Study)

    # Set the train_features, preprocessor_step, and selector_step attributes of the instance
    mock_study.best_params = {"C": 100, "l1_ratio": 0.08888}
    model_trainer.train_features = pd.DataFrame(np.random.rand(10, 5))
    model_trainer.preprocessor_step = ColumnTransformer(transformers=[])
    model_trainer.selector_step = VarianceThreshold()

    # Create a pipeline and set it as the return value of the fit_pipeline method
    model = LogisticRegression()
    pipeline = Pipeline(
        steps=[
            ("preprocessor", model_trainer.preprocessor_step),
            ("selector", model_trainer.selector_step),
            ("classifier", model),
        ]
    )
    mock_optimizer.fit_pipeline.return_value = pipeline

    # Create a spy on the fit_pipeline method to track its calls
    spy = mocker.spy(mock_optimizer, "fit_pipeline")

    fitted_pipeline = model_trainer._fit_best_model(
        mock_study, mock_optimizer, model=model
    )

    assert isinstance(fitted_pipeline, Pipeline)

    # Assert that fit_pipeline was called with the correct parameters
    spy.assert_called_once_with(
        train_features=model_trainer.train_features,
        preprocessor_step=model_trainer.preprocessor_step,
        selector_step=model_trainer.selector_step,
        model=model,
    )

    # Assert that the last step is a model with the expected parameters
    model_params = fitted_pipeline.named_steps["classifier"].get_params()
    assert {
        k: model_params[k] for k in mock_study.best_params
    } == mock_study.best_params


def test_evaluate_model(mocker, model_trainer):
    """Tests if the _evaluate_model method returns the expected values. It mocks the
    Experiment, Pipeline, and ModelEvaluator objects and checks if the expected metrics
    and ECE values are returned. It also checks if the internal methods were called as
    expected.
    """

    # Create required mock objects
    mock_comet_exp = mocker.MagicMock(spec=Experiment)
    mock_model = mocker.MagicMock(spec=LogisticRegression)
    mock_preprocessor = mocker.MagicMock(ColumnTransformer)
    mock_selector = mocker.MagicMock(VarianceThreshold)
    mock_evaluator = mocker.MagicMock(spec=ModelEvaluator)
    mocker.patch("src.training.utils.model.ModelEvaluator", return_value=mock_evaluator)

    # Mock the pipeline and its fit, predict_proba methods and named_steps attribute
    mock_pipeline = mocker.MagicMock(spec=Pipeline)
    mock_pipeline.fit.return_value = None
    mock_pipeline.predict_proba.side_effect = lambda X: np.array(
        [[1, 0], [0, 1]] * (len(X) // 2) + [[1, 0]] * (len(X) % 2)
    )
    mock_pipeline.named_steps = {
        "preprocessor": mock_preprocessor,
        "selector": mock_selector,
        "classifier": mock_model,
    }
    mock_pipeline.classes_ = np.array([0, 1])
    mock_model.coef_ = np.array([[0.5, 0.5]])

    # Add transformers_ attribute as to the mock preprocessor must be fitted
    mock_preprocessor.transformers_ = [
        ("mock_transformer1", mocker.MagicMock(), [0]),
        ("mock_transformer2", mocker.MagicMock(), [1]),
    ]

    # Create spies for internal methods calls
    spy_extract_feature_importance = mocker.spy(
        ModelEvaluator, "extract_feature_importance"
    )
    spy_convert_metrics_from_df_to_dict = mocker.spy(
        ModelEvaluator, "convert_metrics_from_df_to_dict"
    )
    spy_calc_expected_calibration_error = mocker.spy(
        ModelEvaluator, "calc_expected_calibration_error"
    )

    mocker.patch("src.training.utils.model.ModelEvaluator", return_value=mock_evaluator)

    # Call the _evaluate_model method
    train_metric_values, valid_metric_values, model_ece = model_trainer._evaluate_model(
        comet_exp=mock_comet_exp, fitted_pipeline=mock_pipeline
    )

    # Check the structure and types of the output
    assert isinstance(train_metric_values, dict)
    assert isinstance(valid_metric_values, dict)
    assert isinstance(model_ece, (float, np.float64))

    expected_keys = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        f"f_{model_trainer.fbeta_score_beta}_score",
        "roc_auc",
    ]

    assert all("train_" + key in train_metric_values for key in expected_keys)
    assert all(
        isinstance(value, (float, np.float64)) for value in train_metric_values.values()
    )

    assert all("valid_" + key in valid_metric_values for key in expected_keys)
    assert all(
        isinstance(value, (float, np.float64)) for value in valid_metric_values.values()
    )

    # Check if internal methods were called
    mock_pipeline.predict_proba.assert_called()
    spy_extract_feature_importance.assert_called_once()
    spy_convert_metrics_from_df_to_dict.call_count == 2
    spy_calc_expected_calibration_error.assert_called_once()


def test_log_model_metrics(mocker, model_trainer):
    """Tests if the _log_model_metrics method logs the model metrics to Comet as expected.
    It mocks the Experiment object and checks if the log_metrics method was called with
    the expected arguments.
    """

    # Create required mock objects
    comet_exp = mocker.MagicMock(spec=Experiment)

    # Define the metric values and expected output
    train_metric_values = {"train_accuracy": 0.9, "train_loss": 0.1}
    valid_metric_values = pd.DataFrame(
        {"valid_accuracy": [0.8], "valid_loss": [0.2]}
    ).to_dict("records")[0]
    model_ece = 0.05

    expected_metrics = {
        "train_accuracy": 0.9,
        "train_loss": 0.1,
        "valid_accuracy": 0.8,
        "valid_loss": 0.2,
        "model_ece": 0.05,
    }

    model_trainer._log_model_metrics(
        comet_exp, train_metric_values, valid_metric_values, model_ece
    )

    # Check if the log_metrics method of the Experiment mock object was called with the correct parameters
    comet_exp.log_metrics.assert_called_once_with(expected_metrics)
