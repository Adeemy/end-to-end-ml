import matplotlib.pyplot as plt
import numpy as np
import optuna
import optuna_distributed
import pandas as pd
import pytest
from comet_ml import Experiment
from sklearn.calibration import CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder

from src.training.utils.model import ModelEvaluator, ModelOptimizer


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


def test_plot_feature_importance(model_evaluator):
    """Tests that the plot_feature_importance method returns a matplotlib figure object."""

    feature_importance_scores = np.array([0.1, 0.2])
    feature_names = ["num_feature", "cat_feature"]
    figure_obj = plt.figure()
    fig = model_evaluator.plot_feature_importance(
        feature_importance_scores,
        feature_names,
        figure_obj,
        n_top_features=len(feature_importance_scores),
    )
    assert isinstance(fig, plt.Figure)

    # Check contents of the figure
    ax = fig.axes[0]
    assert ax.get_title() == f"Top {len(feature_importance_scores)} important features"
    assert ax.get_xlabel() == "Contribution"
    assert ax.get_ylabel() == "Feature Name"

    # Check if the function's behavior with different inputs
    with pytest.raises(IndexError):
        model_evaluator.plot_feature_importance(np.array([]), [], figure_obj)
    with pytest.raises(TypeError):
        model_evaluator.plot_feature_importance(
            np.array([0.1, 0.2]), [], n_top_features=3
        )
    with pytest.raises(TypeError):
        model_evaluator.plot_feature_importance(
            np.array([0.1, 0.2]), ["num_feature", "cat_feature"], fig_size=(0, 0)
        )


def test_extract_feature_importance(model_evaluator):
    """Tests that the extract_feature_importance method returns feature
    importance scores for the specified features. This test also checks
    that the returned feature importance scores are of the expected length.
    """

    num_feature_names = ["num_feature"]
    cat_feature_names = ["cat_feature"]

    model_evaluator.pipeline.fit(
        model_evaluator.train_features, model_evaluator.train_class
    )
    model_evaluator.extract_feature_importance(
        model_evaluator.pipeline, num_feature_names, cat_feature_names
    )

    # Check if two coefficients (two features) are extracted
    assert len(model_evaluator.pipeline.named_steps["classifier"].coef_[0]) == len(
        num_feature_names + cat_feature_names
    )


def test__log_feature_importance_fig(mocker, model_evaluator):
    """Tests that the _log_feature_importance_fig method logs the
    feature importance figure. It should not log the figure if the classifier
    name is 'VotingClassifier' but it should log the figure otherwise.
    """

    model_evaluator.comet_exp = mocker.MagicMock()

    # Check if the feature importance fig is not logged if classifier name is 'VotingClassifier'
    model_evaluator._log_feature_importance_fig(
        "VotingClassifier", np.array([0.1, 0.2]), ["num_feature", "cat_feature"]
    )
    model_evaluator.comet_exp.log_figure.assert_not_called()  # no figure logged

    # Check if the feature importance fig is logged
    model_evaluator._log_feature_importance_fig(
        "LogisticRegression",
        np.array([0.1, 0.2]),
        ["num_feature", "cat_feature"],
        2,
        (24, 36),
        10.0,
        "Feature Importance",
    )
    model_evaluator.comet_exp.log_figure.assert_called_once()


def test_plot_roc_curve():
    """Tests that the plot_roc_curve method returns a matplotlib figure object. This test also
    checks that the AUC is correctly calculated and displayed in the plot.
    """
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0.1, 0.9, 0.2, 0.2, 0.1])
    expected_auc = roc_auc_score(y_true, y_pred)

    fig = ModelEvaluator.plot_roc_curve(y_true, y_pred, (6, 6))
    ax = fig.axes[0]
    lines = ax.get_lines()

    assert isinstance(fig, plt.Figure)

    # There should be two lines: the ROC curve and the diagonal
    assert len(lines) == 2

    # The AUC should be correctly calculated
    assert f"ROC curve (AUC = {expected_auc:.2f})" in lines[0].get_label()


def test_plot_precision_recall_curve():
    """Tests that the plot_precision_recall_curve method returns a matplotlib figure object. This test also
    checks that the average precision is correctly calculated and displayed in the plot.
    """

    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.1])

    fig = ModelEvaluator.plot_precision_recall_curve(y_true, y_pred)
    ax = fig.axes[0]
    lines = ax.get_lines()
    expected_ap = average_precision_score(y_true, y_pred)

    assert isinstance(fig, plt.Figure)

    assert ax.get_title() == "Precision-Recall Curve"
    assert ax.get_xlabel() == "Recall"
    assert ax.get_ylabel() == "Precision"

    # There should be one line: the precision-recall curve
    assert len(lines) == 1

    # The AP should be correctly calculated
    assert f"Precision-Recall curve (AP = {expected_ap:.2f})" in lines[0].get_label()


def test_plot_cumulative_gains():
    """Tests that the plot_cumulative_gains method returns a matplotlib figure object. This test also
    checks that the plot has the expected title, x-axis label, and y-axis label. The plot should also
    contain four lines: the precision-recall curve, the model, the wizard, and the random line.
    """

    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.1])

    fig = ModelEvaluator.plot_cumulative_gains(y_true, y_pred)
    ax = fig.axes[0]
    lines = ax.get_lines()

    assert isinstance(fig, plt.Figure)

    # Check the contents of the plot
    assert ax.get_title() == "Cumulative Gain Plot"
    assert ax.get_xlabel() == "Deciles"
    assert ax.get_ylabel() == "% Resonders"

    # There should be four lines: 'Precision-Recall curve (AP = 1.00)', 'Model', 'Wizard', 'Random'
    assert len(lines) == 4
    assert "Precision-Recall curve (AP = 1.00)" in [line.get_label() for line in lines]
    assert "Model" in [line.get_label() for line in lines]
    assert "Wizard" in [line.get_label() for line in lines]
    assert "Random" in [line.get_label() for line in lines]


def test_plot_lift_curve():
    """Tests that the plot_cumulative_gains method returns a matplotlib figure object. This test also
    checks that the plot has the expected title, x-axis label, and y-axis label. The plot should also
    contain six lines: the precision-recall curve, the model, the wizard, the random line, the model,
    and the random line.
    """

    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.1])

    fig = ModelEvaluator.plot_lift_curve(y_true, y_pred)
    ax = fig.axes[0]
    lines = ax.get_lines()

    assert isinstance(fig, plt.Figure)

    # Check the contents of the plot
    assert ax.get_title() == "Lift Plot"
    assert ax.get_xlabel() == "Deciles"
    assert ax.get_ylabel() == "Lift"

    # There should be six lines: Precision-Recall curve (AP = 1.00)', 'Model', 'Wizard', 'Random', 'Model', 'Random'
    assert len(lines) == 6
    assert "Precision-Recall curve (AP = 1.00)" in [line.get_label() for line in lines]
    assert "Model" in [line.get_label() for line in lines]
    assert "Wizard" in [line.get_label() for line in lines]
    assert "Random" in [line.get_label() for line in lines]


def test_convert_metrics_from_df_to_dict():
    """Tests that the convert_metrics_from_df_to_dict method returns a dictionary with the expected keys and values.
    This test also checks that the returned dictionary has the expected keys and values when a prefix is provided.
    """

    scores = pd.DataFrame(
        {"Metric": ["accuracy", "precision", "recall"], "Score": [0.9, 0.8, 0.7]}
    )

    # Call the function without a prefix
    metrics_values = ModelEvaluator.convert_metrics_from_df_to_dict(scores)
    assert metrics_values == {"accuracy": 0.9, "precision": 0.8, "recall": 0.7}

    # Call the function with a prefix
    metrics_values = ModelEvaluator.convert_metrics_from_df_to_dict(
        scores, prefix="test_"
    )
    assert metrics_values == {
        "test_accuracy": 0.9,
        "test_precision": 0.8,
        "test_recall": 0.7,
    }


def test_evaluate_model_perf(mocker, model_evaluator):
    """Tests that the evaluate_model_perf method returns the expected performance metrics dataframes.
    This test also mocks the methods that evaluate_model_perf calls internally. The pipeline is also
    mocked because it must be fitted before calling evaluate_model_perf. The LabelEncoder is also mocked
    because it's required to call _log_confusion_matrix. The function calc_perf_metrics is also mocked
    to return a dataframe with the expected metrics. To ensure the test is isolated, functions called by
    evaluate_model_perf are mocked to return the expected results and not tested in this test.
    """

    # Create a mock pipeline with a predict_proba method
    mock_pipeline = mocker.MagicMock(Pipeline)
    mock_pipeline.predict_proba.return_value = np.array(
        [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]]
    )
    mock_pipeline.fit.return_value = None

    # Set pipeline to mock pipeline because it must be fitted before calling evaluate_model_perf
    model_evaluator.pipeline = mock_pipeline

    # Create a mock LabelEncoder as it's required to call _log_confusion_matrix
    mock_encoder = mocker.MagicMock(LabelEncoder)
    mock_encoder.inverse_transform.return_value = ["class1", "class2"]

    # Mock the methods that evaluate_model_perf calls internally to return the expected results
    mocker.patch.object(
        model_evaluator, "_get_pred_class", return_value=np.array([0, 1])
    )
    mocker.patch.object(
        model_evaluator,
        "calc_perf_metrics",
        return_value=pd.DataFrame({"accuracy": [0.5]}),
    )

    # Call the function with the mock objects
    train_scores, valid_scores = model_evaluator.evaluate_model_perf(
        class_encoder=mock_encoder
    )

    # Assert that the function returns the expected results
    assert train_scores.equals(pd.DataFrame({"accuracy": [0.5]}))
    assert valid_scores.equals(pd.DataFrame({"accuracy": [0.5]}))


def test_get_pred_class(model_evaluator):
    """Tests that the _get_pred_class method returns the expected class labels. This test also
    checks that the method raises an AssertionError if the threshold is not between 0 and 1.
    """

    # Define test inputs and expected output
    model_evaluator.encoded_pos_class_label = (
        1  # assuming positive class is encoded as 1
    )
    pred_probs = np.array([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])
    threshold = 0.5
    expected_output = np.array([1, 0, 1])

    pred_class = model_evaluator._get_pred_class(pred_probs, threshold)

    assert np.array_equal(
        pred_class, expected_output
    ), "The method did not return the expected class labels"

    # Test the method with different threshold
    with pytest.raises(AssertionError):
        new_pred_class = model_evaluator._get_pred_class(pred_probs, threshold=0.15)
        assert np.array_equal(new_pred_class, expected_output)


def test_get_original_class_labels(mocker, model_evaluator):
    """Tests that the _get_original_class_labels method returns the expected original class
    labels and class labels. This test also mocks the pipeline attribute of the ModelEvaluator
    instance because it's required to call the method.
    """

    # Create a mock pipeline with a classes_ attribute
    mock_pipeline = mocker.MagicMock(Pipeline)
    mock_pipeline.classes_ = np.array([0, 1])
    model_evaluator.pipeline = mock_pipeline
    class_encoder = LabelEncoder()
    class_encoder.fit(["class0", "class1"])

    # Define test inputs
    true_class = np.array([0, 1, 1])
    pred_class = np.array([1, 0, 1])

    (
        true_original_class,
        pred_original_class,
        original_class_labels,
    ) = model_evaluator._get_original_class_labels(
        true_class, pred_class, class_encoder
    )

    # Define the expected output
    expected_true_original_class = np.array(["class0", "class1", "class1"])
    expected_pred_original_class = np.array(["class1", "class0", "class1"])
    expected_original_class_labels = ["class0", "class1"]

    # Assert that the method output matches the expected output
    assert np.array_equal(
        true_original_class, expected_true_original_class
    ), "The method did not return the expected true original class labels"
    assert np.array_equal(
        pred_original_class, expected_pred_original_class
    ), "The method did not return the expected predicted original class labels"
    assert (
        original_class_labels == expected_original_class_labels
    ), "The method did not return the expected original class labels"


def test_log_confusion_matrix(mocker, model_evaluator):
    """Tests that the _log_confusion_matrix method logs the confusion matrix. This test also
    checks that the method logs the confusion matrix the correct number of times.
    """

    # Create a mock comet_exp to test number of calls by the method
    mock_comet_exp = mocker.MagicMock(Experiment)
    model_evaluator.comet_exp = mock_comet_exp

    # Define test inputs
    original_train_class = np.array(["class0", "class1", "class1"])
    pred_original_train_class = np.array(["class1", "class0", "class1"])
    original_valid_class = np.array(["class0", "class1", "class1"])
    pred_original_valid_class = np.array(["class1", "class0", "class1"])
    original_class_labels = ["class0", "class1"]

    model_evaluator._log_confusion_matrix(
        original_train_class,
        pred_original_train_class,
        original_valid_class,
        pred_original_valid_class,
        original_class_labels,
    )

    # Assert that the log_confusion_matrix method was called the correct number of times
    assert (
        mock_comet_exp.log_confusion_matrix.call_count == 4
    ), f"""The log_confusion_matrix method was not called the correct number of times 
    ({mock_comet_exp.log_confusion_matrix.call_count} calls)."""


def test_log_calibration_curve(mocker, model_evaluator):
    """Tests that the _log_calibration_curve method logs the calibration curve. This test also
    checks that the method logs the calibration curve the correct number of times.
    """

    # Create a mock comet_exp to test number of calls by the method
    mock_comet_exp = mocker.MagicMock(Experiment)
    model_evaluator.comet_exp = mock_comet_exp

    # Define test inputs
    model_evaluator.valid_class = np.array([0, 1, 1])
    model_evaluator.encoded_pos_class_label = (
        1  # assuming positive class is encoded as 1
    )
    pred_probs = np.array([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])

    # Call the method with the test inputs
    model_evaluator._log_calibration_curve(pred_probs)

    # Assert that the log_figure method was called once
    assert (
        mock_comet_exp.log_figure.call_count == 1
    ), f"""The log_figure method was not called the correct number of times 
    ({mock_comet_exp.log_figure.call_count} calls)"""


def test_log_roc_curve(mocker, model_evaluator):
    """Tests that the _log_roc_curve method logs the ROC curve. This test also checks that the method
    logs the ROC curve the correct number of times.
    """

    # Create a mock comet_exp to test number of calls by the method
    mock_comet_exp = mocker.MagicMock(Experiment)
    model_evaluator.comet_exp = mock_comet_exp

    # Define test inputs
    model_evaluator.valid_class = np.array([0, 1, 1])
    encoded_pos_class_label = 1  # assuming positive class is encoded as 1
    pred_probs = np.array([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])

    # Mock the plot_roc_curve method to return a MagicMock (which will stand in for the figure)
    mock_figure = mocker.MagicMock(plt.Figure)
    with mocker.patch.object(
        ModelEvaluator, "plot_roc_curve", return_value=mock_figure
    ):
        # Call the method with the test inputs
        model_evaluator._log_roc_curve(pred_probs, encoded_pos_class_label)

    # Assert that the log_figure method was called once with the correct arguments
    mock_comet_exp.log_figure.assert_called_once_with(
        figure_name="ROC Curve", figure=mock_figure, overwrite=True
    )


def test_log_precision_recall_curve(monkeypatch, mocker, model_evaluator):
    """Tests that the _log_precision_recall_curve method logs the precision-recall curve. This test also
    checks that the method logs the precision-recall curve the correct number of times.
    """

    # Create a mock comet_exp to test number of calls by the method
    mock_comet_exp = mocker.MagicMock(Experiment)
    model_evaluator.comet_exp = mock_comet_exp

    # Define test inputs
    model_evaluator.valid_class = np.array([0, 1, 1])
    encoded_pos_class_label = 1  # assuming positive class is encoded as 1
    pred_probs = np.array([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])

    # Mock the plot_precision_recall_curve method to return a MagicMock (which will stand in for the figure)
    mock_figure = mocker.MagicMock(plt.Figure)
    monkeypatch.setattr(
        model_evaluator,
        "plot_precision_recall_curve",
        lambda *args, **kwargs: mock_figure,
    )

    # Call the method with the test inputs
    model_evaluator._log_precision_recall_curve(pred_probs, encoded_pos_class_label)

    # Assert that the log_figure method was called once with the correct arguments
    mock_comet_exp.log_figure.assert_called_once_with(
        figure_name="Precision-Recall Curve", figure=mock_figure, overwrite=True
    )


def test_log_cumulative_gains(mocker, model_evaluator):
    """Tests that the _log_cumulative_gains method logs the cumulative gains plot. This test also
    checks that the method logs the cumulative gains plot the correct number of times.
    """

    # Create a mock comet_exp to test number of calls by the method
    mock_comet_exp = mocker.MagicMock(Experiment)
    model_evaluator.comet_exp = mock_comet_exp

    # Define test inputs
    valid_class = np.array([0, 1, 1])
    pred_probs = np.array([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])
    model_evaluator.encoded_pos_class_label = (
        1  # Assuming positive class is encoded as 1
    )

    # Mock the plot_cumulative_gains method to return a MagicMock (which will stand in for the figure)
    mock_figure = mocker.MagicMock(plt.Figure)
    mocker.patch.object(
        model_evaluator, "plot_cumulative_gains", return_value=mock_figure
    )

    # Call the method with the test inputs
    model_evaluator._log_cumulative_gains(pred_probs, valid_class)

    # Assert that the log_figure method was called once with the correct arguments
    mock_comet_exp.log_figure.assert_called_once_with(
        figure_name="Cumulative Gain", figure=mock_figure, overwrite=True
    )


def test_log_lift_curve(mocker, model_evaluator):
    """Tests that the _log_lift_curve method logs the lift curve. This test also checks that the method
    logs the lift curve the correct number of times.
    """

    # Create a mock comet_exp to test number of calls by the method
    mock_comet_exp = mocker.MagicMock(Experiment)
    model_evaluator.comet_exp = mock_comet_exp

    # Define test inputs
    valid_class = np.array([0, 1, 1])
    pred_probs = np.array([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])
    model_evaluator.encoded_pos_class_label = (
        1  # Assuming positive class is encoded as 1
    )

    # Mock the plot_lift_curve method to return a MagicMock (which will stand in for the figure)
    mock_figure = mocker.MagicMock(plt.Figure)
    mocker.patch.object(model_evaluator, "plot_lift_curve", return_value=mock_figure)
    model_evaluator._log_lift_curve(pred_probs, valid_class)

    # Assert that the log_figure method was called once with the correct arguments
    mock_comet_exp.log_figure.assert_called_once_with(
        figure_name="Lift Curve", figure=mock_figure, overwrite=True
    )


# @pytest.fixture
# def model_optimizer(mocker):
#     """Create a ModelOptimizer instance for testing. This fixture is used by all the tests
#     related to ModelOptimizer in this module.
#     """

#     # Create mock objects for the required attributes
#     comet_exp = mocker.MagicMock()
#     train_features_preprocessed = mocker.create_autospec(pd.DataFrame)
#     train_class = mocker.create_autospec(np.ndarray)
#     valid_features_preprocessed = mocker.create_autospec(pd.DataFrame)
#     valid_class = mocker.create_autospec(np.ndarray)
#     n_features = 10
#     model = mocker.create_autospec(LogisticRegression)
#     search_space_params = {"C": [1, 10], "l1_ratio": [0.1, 0.9, True]}
#     fbeta_score_beta = 1.0
#     encoded_pos_class_label = 1
#     is_voting_ensemble = False

#     model_optimizer = ModelOptimizer(
#         comet_exp,
#         train_features_preprocessed,
#         train_class,
#         valid_features_preprocessed,
#         valid_class,
#         n_features,
#         model,
#         search_space_params,
#         fbeta_score_beta,
#         encoded_pos_class_label,
#         is_voting_ensemble,
#     )

#     return model_optimizer


# def test_model_optimizer_init(model_optimizer):
#     """Tests that the ModelOptimizer instance is created with the expected attributes.
#     This test also checks that the attributes are not None.
#     """

#     assert model_optimizer.comet_exp is not None
#     assert model_optimizer.train_features_preprocessed is not None
#     assert model_optimizer.train_class is not None
#     assert model_optimizer.valid_features_preprocessed is not None
#     assert model_optimizer.valid_class is not None
#     assert model_optimizer.n_features == 10
#     assert model_optimizer.model is not None
#     assert model_optimizer.search_space_params == {
#         "C": [1, 10],
#         "l1_ratio": [0.1, 0.9, True],
#     }
#     assert model_optimizer.fbeta_score_beta == 1.0
#     assert model_optimizer.encoded_pos_class_label == 1
#     assert model_optimizer.is_voting_ensemble == False


# def test_model_optimizer_generate_trial_params(mocker, model_optimizer):
#     """Tests that the generate_trial_params method returns a non-empty dictionary."""

#     # Create a mock optuna trial object
#     trial = mocker.create_autospec(optuna.trial.Trial)

#     # Call the generate_trial_params method
#     params = model_optimizer.generate_trial_params(trial)

#     # Check that the returned params dictionary is not empty
#     assert params
#     assert isinstance(params, dict)


# def test_model_optimizer_calc_perf_metrics(model_optimizer):
#     """Tests that the calc_perf_metrics method returns a non-empty dataframe."""

#     # Create mock arrays for true_class and pred_class
#     true_class = [0, 1, 0, 1, 0]
#     pred_class = [0, 1, 1, 1, 0]

#     performance_metrics = model_optimizer.calc_perf_metrics(true_class, pred_class)

#     # Check that the returned performance_metrics dataframe is not empty
#     assert not performance_metrics.empty
#     assert isinstance(performance_metrics, pd.DataFrame)

#     # Check that the returned performance_metrics dataframe has the expected metrics
#     assert performance_metrics["Metric"].to_list() == [
#         "accuracy",
#         "precision",
#         "recall",
#         "f1",
#         "f_1.0_score",
#         "roc_auc",
#     ]


# def test_model_optimizer_objective_function(mocker, model_optimizer):
#     """Tests that the objective_function method returns a float value.
#     This test also mocks the suggest_float method of the optuna trial object.
#     """

#     # Create a mock optuna trial object
#     trial = mocker.create_autospec(optuna.trial.Trial)

#     # Configure the suggest_float method of the trial mock object
#     trial.suggest_float.return_value = 0.5

#     # Mock the generate_trial_params method
#     model_optimizer.generate_trial_params = mocker.MagicMock()
#     model_optimizer.generate_trial_params.return_value = {
#         "C": 1,
#         "l1_ratio": 2,
#     }  # replace with actual params

#     # Set train_features_preprocessed and train_class attributes
#     model_optimizer.train_features_preprocessed = pd.DataFrame(
#         {"feat1": [5, 0], "feat2": [8, 7]}
#     )
#     model_optimizer.train_class = np.array([0, 1])

#     model_optimizer.valid_features_preprocessed = pd.DataFrame(
#         {"feat1": [3, 4], "feat2": [6, 5]}
#     )
#     model_optimizer.valid_class = np.array([1, 0])

#     # Mock the predict method of the model to return predictions of the same length as the true labels
#     model_optimizer.model.predict = mocker.MagicMock(return_value=np.array([0, 1]))

#     # Call the objective_function method
#     valid_score = float(model_optimizer.objective_function(trial))

#     # Check that the returned valid_score is a float
#     assert isinstance(valid_score, float)


# def test_model_optimizer_tune_model(mocker, model_optimizer):
#     """Tests that the tune_model method returns a non-empty optuna study object. This test also
#     mocks the predict method of the model to return predictions of the same length as the true labels.
#     """

#     # Mock required attributes for tune_model method
#     model_optimizer.train_class = np.array([0, 1])
#     model_optimizer.valid_class = np.array([1, 0])

#     # Mock the predict method of the model to return predictions of the same length as the true labels
#     model_optimizer.model.predict = mocker.MagicMock(return_value=np.array([0, 1]))

#     # Call the tune_model method
#     study = model_optimizer.tune_model()

#     # Check that the returned study object is not None
#     assert study
#     assert isinstance(study, optuna.study.Study)


# def test_model_optimizer_tune_model_in_parallel(mocker, model_optimizer):
#     """Tests that the tune_model method returns a non-empty optuna study object. This test also
#     mocks the predict method of the model to return predictions of the same length as the true labels.
#     """

#     # Mock required attributes for tune_model_in_parallel method
#     model_optimizer.train_class = np.array([0, 1])
#     model_optimizer.valid_class = np.array([1, 0])

#     # Mock the predict method of the model to return predictions of the same length as the true labels
#     model_optimizer.model.predict = mocker.MagicMock(return_value=np.array([0, 1]))

#     # Call the tune_model method
#     study = model_optimizer.tune_model_in_parallel()

#     # Check that the returned study object is not None
#     assert study
#     assert isinstance(study, optuna_distributed.study.DistributedStudy)


# def test_create_pipeline():
#     """Tests that the create_pipeline method returns a pipeline with the expected steps."""

#     # Create mock preprocessor, selector, and model
#     preprocessor_step = ColumnTransformer(transformers=[])
#     selector_step = VarianceThreshold()
#     model = LogisticRegression()

#     pipeline = ModelOptimizer.create_pipeline(preprocessor_step, selector_step, model)

#     assert isinstance(pipeline, Pipeline)
#     assert len(pipeline.steps) == 3
#     assert pipeline.steps[0][0] == "preprocessor"
#     assert isinstance(pipeline.steps[0][1], ColumnTransformer)
#     assert pipeline.steps[1][0] == "selector"
#     assert isinstance(pipeline.steps[1][1], VarianceThreshold)
#     assert pipeline.steps[2][0] == "classifier"
#     assert isinstance(pipeline.steps[2][1], LogisticRegression)


# def test_fit_pipeline(mocker, model_optimizer):
#     """Tests that the fit_pipeline method returns a pipeline with the expected steps.
#     This test also mocks the fit method of the pipeline to track its calls, i.e.,
#     interaction testing.
#     """

#     # Set the required attributes
#     model_optimizer.train_class = np.random.randint(0, 2, size=10)
#     model_optimizer.n_features = 5

#     # Create mock train_features, preprocessor, selector, and model
#     train_features = pd.DataFrame(
#         {
#             "feat1": np.random.rand(10),
#             "feat2": np.random.rand(10),
#             "feat3": np.random.rand(10),
#             "feat4": np.random.rand(10),
#             "feat5": np.random.rand(10),
#         }
#     )
#     preprocessor_step = ColumnTransformer(
#         transformers=[
#             (
#                 "passthrough",
#                 FunctionTransformer(),
#                 ["feat1", "feat2", "feat3", "feat4", "feat5"],
#             )
#         ]
#     )
#     selector_step = VarianceThreshold(
#         threshold=0
#     )  # threshold=0 means no features will be removed
#     model = LogisticRegression()

#     # Create a spy on the fit_pipeline method to track its calls
#     spy = mocker.spy(model_optimizer, "fit_pipeline")

#     # Call the fit_pipeline method
#     pipeline = model_optimizer.fit_pipeline(
#         train_features, preprocessor_step, selector_step, model
#     )

#     # Check that the fit method of the pipeline was called with the correct arguments
#     called_args = spy.call_args

#     # Check that the returned pipeline is not None
#     assert isinstance(pipeline, Pipeline)

#     # Check if fit_pipeline was called with the expected arguments (interaction testing assertions)
#     assert isinstance(called_args[0][0], pd.DataFrame)
#     assert called_args[0][0].equals(train_features)
#     assert isinstance(called_args[0][1], ColumnTransformer)
#     assert isinstance(called_args[0][2], VarianceThreshold)
#     assert isinstance(called_args[0][3], LogisticRegression)

#     # Check if fit_pipeline fits the logistic regression model and returns 5 coefficients (5 features)
#     assert len(pipeline.named_steps["classifier"].coef_[0]) == train_features.shape[1]
