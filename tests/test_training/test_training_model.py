# """
# Test functions for the ModelOptimizer and ModelEvaluator classes in
# the model module src/training/utils/model.py.
# """

# import os

# import joblib
# import matplotlib.pyplot as plt
# import numpy as np
# import optuna
# import optuna_distributed
# import pandas as pd
# import pytest
# from comet_ml import API, Experiment
# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.compose import ColumnTransformer
# from sklearn.feature_selection import SelectKBest, VarianceThreshold, chi2
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import average_precision_score, roc_auc_score
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import (
#     FunctionTransformer,
#     LabelEncoder,
#     OneHotEncoder,
#     StandardScaler,
# )

# from src.training.utils.model import (
#     ModelChampionManager,
#     ModelEvaluator,
#     ModelOptimizer,
# )


# @pytest.fixture
# def model_optimizer(mocker):
#     """Creates a ModelOptimizer instance for testing. This fixture is used by all the tests
#     related to ModelOptimizer in this module.
#     """

#     # Create mock objects for the required attributes
#     train_exp = mocker.MagicMock()
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

#     _optimizer = ModelOptimizer(
#         train_exp,
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

#     return _optimizer


# def test_model_optimizer_init(model_optimizer):  # pylint: disable=redefined-outer-name
#     """Tests that the ModelOptimizer instance is created with the expected attributes.
#     This test also checks that the attributes are not None.
#     """

#     assert model_optimizer.train_exp is not None
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


# def test_model_optimizer_generate_trial_params(
#     mocker, model_optimizer  # pylint: disable=redefined-outer-name
# ):
#     """Tests that the generate_trial_params method returns a non-empty dictionary."""

#     # Create a mock optuna trial object
#     trial = mocker.create_autospec(optuna.trial.Trial)

#     params = model_optimizer.generate_trial_params(trial)

#     assert params
#     assert isinstance(params, dict)


# def test_model_optimizer_calc_perf_metrics(
#     model_optimizer,  # pylint: disable=redefined-outer-name
# ):
#     """Tests that the calc_perf_metrics method returns a non-empty dataframe.
#     It also checks that the returned dataframe has the expected metrics.
#     """

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


# def test_model_optimizer_objective_function(
#     mocker, model_optimizer  # pylint: disable=redefined-outer-name
# ):
#     """Tests that the objective_function method returns a float value."""

#     # Create a mock optuna trial object
#     trial = mocker.create_autospec(optuna.trial.Trial)
#     trial.suggest_float.return_value = 0.5

#     # Mock the generate_trial_params method
#     model_optimizer.generate_trial_params = mocker.MagicMock()
#     model_optimizer.generate_trial_params.return_value = {
#         "C": 1,
#         "l1_ratio": 2,
#     }

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

#     valid_score = float(model_optimizer.objective_function(trial))

#     assert isinstance(valid_score, float)
#     assert (
#         valid_score >= 0.0 and valid_score <= 1.0
#     ), "The valid_score is not between 0 and 1"


# def test_model_optimizer_tune_model(
#     mocker, model_optimizer  # pylint: disable=redefined-outer-name
# ):
#     """Tests that the tune_model method returns a non-empty optuna study object. The test
#     mocks the predict method of the model to return predictions of the same length as the
#     true labels. It mocks the tune_model_in_parallel method to return the mock study and
#     avoid running the actual optimization process, which is tested in another test.
#     """

#     # Mock required attributes for tune_model method
#     model_optimizer.train_class = np.array([0, 1])
#     model_optimizer.valid_class = np.array([1, 0])

#     # Mock the predict method of the model to return predictions of the same length as the true labels
#     model_optimizer.model.predict = mocker.MagicMock(return_value=np.array([0, 1]))

#     # Mock tune_model to return the mock study
#     mock_study = mocker.MagicMock(spec=optuna.study.Study)
#     model_optimizer.tune_model = mocker.MagicMock(return_value=mock_study)

#     study = model_optimizer.tune_model()

#     assert study is not None
#     assert isinstance(study, optuna.study.Study)


# def test_model_optimizer_tune_model_in_parallel(
#     mocker, model_optimizer  # pylint: disable=redefined-outer-name
# ):
#     """Tests that the tune_model method returns a non-empty optuna study object. The test
#     mocks the predict method of the model to return predictions of the same length as the
#     true labels. It mocks the tune_model_in_parallel method to return the mock study and
#     avoid running the actual optimization process, which is tested in another test.
#     """

#     # Mock required attributes for tune_model_in_parallel method
#     model_optimizer.train_class = np.array([0, 1])
#     model_optimizer.valid_class = np.array([1, 0])

#     # Mock the predict method of the model to return predictions of the same length as the true labels
#     model_optimizer.model.predict = mocker.MagicMock(return_value=np.array([0, 1]))

#     # Mock tune_model_in_parallel to return the mock study
#     mock_study = mocker.MagicMock(spec=optuna_distributed.study.DistributedStudy)
#     model_optimizer.tune_model_in_parallel = mocker.MagicMock(return_value=mock_study)

#     # Call the tune_model method
#     study = model_optimizer.tune_model_in_parallel()

#     # Check that the returned study object is not None
#     assert study is not None
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


# def test_fit_pipeline(mocker, model_optimizer):  # pylint: disable=redefined-outer-name
#     """Tests that the fit_pipeline method returns a pipeline with the expected steps.
#     The test also mocks the fit method of the pipeline to track its calls, i.e.,
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


# @pytest.fixture
# def model_evaluator():
#     """Creates a ModelEvaluator instance for testing. This fixture is used by all the tests
#     related to ModelEvaluator in this module.
#     """

#     train_exp = Experiment(
#         api_key="dummy_key", project_name="general", workspace="test"
#     )
#     pipeline = Pipeline(
#         steps=[
#             (
#                 "preprocessor",
#                 ColumnTransformer(
#                     transformers=[
#                         ("num", "passthrough", ["num_feature"]),
#                         ("cat", OneHotEncoder(), ["cat_feature"]),
#                     ]
#                 ),
#             ),
#             ("selector", SelectKBest(chi2, k=2)),
#             ("classifier", LogisticRegression()),
#         ]
#     )
#     train_features = pd.DataFrame(
#         {"num_feature": [1, 2, 3], "cat_feature": ["a", "b", "a"]}
#     )
#     train_class = np.array([0, 1, 0])
#     valid_features = pd.DataFrame(
#         {"num_feature": [2, 3, 1], "cat_feature": ["b", "a", "b"]}
#     )
#     valid_class = np.array([1, 0, 1])
#     return ModelEvaluator(
#         train_exp, pipeline, train_features, train_class, valid_features, valid_class
#     )


# def test_plot_feature_importance(model_evaluator):
#     """Tests that the plot_feature_importance method returns a matplotlib figure object."""

#     feature_importance_scores = np.array([0.1, 0.2])
#     feature_names = ["num_feature", "cat_feature"]
#     figure_obj = plt.figure()
#     fig = model_evaluator.plot_feature_importance(
#         feature_importance_scores,
#         feature_names,
#         figure_obj,
#         n_top_features=len(feature_importance_scores),
#     )
#     assert isinstance(fig, plt.Figure)

#     # Check contents of the figure
#     ax = fig.axes[0]
#     assert ax.get_title() == f"Top {len(feature_importance_scores)} important features"
#     assert ax.get_xlabel() == "Contribution"
#     assert ax.get_ylabel() == "Feature Name"

#     # Check if the function's behavior with different inputs
#     with pytest.raises(IndexError):
#         model_evaluator.plot_feature_importance(np.array([]), [], figure_obj)
#     with pytest.raises(TypeError):
#         model_evaluator.plot_feature_importance(
#             np.array([0.1, 0.2]), [], n_top_features=3
#         )
#     with pytest.raises(TypeError):
#         model_evaluator.plot_feature_importance(
#             np.array([0.1, 0.2]), ["num_feature", "cat_feature"], fig_size=(0, 0)
#         )


# def test_extract_feature_importance(model_evaluator):
#     """Tests that the extract_feature_importance method returns feature
#     importance scores for the specified features. This test also checks
#     that the returned feature importance scores are of the expected length.
#     """

#     num_feature_names = ["num_feature"]
#     cat_feature_names = ["cat_feature"]

#     model_evaluator.pipeline.fit(
#         model_evaluator.train_features, model_evaluator.train_class
#     )
#     model_evaluator.extract_feature_importance(
#         model_evaluator.pipeline, num_feature_names, cat_feature_names
#     )

#     # Check if two coefficients (two features) are extracted
#     assert len(model_evaluator.pipeline.named_steps["classifier"].coef_[0]) == len(
#         num_feature_names + cat_feature_names
#     )


# def test_log_feature_importance_fig(mocker, model_evaluator):
#     """Tests that the _log_feature_importance_fig method logs the feature importance
#     figure if the classifier name is not 'VotingClassifier'
#     """

#     model_evaluator.train_exp = mocker.MagicMock()

#     # Check if the feature importance fig is not logged if classifier name is 'VotingClassifier'
#     model_evaluator._log_feature_importance_fig(
#         "VotingClassifier", np.array([0.1, 0.2]), ["num_feature", "cat_feature"]
#     )
#     model_evaluator.train_exp.log_figure.assert_not_called()

#     # Check if the feature importance fig is logged
#     model_evaluator._log_feature_importance_fig(
#         "LogisticRegression",
#         np.array([0.1, 0.2]),
#         ["num_feature", "cat_feature"],
#         2,
#         (24, 36),
#         10.0,
#         "Feature Importance",
#     )
#     model_evaluator.train_exp.log_figure.assert_called_once()

#     model_evaluator._log_feature_importance_fig(
#         "LightGBM",
#         np.array([0.1, 0.2]),
#         ["num_feature", "cat_feature"],
#         2,
#         (24, 36),
#         10.0,
#         "Feature Importance",
#     )
#     model_evaluator.train_exp.log_figure.call_count == 2


# def test_plot_roc_curve():
#     """Tests that the plot_roc_curve method returns a matplotlib figure object. The test
#     checks that the AUC is correctly calculated and displayed in the plot.
#     """

#     y_true = np.array([0, 1, 0, 1, 0])
#     y_pred = np.array([0.1, 0.9, 0.2, 0.2, 0.1])
#     expected_auc = roc_auc_score(y_true, y_pred)

#     fig = ModelEvaluator.plot_roc_curve(y_true, y_pred, (6, 6))
#     ax = fig.axes[0]
#     lines = ax.get_lines()

#     assert isinstance(fig, plt.Figure)

#     # There should be two lines: the ROC curve and the diagonal
#     assert len(lines) == 2

#     # The AUC should be correctly calculated
#     assert f"ROC curve (AUC = {expected_auc:.2f})" in lines[0].get_label()


# def test_plot_precision_recall_curve():
#     """Tests that the plot_precision_recall_curve method returns a matplotlib figure object.
#     The test checks that the average precision is correctly calculated and displayed in the plot.
#     """

#     y_true = np.array([0, 1, 0, 1, 0])
#     y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.1])

#     fig = ModelEvaluator.plot_precision_recall_curve(y_true, y_pred)
#     ax = fig.axes[0]
#     lines = ax.get_lines()
#     expected_ap = average_precision_score(y_true, y_pred)

#     assert isinstance(fig, plt.Figure)

#     assert ax.get_title() == "Precision-Recall Curve"
#     assert ax.get_xlabel() == "Recall"
#     assert ax.get_ylabel() == "Precision"

#     # There should be one line: the precision-recall curve
#     assert len(lines) == 1

#     # The AP should be correctly calculated
#     assert f"Precision-Recall curve (AP = {expected_ap:.2f})" in lines[0].get_label()


# def test_plot_cumulative_gains():
#     """Tests that the plot_cumulative_gains method returns a matplotlib figure object. The test
#     checks that the plot has the expected title, x-axis label, and y-axis label. The plot should
#     also contain four lines: the precision-recall curve, the model, the wizard, and the random line.
#     """

#     y_true = np.array([0, 1, 0, 1, 0])
#     y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.1])

#     fig = ModelEvaluator.plot_cumulative_gains(y_true, y_pred)
#     ax = fig.axes[0]
#     lines = ax.get_lines()

#     assert isinstance(fig, plt.Figure)

#     # Check the contents of the plot
#     assert ax.get_title() == "Cumulative Gain Plot"
#     assert ax.get_xlabel() == "Deciles"
#     assert ax.get_ylabel() == "% Resonders"

#     # There should be four lines: 'Precision-Recall curve (AP = 1.00)', 'Model', 'Wizard', 'Random'
#     assert len(lines) == 4
#     assert "Precision-Recall curve (AP = 1.00)" in [line.get_label() for line in lines]
#     assert "Model" in [line.get_label() for line in lines]
#     assert "Wizard" in [line.get_label() for line in lines]
#     assert "Random" in [line.get_label() for line in lines]


# def test_plot_lift_curve():
#     """Tests that the plot_lift_curve method returns a matplotlib figure object. The test checks
#     that the plot has the expected title, x-axis label, and y-axis label. The plot should also
#     contain six lines: the precision-recall curve, the model, the wizard, the random line, the
#     model, and the random line.
#     """

#     y_true = np.array([0, 1, 0, 1, 0])
#     y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.1])

#     fig = ModelEvaluator.plot_lift_curve(y_true, y_pred)
#     ax = fig.axes[0]
#     lines = ax.get_lines()

#     assert isinstance(fig, plt.Figure)

#     # Check the contents of the plot
#     assert ax.get_title() == "Lift Plot"
#     assert ax.get_xlabel() == "Deciles"
#     assert ax.get_ylabel() == "Lift"

#     # There should be six lines: Precision-Recall curve (AP = 1.00)', 'Model', 'Wizard', 'Random', 'Model', 'Random'
#     assert len(lines) == 6
#     assert "Precision-Recall curve (AP = 1.00)" in [line.get_label() for line in lines]
#     assert "Model" in [line.get_label() for line in lines]
#     assert "Wizard" in [line.get_label() for line in lines]
#     assert "Random" in [line.get_label() for line in lines]


# def test_convert_metrics_from_df_to_dict():
#     """Tests that the convert_metrics_from_df_to_dict method returns a dictionary with
#     the expected keys and values. The test also checks that the returned dictionary has
#     the expected keys and values when a prefix is provided.
#     """

#     scores = pd.DataFrame(
#         {"Metric": ["accuracy", "precision", "recall"], "Score": [0.9, 0.8, 0.7]}
#     )

#     # Call the function without a prefix
#     metrics_values = ModelEvaluator.convert_metrics_from_df_to_dict(scores)
#     assert metrics_values == {"accuracy": 0.9, "precision": 0.8, "recall": 0.7}

#     # Call the function with a prefix
#     metrics_values = ModelEvaluator.convert_metrics_from_df_to_dict(
#         scores, prefix="test_"
#     )
#     assert metrics_values == {
#         "test_accuracy": 0.9,
#         "test_precision": 0.8,
#         "test_recall": 0.7,
#     }


# def test_evaluate_model_perf(mocker, model_evaluator):
#     """Tests that the evaluate_model_perf method returns the expected performance metrics dataframes.
#     The test mocks the methods that evaluate_model_perf calls internally. The pipeline is also mocked
#     because it must be fitted before calling evaluate_model_perf. The LabelEncoder is also mocked because
#     it's required to call _log_confusion_matrix. The function calc_perf_metrics is also mocked to return
#     a dataframe with the expected metrics. To ensure the test is isolated, functions called by
#     evaluate_model_perf are mocked to return the expected results and not tested in this test.
#     """

#     # Create a mock pipeline with a predict_proba method
#     mock_pipeline = mocker.MagicMock(Pipeline)
#     mock_pipeline.predict_proba.return_value = np.array(
#         [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]]
#     )
#     mock_pipeline.fit.return_value = None

#     # Set pipeline to mock pipeline because it must be fitted before calling evaluate_model_perf
#     model_evaluator.pipeline = mock_pipeline

#     # Create a mock LabelEncoder as it's required to call _log_confusion_matrix
#     mock_encoder = mocker.MagicMock(LabelEncoder)
#     mock_encoder.inverse_transform.return_value = ["class1", "class2"]

#     # Mock the methods that evaluate_model_perf calls internally to return the expected results
#     mocker.patch.object(
#         model_evaluator, "_get_pred_class", return_value=np.array([0, 1])
#     )
#     mocker.patch.object(
#         model_evaluator,
#         "calc_perf_metrics",
#         return_value=pd.DataFrame({"accuracy": [0.5]}),
#     )

#     # Call the function with the mock objects
#     train_scores, valid_scores = model_evaluator.evaluate_model_perf(
#         class_encoder=mock_encoder
#     )

#     # Assert that the function returns the expected results
#     assert train_scores.equals(pd.DataFrame({"accuracy": [0.5]}))
#     assert valid_scores.equals(pd.DataFrame({"accuracy": [0.5]}))


# def test_get_pred_class(model_evaluator):
#     """Tests that the _get_pred_class method returns the expected class labels."""

#     # Define test inputs and expected output
#     model_evaluator.encoded_pos_class_label = (
#         1  # assuming positive class is encoded as 1
#     )
#     pred_probs = np.array([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])
#     threshold = 0.5
#     expected_output = np.array([1, 0, 1])

#     pred_class = model_evaluator._get_pred_class(pred_probs, threshold)

#     assert np.array_equal(
#         pred_class, expected_output
#     ), "The method did not return the expected class labels"

#     # Test the method with different threshold
#     with pytest.raises(AssertionError):
#         new_pred_class = model_evaluator._get_pred_class(pred_probs, threshold=0.15)
#         assert np.array_equal(new_pred_class, expected_output)


# def test_get_original_class_labels(mocker, model_evaluator):
#     """Tests that the _get_original_class_labels method returns the expected original class
#     labels and class labels. The test mocks the pipeline attribute of the ModelEvaluator
#     instance because it's required to call the method.
#     """

#     # Create a mock pipeline with a classes_ attribute
#     mock_pipeline = mocker.MagicMock(Pipeline)
#     mock_pipeline.classes_ = np.array([0, 1])
#     model_evaluator.pipeline = mock_pipeline
#     class_encoder = LabelEncoder()
#     class_encoder.fit(["class0", "class1"])

#     # Define test inputs
#     true_class = np.array([0, 1, 1])
#     pred_class = np.array([1, 0, 1])

#     (
#         true_original_class,
#         pred_original_class,
#         original_class_labels,
#     ) = model_evaluator._get_original_class_labels(
#         true_class, pred_class, class_encoder
#     )

#     # Define the expected output
#     expected_true_original_class = np.array(["class0", "class1", "class1"])
#     expected_pred_original_class = np.array(["class1", "class0", "class1"])
#     expected_original_class_labels = ["class0", "class1"]

#     # Assert that the method output matches the expected output
#     assert np.array_equal(
#         true_original_class, expected_true_original_class
#     ), "The method did not return the expected true original class labels"
#     assert np.array_equal(
#         pred_original_class, expected_pred_original_class
#     ), "The method did not return the expected predicted original class labels"
#     assert (
#         original_class_labels == expected_original_class_labels
#     ), "The method did not return the expected original class labels"


# def test_log_confusion_matrix(mocker, model_evaluator):
#     """Tests that the _log_confusion_matrix method logs the confusion matrix. The test
#     checks that the method logs the confusion matrix the correct number of times.
#     """

#     # Create a mock train_exp to test number of calls by the method
#     mock_train_exp = mocker.MagicMock(Experiment)
#     model_evaluator.train_exp = mock_train_exp

#     # Define test inputs
#     original_train_class = np.array(["class0", "class1", "class1"])
#     pred_original_train_class = np.array(["class1", "class0", "class1"])
#     original_valid_class = np.array(["class0", "class1", "class1"])
#     pred_original_valid_class = np.array(["class1", "class0", "class1"])
#     original_class_labels = ["class0", "class1"]

#     model_evaluator._log_confusion_matrix(
#         original_train_class,
#         pred_original_train_class,
#         original_valid_class,
#         pred_original_valid_class,
#         original_class_labels,
#     )

#     # Assert that the log_confusion_matrix method was called four times
#     # Note log_confusion_matrix logs normalized and non-normalized confusion matrices
#     # for both the training and validation sets
#     assert (
#         mock_train_exp.log_confusion_matrix.call_count == 4
#     ), f"""The log_confusion_matrix method was not called the correct number of times
#     ({mock_train_exp.log_confusion_matrix.call_count} calls)."""


# def test_log_calibration_curve(mocker, model_evaluator):
#     """Tests that the _log_calibration_curve method logs the calibration curve. The test
#     checks that the method logs the calibration curve the correct number of times.
#     """

#     # Create a mock train_exp to test number of calls by the method
#     mock_train_exp = mocker.MagicMock(Experiment)
#     model_evaluator.train_exp = mock_train_exp

#     # Define test inputs
#     model_evaluator.valid_class = np.array([0, 1, 1])
#     model_evaluator.encoded_pos_class_label = (
#         1  # assuming positive class is encoded as 1
#     )
#     pred_probs = np.array([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])

#     model_evaluator._log_calibration_curve(pred_probs)

#     # Assert that the log_figure method was called once
#     assert (
#         mock_train_exp.log_figure.call_count == 1
#     ), f"""The log_figure method was not called the correct number of times
#     ({mock_train_exp.log_figure.call_count} calls)"""


# def test_log_roc_curve(mocker, model_evaluator):
#     """Tests that the _log_roc_curve method logs the ROC curve. The test checks
#     that the method logs the ROC curve the correct number of times.
#     """

#     # Create a mock train_exp to test number of calls by the method
#     mock_train_exp = mocker.MagicMock(Experiment)
#     model_evaluator.train_exp = mock_train_exp

#     # Define test inputs
#     model_evaluator.valid_class = np.array([0, 1, 1])
#     encoded_pos_class_label = 1  # assuming positive class is encoded as 1
#     pred_probs = np.array([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])

#     # Mock the plot_roc_curve method to return a MagicMock (which will stand in for the figure)
#     mock_figure = mocker.MagicMock(plt.Figure)
#     mocker.patch.object(ModelEvaluator, "plot_roc_curve", return_value=mock_figure)
#     model_evaluator._log_roc_curve(pred_probs, encoded_pos_class_label)

#     # Assert that the log_figure method was called once with the correct arguments
#     mock_train_exp.log_figure.assert_called_once_with(
#         figure_name="ROC Curve", figure=mock_figure, overwrite=True
#     )


# def test_log_precision_recall_curve(monkeypatch, mocker, model_evaluator):
#     """Tests that the _log_precision_recall_curve method logs the precision-recall curve. The
#     test checks that the method logs the precision-recall curve the correct number of times.
#     """

#     # Create a mock train_exp to test number of calls by the method
#     mock_train_exp = mocker.MagicMock(Experiment)
#     model_evaluator.train_exp = mock_train_exp

#     # Define test inputs
#     model_evaluator.valid_class = np.array([0, 1, 1])
#     encoded_pos_class_label = 1  # assuming positive class is encoded as 1
#     pred_probs = np.array([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])

#     # Mock the plot_precision_recall_curve method to return a MagicMock (which will stand in for the figure)
#     mock_figure = mocker.MagicMock(plt.Figure)
#     monkeypatch.setattr(
#         model_evaluator,
#         "plot_precision_recall_curve",
#         lambda *args, **kwargs: mock_figure,
#     )

#     # Call the method with the test inputs
#     model_evaluator._log_precision_recall_curve(pred_probs, encoded_pos_class_label)

#     # Assert that the log_figure method was called once with the correct arguments
#     mock_train_exp.log_figure.assert_called_once_with(
#         figure_name="Precision-Recall Curve", figure=mock_figure, overwrite=True
#     )


# def test_log_cumulative_gains(mocker, model_evaluator):
#     """Tests that the _log_cumulative_gains method logs the cumulative gains plot. The test
#     checks that the method logs the cumulative gains plot the correct number of times.
#     """

#     # Create a mock train_exp to test number of calls by the method
#     mock_train_exp = mocker.MagicMock(Experiment)
#     model_evaluator.train_exp = mock_train_exp

#     # Define test inputs
#     valid_class = np.array([0, 1, 1])
#     pred_probs = np.array([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])
#     model_evaluator.encoded_pos_class_label = (
#         1  # Assuming positive class is encoded as 1
#     )

#     # Mock the plot_cumulative_gains method to return a MagicMock (which will stand in for the figure)
#     mock_figure = mocker.MagicMock(plt.Figure)
#     mocker.patch.object(
#         model_evaluator, "plot_cumulative_gains", return_value=mock_figure
#     )

#     # Call the method with the test inputs
#     model_evaluator._log_cumulative_gains(pred_probs, valid_class)

#     # Assert that the log_figure method was called once with the correct arguments
#     mock_train_exp.log_figure.assert_called_once_with(
#         figure_name="Cumulative Gain", figure=mock_figure, overwrite=True
#     )


# def test_log_lift_curve(mocker, model_evaluator):
#     """Tests that the _log_lift_curve method logs the lift curve. The test checks that
#     the method logs the lift curve the correct number of times.
#     """

#     # Create a mock train_exp to test number of calls by the method
#     mock_train_exp = mocker.MagicMock(Experiment)
#     model_evaluator.train_exp = mock_train_exp

#     # Define test inputs
#     valid_class = np.array([0, 1, 1])
#     pred_probs = np.array([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])
#     model_evaluator.encoded_pos_class_label = (
#         1  # Assuming positive class is encoded as 1
#     )

#     # Mock the plot_lift_curve method to return a MagicMock (which will stand in for the figure)
#     mock_figure = mocker.MagicMock(plt.Figure)
#     mocker.patch.object(model_evaluator, "plot_lift_curve", return_value=mock_figure)
#     model_evaluator._log_lift_curve(pred_probs, valid_class)

#     # Assert that the log_figure method was called once with the correct arguments
#     mock_train_exp.log_figure.assert_called_once_with(
#         figure_name="Lift Curve", figure=mock_figure, overwrite=True
#     )


# def test_calc_expected_calibration_error(model_evaluator):
#     """Tests that the calc_expected_calibration_error method returns the expected ECE for a perfectly
#     calibrated model and a highly uncalibrated model. The test checks that the method raises an
#     AssertionError if the decision threshold is not between 0 and 1.
#     """

#     # 900 samples where the model is 90% confident and is correct
#     pred_probs_90_correct = np.full(900, 0.9)
#     true_labels_90_correct = np.full(900, 1)

#     # 100 samples where the model is 90% confident and is incorrect
#     pred_probs_90_incorrect = np.full(100, 0.9)
#     true_labels_90_incorrect = np.full(100, 0)

#     # 90 samples where the model is 10% confident and is correct
#     pred_probs_10_correct = np.full(90, 0.1)
#     true_labels_10_correct = np.full(90, 0)

#     # 10 samples where the model is 10% confident and is incorrect
#     pred_probs_10_incorrect = np.full(10, 0.1)
#     true_labels_10_incorrect = np.full(10, 1)

#     # Calculate ECE with test inputs for a perfectly calibrated model
#     near_perfect_ece = model_evaluator.calc_expected_calibration_error(
#         pred_probs=np.concatenate(
#             [
#                 pred_probs_90_correct,
#                 pred_probs_90_incorrect,
#                 pred_probs_10_correct,
#                 pred_probs_10_incorrect,
#             ]
#         ),
#         true_labels=np.concatenate(
#             [
#                 true_labels_90_correct,
#                 true_labels_90_incorrect,
#                 true_labels_10_correct,
#                 true_labels_10_incorrect,
#             ]
#         ),
#         decision_thresh_val=0.5,
#         nbins=5,
#     )

#     # Assert that the ECE is 0 for a perfectly calibrated model
#     assert isinstance(near_perfect_ece, float)
#     assert np.isclose(0, near_perfect_ece, atol=0.1)  # ECE should be close to 0

#     # 50 samples where the model is 95% confident and is correct
#     pred_probs_95_correct = np.full(50, 0.95)
#     true_labels_95_correct = np.full(50, 1)

#     # 950 samples where the model is 95% confident and is incorrect
#     pred_probs_95_incorrect = np.full(950, 0.95)
#     true_labels_95_incorrect = np.full(950, 0)

#     # 5 samples where the model is 5% confident and is correct
#     pred_probs_5_correct = np.full(5, 0.05)
#     true_labels_5_correct = np.full(5, 0)

#     # 95 samples where the model is 5% confident and is incorrect
#     pred_probs_5_incorrect = np.full(95, 0.05)
#     true_labels_5_incorrect = np.full(95, 1)

#     # Calculate ECE with test inputs for a uncalibrated model
#     high_ece = model_evaluator.calc_expected_calibration_error(
#         pred_probs=np.concatenate(
#             [
#                 pred_probs_95_correct,
#                 pred_probs_95_incorrect,
#                 pred_probs_5_correct,
#                 pred_probs_5_incorrect,
#             ]
#         ),
#         true_labels=np.concatenate(
#             [
#                 true_labels_95_correct,
#                 true_labels_95_incorrect,
#                 true_labels_5_correct,
#                 true_labels_5_incorrect,
#             ]
#         ),
#         decision_thresh_val=0.5,
#         nbins=5,
#     )

#     # Assert that the ECE is 0 for a highly uncalibrated model
#     assert isinstance(high_ece, float)
#     assert np.isclose(1, high_ece, atol=0.2)  # ECE should be close to 1


# def test_select_best_performer(mocker):
#     """Tests that the select_best_performer method returns the expected best model
#     name. The test checks that the method calls the Comet API to get the experiment
#     metrics and that it selects the best model based on the comparison metric. The
#     test also checks that the method raises an AssertionError if the wrong model is
#     selected.
#     """

#     # Create a ModelChampionManager instance that includes the select_best_performer method
#     manager = ModelChampionManager(champ_model_name="champion_model")

#     # Set up the test data
#     project_name = "project"
#     workspace_name = "workspace"
#     comparison_metric = "accuracy"
#     comet_exp_keys = pd.DataFrame(
#         [
#             {"0": "model1", "1": "mock_exp1"},
#             {"0": "model2", "1": "mock_exp2"},
#             {"0": "model3", "1": "mock_exp3"},
#         ]
#     )

#     mock_api = mocker.Mock(API)
#     mock_exp1 = mocker.MagicMock()
#     mock_exp2 = mocker.MagicMock()
#     mock_exp3 = mocker.MagicMock()

#     # Set up the mock experiment metrics, where exp2 has the highest metric value
#     mock_exp1.get_metrics.return_value = [
#         {"metricName": comparison_metric, "metricValue": 0.3}
#     ]
#     mock_exp2.get_metrics.return_value = [
#         {"metricName": comparison_metric, "metricValue": 0.9}
#     ]
#     mock_exp3.get_metrics.return_value = [
#         {"metricName": comparison_metric, "metricValue": 0.7}
#     ]

#     # Set the side effect of `get_experiment` method on the `mock_api` object to return different
#     # mock experiment objects for each call, each with different model name and metric value
#     mock_api.get_experiment.side_effect = [mock_exp1, mock_exp2, mock_exp3]

#     best_model_name = manager.select_best_performer(
#         project_name=project_name,
#         workspace_name=workspace_name,
#         comparison_metric=comparison_metric,
#         comet_exp_keys=comet_exp_keys,
#         comet_api=mock_api,
#     )

#     # Assert that get_experiment was called the correct number of times
#     assert mock_api.get_experiment.call_count == comet_exp_keys.shape[0]

#     # Assert that the best model was selected correctly, i.e., the model with highest score
#     assert best_model_name == "model2"

#     # Test that the function raises a AssertionError if the comparison metric is not found
#     with pytest.raises(AssertionError):
#         assert best_model_name == "model1" or best_model_name == "model3"


# def test_calibrate_pipeline():
#     """Tests that the calibrate_pipeline method returns a calibrated pipeline. Thie test
#     checks that the method calls the fit method of the pipeline and the calibration
#     display object.
#     """

#     # Create a ModelChampionManager instance that includes the calibrate_pipeline method
#     mock_self = ModelChampionManager(champ_model_name="champion_model")

#     # Set up the test data
#     valid_features = pd.DataFrame(
#         np.random.randint(0, 100, size=(100, 4)), columns=list("ABCD")
#     )
#     valid_class = np.random.randint(2, size=100)
#     preprocessor_step = ColumnTransformer(
#         transformers=[("scaler", StandardScaler(), ["A", "B", "C", "D"])]
#     )
#     selector_step = VarianceThreshold()
#     model = LogisticRegression()
#     cv_folds = 5

#     # Create a new pipeline with the calibrated classifier
#     unfitted_pipeline = Pipeline(
#         steps=[
#             ("preprocessor", preprocessor_step),
#             ("selector", selector_step),
#             ("classifier", model),
#         ]
#     )

#     # Test that the function raises a ValueError if the supplied pipeline is not fitted yet
#     with pytest.raises(
#         ValueError, match="The classifier in the fitted pipeline is not fitted."
#     ):
#         _ = mock_self.calibrate_pipeline(
#             valid_features=valid_features,
#             valid_class=valid_class,
#             fitted_pipeline=unfitted_pipeline,
#             cv_folds=cv_folds,
#         )

#     fitted_pipeline = Pipeline(
#         steps=[
#             ("preprocessor", preprocessor_step),
#             ("selector", selector_step),
#             ("classifier", model),
#         ]
#     )
#     fitted_pipeline = fitted_pipeline.fit(valid_features, valid_class)

#     calib_pipeline = mock_self.calibrate_pipeline(
#         valid_features=valid_features,
#         valid_class=valid_class,
#         fitted_pipeline=fitted_pipeline,
#         cv_folds=cv_folds,
#     )

#     assert isinstance(calib_pipeline, Pipeline)
#     assert isinstance(calib_pipeline.named_steps["classifier"], CalibratedClassifierCV)
#     assert type(model).__name__ in str(calib_pipeline.named_steps["classifier"])


# def test_log_and_register_champ_model(mocker, monkeypatch):
#     """Tests that the log_and_register_champ_model method logs and registers the champion model.
#     The test checks that the method calls the log_model and register_model methods of the experiment
#     object and the end method of the experiment object.
#     """

#     # Create a ModelChampionManager instance that includes the log_and_register_champ_model method
#     instance = ModelChampionManager(champ_model_name="champion_model")

#     # Set up the test data
#     local_path = "/path/to/model"
#     pipeline = mocker.MagicMock(Pipeline)
#     exp_obj = mocker.MagicMock(Experiment)

#     # Mock os.makedirs and joblib.dump
#     monkeypatch.setattr(os, "makedirs", lambda x: None)
#     monkeypatch.setattr(joblib, "dump", lambda x, y: None)
#     monkeypatch.setattr(os.path, "exists", lambda x: False)

#     instance.log_and_register_champ_model(local_path, pipeline, exp_obj)

#     exp_obj.log_model.assert_called_once_with(
#         name=instance.champ_model_name,
#         file_or_folder=f"{local_path}/{instance.champ_model_name}.pkl",
#         overwrite=False,
#     )
#     exp_obj.register_model.assert_called_once_with(model_name=instance.champ_model_name)
#     exp_obj.end.assert_called_once()
