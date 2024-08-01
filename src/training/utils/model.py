"""
Includes functions for model optimization and evaluation.
"""

from datetime import datetime
from typing import Callable, Literal, Optional, Union

import joblib
import kds
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import optuna_distributed
import pandas as pd
from azureml.core import Model, Workspace
from dask.distributed import Client
from matplotlib.figure import Figure
from mlflow.models import infer_signature
from numpy.typing import ArrayLike
from sklearn.calibration import CalibrationDisplay, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.utils.logger import create_console_logger

##########################################################

# Get console logger
logger = create_console_logger("model_logger")


class ModelOptimizer:
    """A class to optimize model hyperparameters. It requires supplying preprocessed
    versions of the train and validation features to avoid fitting the whole pipeline
    in each objective function call during hyperparameters optimization.

    Attributes:
        train_features_preprocessed (pd.DataFrame): preprocessed train features.
        train_class (np.ndarray): train class labels.
        valid_features_preprocessed (pd.DataFrame): preprocessed validation features.
        valid_class (np.ndarray): validation class labels.
        n_features (int): number of features in the data.
        model (Callable): model object.
        fbeta_score_beta (float): beta value for fbeta score.
        encoded_pos_class_label (int): encoded positive class label.
        is_voting_ensemble (bool): whether the model is a voting ensemble or not.
        classifier_name (str): name of the classifier.
    """

    # List of supported models in this class
    # Note: this private variable shouldn't be mutated outside the class. It
    # should be updated when a new model is added to the class, which requires
    # adding its search space definition to generate_trial_params method.
    _supported_models = (
        "LogisticRegression",
        "RandomForestClassifier",
        "LGBMClassifier",
        "XGBClassifier",
        "CalibratedClassifierCV",
    )

    def __init__(
        self,
        train_features_preprocessed: pd.DataFrame,
        train_class: np.ndarray,
        valid_features_preprocessed: pd.DataFrame,
        valid_class: np.ndarray,
        n_features: int,
        model: Callable,
        search_space_params: dict,
        encoded_pos_class_label: int,
        fbeta_score_beta: float = 1.0,
        is_voting_ensemble: bool = False,
    ) -> None:
        """Creates a ModelOptimizer instance.

        Raises:
            AssertionError: if the specified model name is not supported.
        """
        self.train_features_preprocessed = train_features_preprocessed
        self.train_class = train_class
        self.valid_features_preprocessed = valid_features_preprocessed
        self.valid_class = valid_class
        self.n_features = n_features
        self.model = model
        self.search_space_params = search_space_params
        self.encoded_pos_class_label = encoded_pos_class_label
        self.fbeta_score_beta = fbeta_score_beta
        self.is_voting_ensemble = is_voting_ensemble
        self.classifier_name = self.model.__class__.__name__

        if not self.is_voting_ensemble:
            assert (
                self.classifier_name in self._supported_models
            ), f"Supported models are: {self._supported_models}. Got {self.classifier_name}!"

    def generate_trial_params(self, trial: optuna.trial.Trial) -> dict:
        """Samples model parameters values from search space as specified in
        config file.

        Args:
            trial (optuna.trial.Trial): an optuna trial object.

        Returns:
            params (dict): a dictionary of model parameters and their values.
        """

        params = {}
        for param, values in self.search_space_params.items():
            if isinstance(values[0], list):
                params[param] = trial.suggest_categorical(param, values[0])
            elif isinstance(values[0], int):
                params[param] = trial.suggest_int(param, int(values[0]), int(values[1]))
            else:
                params[param] = trial.suggest_float(
                    param, float(values[0]), float(values[1]), log=bool(values[2])
                )

        return params

    def calc_perf_metrics(
        self,
        true_class: ArrayLike,
        pred_class: ArrayLike,
    ) -> pd.DataFrame:
        """Calculates different performance metrics for binary classification models.

        Args:
            true_class (ArrayLike): true class label.
            pred_class (ArrayLike): predicted class label not probability.

        Returns:
            performance_metrics (pd.DataFrame): a dataframe with metric name and score columns.
        """

        cal_metrics = [
            ("accuracy", accuracy_score(true_class, pred_class)),
            (
                "precision",
                precision_score(
                    true_class,
                    pred_class,
                ),
            ),
            ("recall", recall_score(true_class, pred_class)),
            ("f1", f1_score(true_class, pred_class)),
            (
                f"f_{self.fbeta_score_beta}_score",
                fbeta_score(
                    true_class,
                    pred_class,
                    beta=self.fbeta_score_beta,
                ),
            ),
            (
                "roc_auc",
                roc_auc_score(true_class, pred_class, average=None),
            ),
        ]

        performance_metrics = pd.DataFrame(cal_metrics, columns=["Metric", "Score"])

        return performance_metrics

    def objective_function(
        self,
        trial: optuna.trial.Trial,
    ) -> float:
        """Objective function that evaluates the provided hyperparameters for a
        specified model, where the search metric being optimized is fbeta score.
        A trial hyperparameters are sampled from the search space using generate_trial_params
        method and then the model is fitted on training set and evaluated on the
        validation set.

        Args:
            trial (optuna.trial.Trial): an optuna trial object.

        Returns:
            valid_score (float): validation score.
        """

        # Define parameters search space
        params = self.generate_trial_params(trial=trial)

        # Fit model and calculate training score
        self.model.set_params(**params)
        self.model.fit(self.train_features_preprocessed, self.train_class)

        # Evaluate model on training and validation set
        # Note: default threshold of 0.5 is used for positive class but
        # other htreshold values can be used, which is problem-dependent.
        pred_train_preds = self.model.predict(self.train_features_preprocessed)
        pred_valid_preds = self.model.predict(self.valid_features_preprocessed)
        train_scores = self.calc_perf_metrics(
            true_class=self.train_class,
            pred_class=pred_train_preds,
        )
        valid_scores = self.calc_perf_metrics(
            true_class=self.valid_class,
            pred_class=pred_valid_preds,
        )

        train_score = train_scores.loc[
            train_scores["Metric"] == f"f_{self.fbeta_score_beta}_score", "Score"
        ]
        valid_score = valid_scores.loc[
            valid_scores["Metric"] == f"f_{self.fbeta_score_beta}_score", "Score"
        ]

        mlflow.log_metric("training_score", train_score)
        mlflow.log_metric("validation_score", valid_score)

        # Return the validation score to ensure it's used for model selsection
        return -valid_score

    def tune_model(
        self,
        max_search_iters: int = 100,
        model_opt_timeout_secs: int = 180,
    ) -> optuna.study.Study:
        """Performs hyperparameters optimization using Optuna package.

        Args:
            max_search_iters (int): maximum number of search iterations.
            model_opt_timeout_secs (int): maximum time in seconds to optimize model.

        Returns:
            study (optuna.study.Study): optuna study object.
        """

        # Turn off optuna log notes
        # Note: uncomment this during dev to see warnings.
        optuna.logging.set_verbosity(optuna.logging.WARN)

        # A callback function to output a log only when the best value is updated
        # Note: this callback may show incorrect values when optimizing an objective
        # function with n_jobs > 1
        def logging_callback(study, frozen_trial):
            previous_best_value = study.user_attrs.get("previous_best_value", None)
            if previous_best_value != study.best_value:
                study.set_user_attr("previous_best_value", study.best_value)
                logger.info(
                    "Trial %d finished, best value: %s hyperparameters: %s",
                    int(frozen_trial.number),
                    frozen_trial.value,
                    frozen_trial.params,
                )

        sampler = optuna.samplers.TPESampler(
            n_startup_trials=int(0.1 * max_search_iters),
            warn_independent_sampling=False,
            multivariate=True,
            constant_liar=True,
        )
        study = optuna.create_study(sampler=sampler)

        logger.info(
            "Hyperparameter Optimization of %s Starts ...\n", self.classifier_name
        )

        study.optimize(
            func=self.objective_function,
            n_trials=max_search_iters,
            timeout=model_opt_timeout_secs,
            gc_after_trial=False,  # Set to True if memory consumption increases over several trials
            callbacks=[logging_callback],
        )

        return study

    def tune_model_in_parallel(
        self,
        max_search_iters: int = 100,
        n_parallel_jobs: int = 1,
        model_opt_timeout_secs: int = 180,
    ) -> optuna.study.Study:
        """Tunes model hyperparameters using Optuna package.

        Args:
            max_search_iters (int): maximum number of search iterations.
            n_parallel_jobs (int): number of parallel jobs.
            model_opt_timeout_secs (int): maximum time in seconds to optimize model.

        Returns:
            study (optuna.study.Study): optuna study object.
        """

        sampler = optuna.samplers.TPESampler(
            n_startup_trials=int(0.1 * max_search_iters),
            warn_independent_sampling=False,
            multivariate=True,
            constant_liar=True,
        )
        client = Client()
        study = optuna_distributed.from_study(
            optuna.create_study(sampler=sampler), client=client
        )

        study.optimize(
            direction="minimize",
            func=self.objective_function,
            n_trials=max_search_iters,
            n_jobs=n_parallel_jobs,
            timeout=model_opt_timeout_secs,
        )

        # Shutdown Dask cluster
        client.shutdown()

        return study

    @staticmethod
    def create_pipeline(
        preprocessor_step: ColumnTransformer,
        selector_step: VarianceThreshold,
        model: Callable,
    ) -> Pipeline:
        """Creates a pipeline including data prep steps and fitted model.

        Args:
            preprocessor_step (ColumnTransformer): data preprocessing step.
            selector_step (VarianceThreshold): feature selection step.
            model (Callable): model object.

        Returns:
            pipeline (Pipeline): pipeline including data prep steps and fitted model.
        """

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor_step),
                ("selector", selector_step),
                ("classifier", model),
            ]
        )

        return pipeline

    ###########################################################
    def fit_pipeline(
        self,
        train_features: pd.DataFrame,
        preprocessor_step: ColumnTransformer,
        selector_step: VarianceThreshold,
        model: Callable,
    ) -> Pipeline:
        """Fits a pipeline including model with data preprocessing steps.

        Args:
            train_features (pd.DataFrame): train features.
            preprocessor_step (ColumnTransformer): data preprocessing step.
            selector_step (VarianceThreshold): feature selection step.
            model (Callable): model object.

        Returns:
            pipeline (Pipeline): fitted pipeline.
        """

        # Fit a pipeline
        pipeline = self.create_pipeline(
            preprocessor_step=preprocessor_step,
            selector_step=selector_step,
            model=model,
        )
        pipeline.fit(train_features, self.train_class)

        return pipeline


class ModelEvaluator(ModelOptimizer):
    """A class to evaluate models beyond mere scores, like feature
    importance plots and confusion matrices that are not produced
    by ModelOptimizer class.

    Attributes:
        ModelOptimizer (ModelOptimizer): ModelOptimizer class.
        pipeline (Pipeline): fitted pipeline.
        train_features (pd.DataFrame): train features.
        train_class (np.ndarray): train class labels.
        valid_features (pd.DataFrame): validation features.
        valid_class (np.ndarray): validation class labels.
        encoded_pos_class_label (int): encoded positive class label.
        fbeta_score_beta (float): beta value for fbeta score.
        is_voting_ensemble (bool): whether the model is a voting ensemble or not.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        train_features: pd.DataFrame,
        train_class: np.ndarray,
        valid_features: pd.DataFrame,
        valid_class: np.ndarray,
        encoded_pos_class_label: int,
        fbeta_score_beta: float = 0.5,
        is_voting_ensemble: bool = False,
        artifacts_path: str = "tmp",
    ) -> None:
        super().__init__(
            train_features_preprocessed=None,
            train_class=None,
            valid_features_preprocessed=None,
            valid_class=None,
            n_features=None,
            model=(
                pipeline.named_steps["classifier"]
                if is_voting_ensemble
                else pipeline.named_steps["classifier"]
            ),
            search_space_params=None,
            encoded_pos_class_label=encoded_pos_class_label,
            fbeta_score_beta=fbeta_score_beta,
            is_voting_ensemble=is_voting_ensemble,
        )

        self.pipeline = pipeline
        self.train_features = train_features
        self.train_class = train_class
        self.valid_features = valid_features
        self.valid_class = valid_class
        self.artifacts_path = artifacts_path

    def plot_feature_importance(
        self,
        feature_importance_scores: np.ndarray,
        feature_names: list,
        figure_obj: Figure,
        n_top_features: int = 30,
        font_size: int = 10,
        fig_size: tuple = (8, 12),
    ) -> Figure:
        """Plots top feature importance with their encoded names. It requires
        an empty figure object (figure_obj) to add plot to it and return
        plot as a figure object that can be logged.

        Args:
            feature_importance_scores (np.ndarray): feature importance scores.
            feature_names (list): list of feature names.
            figure_obj (Figure): empty figure object.
            n_top_features (int): number of top features to plot.
            font_size (int): font size.
            fig_size (tuple): figure size.

        Returns:
            figure_obj (plt.Figure): figure object that can be logged.

        Raises:
            ValueError: if feature names and feature importance scores have different lengths.
        """
        try:
            feat_importances = pd.Series(feature_importance_scores, index=feature_names)
            feat_importances = feat_importances.nlargest(n_top_features, keep="all")
            feat_importances.sort_values(ascending=True, inplace=True)

            feat_importances.plot(
                kind="barh", fontsize=font_size, legend=None, figsize=fig_size
            )
            plt.title(f"Top {n_top_features} important features")
            plt.xlabel("Contribution")
            plt.ylabel("Feature Name")
            plt.show()

        except ValueError as e:
            logger.info("Error plotting feature importance --> %s", e)

        return figure_obj

    def extract_feature_importance(
        self,
        pipeline: Pipeline,
        num_feature_names: list,
        cat_feature_names: list,
        n_top_features: int = 30,
        figure_size: tuple = (24, 36),
        font_size: float = 10.0,
    ) -> None:
        """Extracts feature importance and returns figure object and
        column names from fitted pipeline.

        Args:
            pipeline (Pipeline): fitted pipeline.
            num_feature_names (list): list of numerical feature names.
            cat_feature_names (list): list of categorical feature names.
            n_top_features (int): number of top features to plot.
            figure_size (tuple): figure size.
            font_size (float): font size.

        Raises:
            ValueError: if num_feature_names and cat_feature_names are both None.
            Exception: if any error occurs during feature importance extraction.
        """

        # Catch any error raised in this method to prevent experiment
        # from registering a model as it's not worth failing experiment.
        try:
            # Note: there is no feature_importances_ attribute for LogisticRegression, hence,
            # this if statement is needed.
            classifier_name = pipeline.named_steps["classifier"].__class__.__name__
            if classifier_name == "LogisticRegression":
                # Return LR coefficients instead.
                feature_importance_scores = pipeline.named_steps["classifier"].coef_[0]

            if classifier_name not in [
                "LogisticRegression",
                "VotingClassifier",
            ]:
                feature_importance_scores = pipeline.named_steps[
                    "classifier"
                ].feature_importances_

                # Get feature names
                num_feature_names = (
                    [] if num_feature_names is None else num_feature_names
                )
                cat_feature_names = (
                    [] if cat_feature_names is None else cat_feature_names
                )
                if len(num_feature_names) == 0 and len(cat_feature_names) > 0:
                    col_names = list(
                        pipeline.named_steps["preprocessor"]
                        .transformers_[1][1]
                        .named_steps["onehot_encoder"]
                        .get_feature_names_out(cat_feature_names)
                    )

                elif len(num_feature_names) > 0 and len(cat_feature_names) == 0:
                    col_names = num_feature_names

                elif len(num_feature_names) > 0 and len(cat_feature_names) > 0:
                    col_names = num_feature_names + list(
                        pipeline.named_steps["preprocessor"]
                        .transformers_[1][1]
                        .named_steps["onehot_encoder"]
                        .get_feature_names_out(cat_feature_names)
                    )

                else:
                    raise ValueError("Numerical or categorical must be provided.")

                # Extract transformed feature names after feature selection
                col_names = [
                    i
                    for (i, v) in zip(
                        col_names,
                        list(pipeline.named_steps["selector"].get_support()),
                    )
                    if v
                ]

                # Log feature importance figure
                self._log_feature_importance_fig(
                    classifier_name=classifier_name,
                    feature_importance_scores=feature_importance_scores,
                    col_names=col_names,
                    n_top_features=n_top_features,
                    figure_size=figure_size,
                    font_size=font_size,
                    fig_name=f"{classifier_name}_feature_importance.png",
                )

        except Exception as e:  # pylint: disable=W0718
            logger.info("Feature importance extraction error --> %s", e)
            col_names = None

    def _log_feature_importance_fig(
        self,
        classifier_name: str,
        feature_importance_scores: np.ndarray,
        col_names: list,
        n_top_features: int = 30,
        figure_size: tuple = (24, 36),
        font_size: float = 10.0,
        fig_name: str = "feature_importance.png",
    ) -> None:
        """Plots feature importance figure given feature importance scores and
        column names and logs it to workspace.

        Args:
            classifier_name (str): name of the classifier.
            feature_importance_scores (np.ndarray): feature importance scores.
            col_names (list): list of column names.
            n_top_features (int): number of top features to plot.
            figure_size (tuple): figure size.
            font_size (float): font size.
            fig_name (str): figure name.
        """

        # Log feature importance figure
        if classifier_name not in ["VotingClassifier"]:
            feature_importance_fig = plt.figure(figsize=figure_size)
            feature_importance_fig = self.plot_feature_importance(
                feature_importance_scores=feature_importance_scores,
                feature_names=col_names,
                figure_obj=feature_importance_fig,
                n_top_features=n_top_features,
                font_size=font_size,
                fig_size=figure_size,
            )

            mlflow.log_figure(feature_importance_fig, fig_name)

    @staticmethod
    def plot_roc_curve(
        y_true: np.array,
        y_pred: np.array,
        fig_size: tuple = (6, 6),
    ) -> Figure:
        """Plots receiver operating characteristic (ROC) curve for a binary classification model.
        It shows the trade-off between the true positive rate (TPR) and the false positive rate
        (FPR) for different probability thresholds.

        Args:
            y_true (np.ndarray): true labels of the data, either 0 or 1.
            y_pred (np.ndarray): predicted probabilities of the positive class by the model.
            fig_size (tuple): figure size.

        Returns
            fig (plt.Figure): matplotlib figure object that can be logged and saved.
        """

        # Compute the FPR, TPR, and thresholds
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)

        # Plot the ROC curve
        fig = plt.figure(figsize=fig_size)
        plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {auc:.2f})")
        plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Random guess")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.show()

        return fig

    @staticmethod
    def plot_precision_recall_curve(
        y_true: np.array,
        y_pred: np.array,
        fig_size: tuple = (6, 6),
    ) -> Figure:
        """Plots plots the precision-recall curve for a binary classification model.
        It shows the trade-off between the precision and the recall for different
        probability thresholds.

        Args:
            y_true (np.ndarray): true labels of the data, either 0 or 1.
            y_pred (np.ndarray): predicted probabilities of the positive class by the model.
            fig_size (tuple): figure size.

        Returns
            fig (plt.Figure): matplotlib figure object that can be logged and saved.
        """

        # Compute the precision, recall, and thresholds
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)

        # Plot the precision-recall curve
        fig = plt.figure(figsize=fig_size)
        plt.plot(
            recall,
            precision,
            color="green",
            label=f"Precision-Recall curve (AP = {ap:.2f})",
        )
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="upper right")
        plt.show()

        return fig

    @staticmethod
    def plot_cumulative_gains(
        y_true: np.array,
        y_pred: np.array,
        fig_size: tuple = (6, 6),
    ) -> Figure:
        """Plots the cumulative gains curve for a binary classification model. It
        shows the percentage of positive cases captured by the model as a function
        of the percentage of the sample that is predicted as positive.

        Args:
            y_true (np.ndarray): true labels of the data, either 0 or 1.
            y_pred (np.ndarray): predicted probabilities of the positive class by the model.
            fig_size (tuple): figure size.

        Returns:
            fig (plt.Figure): matplotlib figure object.
        """

        kds.metrics.plot_cumulative_gain(y_true, y_pred, figsize=fig_size)
        fig = plt.gcf()

        return fig

    @staticmethod
    def plot_lift_curve(
        y_true: np.array, y_pred: np.array, fig_size: tuple = (6, 6)
    ) -> Figure:
        """Plots the lift curve for a binary classification model.
        It shows the ratio of the positive cases captured by the model to the
        baseline (random) model as a function of the percentage of the sample
        that is predicted as positive.

        Args:
            y_true (np.ndarray): true labels of the data, either 0 or 1.
            y_pred (np.ndarray): predicted probabilities of the positive class by the model.
            fig_size (tuple): figure size.

        Returns
            fig (plt.Figure): matplotlib figure object.
        """

        kds.metrics.plot_lift(y_true, y_pred, figsize=fig_size)
        fig = plt.gcf()

        return fig

    @staticmethod
    def convert_metrics_from_df_to_dict(
        scores: pd.DataFrame,
        prefix: Optional[str] = None,
    ) -> dict:
        """Converts scors on training and validation sets from dataframe to
        dictionary for results logging.

        Args:
            scores (pd.DataFrame): dataframe of scores.
            prefix (str): prefix to add to metric names.

        Returns:
            metrics_values (dict): dictionary of metric names and values.
        """

        metrics_values = dict(scores.set_index("Metric").iloc[:, 0])

        # Add prefix to metric names to distinguish train scores from valid scores
        if prefix is not None:
            metrics_values = {prefix + k: v for k, v in metrics_values.items()}

        return metrics_values

    @staticmethod
    def convert_class_prob_to_label(
        value,
        class_threshold: float = 0.5,
        class_label_above_threshold: str = "Y",
        class_label_below_threshold: str = "N",
    ):
        """
        Converts class probability to N/Y label based on threshold value.

        Args:
            value (float): class probability
            class_threshold (float): threshold value to map class probability to No/Yes label
            class_label_above_threshold (string): class label if value >= class_threshold
            class_label_below_threshold (string): class label if value < class_threshold

        Returns:
            predicted_class_label (string): class label ("No" if value < class_threshold, otherwise "Yes")
        """

        if value < class_threshold:
            predicted_class_label = class_label_below_threshold
        else:
            predicted_class_label = class_label_above_threshold

        return predicted_class_label

    def calc_registered_model_scores(
        self,
        serialized_model: Pipeline,
        dataset: pd.DataFrame,
        true_labels: np.ndarray,
        model_name: str,
        model_version: str,
        pos_class_threshold: float = 0.5,
        pos_class_label: str = "Y",
        neg_class_label: str = "N",
    ) -> Union[pd.DataFrame, list]:
        """
        Evaluates binary classifier (pkl files) and returns a pandas dataframe with the scores
        of each model and predicated class labels using the provided threshold value.

        Args:
            serialized_model (Pipeline): the imported pickle file with pipeline transformer.
            dataset (pd.DataFrame): dataset to predict its class label.
            true_labels (np.ndarray): encoded actual class labels.
            model_name (str): full model name as it will appear in the final dataframe.
            model_version (str): model version.
            pos_class_threshold (float): threshold value for positive class label.
            pos_class_label (str): positive class label.
            neg_class_label (str): negative class label.

        Returns:
            models_scores (pandas dataframe): models scores alongsoide a column for model's name.
            pred_class_label (list): predicted class label (not encoded) using the provided thershold value.
        """

        predicted_class_prob = pd.DataFrame(
            serialized_model.predict_proba(dataset),
            columns=[neg_class_label, pos_class_label],
        )
        pred_class_label = (
            predicted_class_prob[pos_class_label]
            .map(
                lambda x: self.convert_class_prob_to_label(
                    x,
                    class_threshold=pos_class_threshold,
                    class_label_above_threshold=pos_class_label,
                    class_label_below_threshold=neg_class_label,
                )
            )
            .values
        )
        predicted_class_labels = LabelEncoder().fit_transform(pred_class_label)

        # Calculate model performance metrics
        models_scores = self.calc_perf_metrics(
            true_class=true_labels,
            pred_class=predicted_class_labels,
        )

        # Add model and version
        models_scores.loc[:, ["Model Name"]] = model_name
        models_scores.loc[:, ["Model Version"]] = model_version

        return models_scores, pred_class_label

    def evaluate_model_perf(
        self,
        class_encoder: Optional[LabelEncoder] = None,
        pos_class_label_thresh: float = 0.5,
    ) -> Union[pd.DataFrame, pd.DataFrame, list]:
        """Evaluates the best model returned by hyperparameters optimization procedure
        on both training and validation set and logs confusion matrices, calibration curve,
        ROC curve, precision-recall curve, cumulative gains curve, and lift curve.

        Args:
            class_encoder (LabelEncoder): class encoder object.
            pos_class_label_thresh (float): decision threshold value for positive class.

        Returns:
            train_scores (pd.DataFrame): training scores.
            valid_scores (pd.DataFrame): validation scores.
            original_class_labels (list): list of original class labels.
        """

        # Generate class labels for validation set based on decision threshold value (0.5)
        pred_train_probs = self.pipeline.predict_proba(self.train_features)
        pred_train_class = self._get_pred_class(
            pred_train_probs, pos_class_label_thresh
        )
        pred_valid_probs = self.pipeline.predict_proba(self.valid_features)
        pred_valid_class = self._get_pred_class(
            pred_valid_probs, pos_class_label_thresh
        )

        # Calculate performance metrics on train and validation sets
        train_scores = self.calc_perf_metrics(
            true_class=self.train_class,
            pred_class=pred_train_class,
        )
        valid_scores = self.calc_perf_metrics(
            true_class=self.valid_class,
            pred_class=pred_valid_class,
        )

        # Extract original class label names
        # Note: only if class_encoder is provided so expressive class
        # labels can be used for confusion matrix.
        if class_encoder is not None:
            (
                original_train_class,
                pred_original_train_class,
                original_class_labels,
            ) = self._get_original_class_labels(
                true_class=self.train_class,
                pred_class=pred_train_class,
                class_encoder=class_encoder,
            )

            (
                original_valid_class,
                pred_original_valid_class,
                _,
            ) = self._get_original_class_labels(
                true_class=self.valid_class,
                pred_class=pred_valid_class,
                class_encoder=class_encoder,
            )

            # Log confusion matrices
            self._log_confusion_matrix(
                original_class=original_train_class,
                pred_original_class=pred_original_train_class,
                original_class_labels=original_class_labels,
                saved_fig_path=f"/{self.artifacts_path}/train_set_cm.png",
                normalize=None,
                fig_size=(11, 11),
            )
            self._log_confusion_matrix(
                original_class=original_train_class,
                pred_original_class=pred_original_train_class,
                original_class_labels=original_class_labels,
                saved_fig_path=f"/{self.artifacts_path}/norm_train_set_cm.png",
                normalize="true",
                fig_size=(11, 11),
            )
            self._log_confusion_matrix(
                original_class=original_valid_class,
                pred_original_class=pred_original_valid_class,
                original_class_labels=original_class_labels,
                saved_fig_path=f"/{self.artifacts_path}/valid_set_cm.png",
                normalize=None,
                fig_size=(11, 11),
            )
            self._log_confusion_matrix(
                original_class=original_valid_class,
                pred_original_class=pred_original_valid_class,
                original_class_labels=original_class_labels,
                saved_fig_path=f"/{self.artifacts_path}/norm_valid_cm.png",
                normalize="true",
                fig_size=(11, 11),
            )

            logger.info("Confusion matrices logged successfully.")

        self._log_calibration_curve(
            pred_probs=pred_valid_probs,
            true_class=self.valid_class,
            saved_fig_path=f"/{self.artifacts_path}/calibration_curve.png",
        )
        logger.info("Calibration curve logged successfully.")

        self._log_roc_curve(
            pred_probs=pred_valid_probs,
            true_class=self.valid_class,
            encoded_pos_class_label=self.encoded_pos_class_label,
            fig_name="roc_curve.png",
        )
        logger.info("ROC curve logged successfully.")
        self._log_precision_recall_curve(
            pred_probs=pred_valid_probs,
            true_class=self.valid_class,
            encoded_pos_class_label=self.encoded_pos_class_label,
            fig_name="precision_recall_curve.png",
        )
        logger.info("Precision-recall curve logged successfully.")
        self._log_cumulative_gains(
            pred_probs=pred_valid_probs,
            true_class=self.valid_class,
            fig_name="cumulative_gains.png",
        )
        logger.info("Cumulative gains curve logged successfully.")
        self._log_lift_curve(
            pred_probs=pred_valid_probs,
            true_class=self.valid_class,
            fig_name="lift_curve.png",
        )
        logger.info("Lift curve logged successfully.")

        return train_scores, valid_scores

    def _get_pred_class(self, pred_probs: np.ndarray, threshold: float) -> np.ndarray:
        """Returns predicted class labels based on decision threshold value.

        Args:
            pred_probs (np.ndarray): predicted probabilities of the positive class.
            threshold (float): decision threshold value for positive class.

        Returns:
            pred_class (np.ndarray): predicted class labels.
        """

        pred_class = np.where(
            pred_probs[:, self.encoded_pos_class_label] > threshold, 1, 0
        )

        return pred_class

    def _get_original_class_labels(
        self,
        true_class: np.ndarray,
        pred_class: np.ndarray,
        class_encoder: LabelEncoder,
    ) -> list:
        """Returns original class labels from encoded class labels.

        Args:
            pred_class (np.ndarray): predicted class labels.
            class_encoder (LabelEncoder): class encoder object.

        Returns:
            original_class_labels (list): list of original class labels.
        """

        # Extract original class label names, which can be expressive, i.e., not encoded.
        original_class_labels = list(
            class_encoder.inverse_transform(self.pipeline.classes_)
        )

        # Extract expressive class names for confusion matrix
        true_original_class = class_encoder.inverse_transform(true_class)
        pred_original_class = class_encoder.inverse_transform(pred_class)

        return true_original_class, pred_original_class, original_class_labels

    def _log_confusion_matrix(
        self,
        original_class: np.ndarray,
        pred_original_class: np.ndarray,
        original_class_labels: list,
        saved_fig_path: str = "confusion_maxtrix.png",
        normalize: Literal[None, "true"] = None,
        fig_size: tuple = (11, 11),
    ) -> None:
        """Logs confusion matrices using expressive labels, e.g., Y/N, instead of encoded
        class labels. If normalize is set to 'true', then confusion matrix is normalized,
        otherwise it's not normalized. It saved_fig_path is provided, then confusion matrix
        figure is saved to that path, otherwise it's saved as 'confusion_matrix.png' in the
        current working directory.

        Args:
            original_class (np.ndarray): true class labels (expressive labels).
            pred_original_class (np.ndarray): predicted class labels (expressive labels).
            original_class_labels (list): list of expressive class labels.
            saved_fig_path (str): path to save confusion matrix figure. Defaults to None (not saved).
            normalize (str): normalization type. Defaults to None (not normalized).
            fig_size (tuple): figure size. Defaults to (11, 11).
        """

        # Plot confusion matrix
        calc_cm = confusion_matrix(
            y_true=original_class,
            y_pred=pred_original_class,
            labels=original_class_labels,
            normalize=normalize,
        )
        calc_cm_fig = ConfusionMatrixDisplay(
            confusion_matrix=calc_cm,
            display_labels=original_class_labels,
        )
        _, ax = plt.subplots(figsize=fig_size)
        calc_cm_fig.plot(ax=ax, xticks_rotation=75)

        # Log confusion matrix figure
        calc_cm_fig.figure_.savefig(saved_fig_path)
        mlflow.log_artifact(saved_fig_path)

    def _log_calibration_curve(
        self,
        pred_probs: np.ndarray,
        true_class: np.ndarray,
        saved_fig_path: str = "calibration_curve.png",
        n_bins: int = 10,
    ) -> None:
        """Logs calibration curve in the experiment. It must be calculated on validation or test set.

        Args:
            pred_probs (np.ndarray): predicted probabilities of the positive class.
            true_class (np.ndarray): true class labels.
            saved_fig_path (str):  path to save calibration curve figure. Defaults to None.
            n_bins (int): number of bins. Defaults to 10.
        """

        _ = CalibrationDisplay.from_predictions(
            true_class,
            pred_probs[:, self.encoded_pos_class_label],
            n_bins=n_bins,
        )

        plt.savefig(saved_fig_path)
        mlflow.log_artifact(saved_fig_path)

    def _log_roc_curve(
        self,
        pred_probs: np.ndarray,
        true_class: np.ndarray,
        encoded_pos_class_label: int,
        fig_name: str = "roc_curve.png",
        fig_size: tuple = (6, 6),
    ) -> None:
        """Logs ROC curve in the experiment.

        Args:
            pred_probs (np.ndarray): predicted probabilities of the positive class.
            true_class (np.ndarray): true class labels (encoded).
            encoded_pos_class_label (int): encoded positive class label.
            fig_name (str): name of ROC curve figure. Defaults to 'roc_curve.png'.
            fig_size (tuple): figure size. Defaults to (6, 6).
        """

        roc_curve_fig = self.plot_roc_curve(
            y_true=true_class,
            y_pred=pred_probs[:, encoded_pos_class_label],
            fig_size=fig_size,
        )
        mlflow.log_figure(roc_curve_fig, fig_name)

    def _log_precision_recall_curve(
        self,
        pred_probs: np.ndarray,
        true_class: np.ndarray,
        encoded_pos_class_label: int,
        fig_name: str = "precision_recall_curve.png",
        fig_size: tuple = (6, 6),
    ) -> None:
        """Logs precision-recall curve in the experiment.

        Args:
            pred_probs (np.ndarray): predicted probabilities of the positive class.
            true_class (np.ndarray): true class labels.
            encoded_pos_class_label (int): encoded positive class label.
            fig_name (str): name of precision-recall curve figure. Defaults to 'precision_recall_curve.png'.
            fig_size (tuple): figure size. Defaults to (6, 6).
        """

        prec_recall_fig = self.plot_precision_recall_curve(
            y_true=true_class,
            y_pred=pred_probs[:, encoded_pos_class_label],
            fig_size=fig_size,
        )
        mlflow.log_figure(prec_recall_fig, fig_name)

    def _log_cumulative_gains(
        self,
        pred_probs: np.ndarray,
        true_class: np.ndarray,
        fig_name: str = "cumulative_gains_curve.png",
        fig_size: tuple = (6, 6),
    ) -> None:
        """Logs cumulative gains curve in the experiment.

        Args:
            pred_probs (1-D np.ndarray): predicted probabilities of the positive class.
            true_class (np.ndarray): true class labels.
            fig_name (str): name of cumulative gains curve figure. Defaults to 'cumulative_gains_curve.png'.
            fig_size (tuple): figure size. Defaults to (6, 6).
        """

        cum_gain_fig = self.plot_cumulative_gains(
            y_true=true_class,
            y_pred=pred_probs[:, self.encoded_pos_class_label],
            fig_size=fig_size,
        )
        mlflow.log_figure(cum_gain_fig, fig_name)

    def _log_lift_curve(
        self,
        pred_probs: np.ndarray,
        true_class: np.ndarray,
        fig_name: str = "lift_curve.png",
        fig_size: tuple = (6, 6),
    ) -> None:
        """Logs lift curve in the experiment.

        Args:
            pred_probs (1-D np.ndarray): predicted probabilities of the positive class.
            true_class (np.ndarray): true class labels.
            fig_name (str): name of lift curve figure. Defaults to 'lift_curve.png'.
            fig_size (tuple): figure size. Defaults to (6, 6).
        """

        lift_curve_fig = self.plot_lift_curve(
            y_true=true_class,
            y_pred=pred_probs[:, self.encoded_pos_class_label],
            fig_size=fig_size,
        )
        mlflow.log_figure(lift_curve_fig, fig_name)

    def plot_confusion_matrix(
        self,
        y_decoded: np.ndarray,
        pred_y_decoded: np.ndarray,
        decoded_class_labels: list,
        normalize_conf_mat: Literal["true", None] = None,
        confusion_matrix_fig_name: str = "confusion_matrix.png",
    ) -> None:
        """Plots confusion matrix: normalized and un-normalized."""
        confusion_mat_norm = confusion_matrix(
            y_true=y_decoded,
            y_pred=pred_y_decoded,
            labels=decoded_class_labels,
            normalize=normalize_conf_mat,
        )
        confusion_mat_norm = ConfusionMatrixDisplay(
            confusion_matrix=confusion_mat_norm,
            display_labels=decoded_class_labels,
        )
        _, ax = plt.subplots(figsize=(11, 11))
        confusion_mat_norm.plot(ax=ax, xticks_rotation=75)
        confusion_mat_norm.figure_.savefig(
            self.artifacts_path + confusion_matrix_fig_name
        )

    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 5,
        plot_title: str = "Calibration plot",
        calibration_matrix_fig_name: Optional[str] = None,
    ) -> None:
        """
        Plot calibration curve for est w/o and with calibration.

        Args:
            y_true (np.ndarray): true class labels.
            y_prob (np.ndarray): predicted probabilities.
            n_bins (int): number of bins to use for calibration curve.
            plot_title (str): title of the plot.
            calibration_matrix_fig_name (str): name of the figure to save.
        """

        # Calculate the calibration curve
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(prob_pred, prob_true, "s-", label="Predictions")
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax1.set_ylabel("Fraction of positives")
        ax1.set_xlabel("Mean predicted value")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend()
        plt.title(plot_title)
        if calibration_matrix_fig_name is not None:
            plt.savefig(self.artifacts_path + calibration_matrix_fig_name)
        plt.show()

    @staticmethod
    def select_best_model(
        models_scores: pd.DataFrame,
        selection_metric: str,
    ) -> str:
        """
        Selects the best model based on a given selection metric, like f1.

        Args:
            models_scores (pd.DataFrame): a dataframe with "Model Name" column and other columns each
                                            representing a performance metric, e.g., Recall and Precision.
            selection_metric (str): the column name of the selection metric.

        Returns:
            best_model_name (str): the name of the best model.

        Raises:
            ValueError: if models_scores is empty.
        """

        # Check if models scores dataframe is not empty
        if models_scores.shape[0] > 0:
            # Sort models by primary and secondary metrics
            models_scores.sort_values(
                by=[selection_metric],
                ascending=[False],
                inplace=True,
                na_position="last",
            )

            # Extract the name of the best model
            best_model_name = models_scores["Model Name"].iloc[0]
            best_model_version = int(models_scores["Model Version"].iloc[0])

        else:
            raise ValueError("Models scores dataframe 'models_scores' is empty.")

        return best_model_name, best_model_version, models_scores

    @staticmethod
    def calc_expected_calibration_error(
        pred_probs: np.ndarray,
        true_labels: np.ndarray,
        decision_thresh_val: float = 0.5,
        nbins: int = 5,
    ) -> float:
        """Calculates Expected Calibration Error (ECE) for a classification model
        (binary or multi-class). A perfectly calibrated model has an ECE = 0, i.e.,
        the lower the ECE is the better.

        Args:
            pred_probs (np.ndarray): predicted probability of positive class if
                                          binary classifier or max
            true_labels (np.ndarray): actual class label (encoded).
            decision_thresh_val (float): probabilities greater than these value will
                                            be considered positive and negative otherwise.
            nbins (int): number of bins (bins values between 0 and 1 into nbins equal
                         sized bins).

        Returns:
            ece (float): ECE value ([0, 1])
        """

        # Equal-size binning approach with nbins number of bins
        bin_boundaries = np.linspace(0, 1, nbins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        # Determine if classifier if binary or multi-class
        if len(np.unique(true_labels)) <= 2:
            # Keep predicted "probabilities" as is for binary classifier
            _confidences = pred_probs

            # Get binary predictions from confidences
            pred_label = (pred_probs > decision_thresh_val).astype(float)

        else:
            # Get max probability per sample i
            _confidences = np.max(pred_probs, axis=1)

            # Get predictions from confidences (positional in this case)
            pred_label = np.argmax(pred_probs, axis=1).astype(float)

        # Get a boolean list of correct/false predictions
        accuracies = pred_label == true_labels

        ece = np.zeros(1)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Determine if sample is in bin m (between bin lower & upper)
            in_bin = np.logical_and(
                _confidences > bin_lower.item(), _confidences <= bin_upper.item()
            )

            # Calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
            prop_in_bin = in_bin.astype(float).mean()

            if prop_in_bin.item() > 0:
                # Accuracy of bin m: acc(Bm)
                accuracy_in_bin = accuracies[in_bin].astype(float).mean()

                # Calculate the average confidence of bin m: conf(Bm)
                avg_confidence_in_bin = _confidences[in_bin].mean()

                # Calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece[0]


class ModelManager:
    """Loads and registers models in Azure workspace."""

    def __init__(self, registered_model_workspace: Workspace):
        self.ws = registered_model_workspace

    def load_registered_model(
        self,
        registered_model_name: str,
        registered_model_version: Optional[str] = None,
        is_mlflow_model: bool = False,
    ) -> Union[Pipeline, str]:
        """
        Imports registered models in Azure workspace if they exists. It's meant to be used in automated process
        where we need to evaluate retraining results (registered models) that may not be registered. One such case is when a
        retraining experiment results in an overfitted model, which should not be registered, so attempting to load that model
        will result in an error that might break model retraining pipeline. Hence, this function accounts for such situations.

        Args:
            registered_model_name (str): name of the registered model
            registered_model_version (str): the version of the registered model (default: None which gives the latest version)
            is_mlflow_model (bool): was the model registered using mlflow. Default to False.

        Returns:
            registered_model_pkl (Pipeline): pkl file of the registered model
            registered_model_path (str): local directory to downloaded model.
        """

        # Get register model path
        registered_model_path = Model.get_model_path(
            model_name=registered_model_name,
            version=registered_model_version,
            _workspace=self.ws,
        )

        if is_mlflow_model:
            registered_model_pkl = mlflow.sklearn.load_model(registered_model_path)

        else:
            registered_model_pkl = joblib.load(registered_model_path)

        return registered_model_pkl, registered_model_path

    @staticmethod
    def log_and_register_model(
        X_train: pd.DataFrame,
        pipeline: Pipeline,
        registered_model_name: str,
        model_uri: str,
        tags: dict,
        conda_env: str = "train_env.yml",
    ) -> None:
        """Logs and registers the model in the experiment using MLflow.

        Args:
            X_train (pd.DataFrame): train features.
            pipeline (Pipeline): fitted pipeline.
            registered_model_name (str): registered model name.
            model_uri (str): model URI.
            tags (dict): dictionary of tags.
            conda_env (str): conda environment file. Defaults to 'train_env.yml'.
        """

        signature = infer_signature(
            X_train, pipeline.predict(X_train.iloc[0 : X_train.shape[0], :])
        )
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",  # Model folder in the experiment UI
            signature=signature,
            input_example=X_train.iloc[0 : X_train.shape[0], :],
            conda_env=conda_env,
        )
        mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name,
            tags=tags,
        )

        logger.info(
            "Model %s was registered on %s",
            registered_model_name,
            datetime.now().strftime("%Y-%d-%m %H:%M:%S"),
        )
