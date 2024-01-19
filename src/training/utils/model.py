"""
This utility module includes functions for 
model optimization and evaluation.
"""

import os
from typing import Callable, Optional, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import optuna_distributed
import pandas as pd
import scikitplot as skplt
from comet_ml import API, ExistingExperiment, Experiment
from dask.distributed import Client
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
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

##########################################################


class ModelOptimizer:
    """A class to optimize model hyperparameters. It requires supplying preprocessed
    versions of the train and validation features to avoid fitting the whole pipeline
    in each objective function call during hyperparameters optimization."""

    # List of supported models in this class
    # Note: this private variable shouldn't be mutated outside the class. It
    # should be updated when a new model is added to the class, which requires
    # adding its search space definition to generate_search_space method.
    _supported_models = (
        "LogisticRegression",
        "RandomForestClassifier",
        "LGBMClassifier",
        "XGBClassifier",
    )

    def __init__(
        self,
        comet_exp: Experiment,
        train_features_preprocessed: pd.DataFrame,
        train_class: np.ndarray,
        valid_features_preprocessed: pd.DataFrame,
        valid_class: np.ndarray,
        n_features: int,
        model: Callable,
        fbeta_score_beta: float = 1.0,
        encoded_pos_class_label: int = 1,
        is_voting_ensemble: bool = False,
    ) -> None:
        self.comet_exp = comet_exp
        self.train_features_preprocessed = train_features_preprocessed
        self.train_class = train_class
        self.valid_features_preprocessed = valid_features_preprocessed
        self.valid_class = valid_class
        self.n_features = n_features
        self.model = model
        self.fbeta_score_beta = fbeta_score_beta
        self.encoded_pos_class_label = encoded_pos_class_label
        self.is_voting_ensemble = is_voting_ensemble
        self.classifier_name = self.model.__class__.__name__

        if not self.is_voting_ensemble:
            assert (
                self.classifier_name in self._supported_models
            ), f"Supported models are: {self._supported_models}. Got {self.classifier_name}!"

    def generate_search_space(self, trial: optuna.trial.Trial):
        """Returns search space provided model name."""

        if self.classifier_name == "LogisticRegression":
            params = {
                "C": trial.suggest_float("C", 0.001, 1000, log=False),
                "l1_ratio": trial.suggest_float(
                    "l1_ratio", 0.001, 1, log=False
                ),  # alpha param in elastic net, 0 means L2 and 1 means L1.
            }

        elif self.classifier_name == "RandomForestClassifier":
            params = {
                "max_features": trial.suggest_int(
                    "max_features",
                    int(max(1, 0.1 * self.n_features)),
                    self.n_features,
                ),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 8),
                "min_samples_split": trial.suggest_float(
                    "min_samples_split", 0, 1, log=False
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "criterion": trial.suggest_categorical(
                    "criterion", ["gini", "entropy"]
                ),
                "n_estimators": trial.suggest_int("n_estimators", 20, 100),
            }

        elif self.classifier_name == "LGBMClassifier":
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "max_bin": trial.suggest_int("max_depth", 100, 512),
                "min_child_weight": trial.suggest_float(
                    "min_child_weight", 1e-5, 1e2, log=True
                ),  # Min. sum of instance weight (hessian) needed in a child.
                "subsample": trial.suggest_float(
                    "subsample", 0.5, 1, log=False
                ),  # Subsample ratio of the training instances.
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.3, 1, log=False
                ),  # Subsample ratio of columns when constructing each tree.
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-4, 1e-1, log=True
                ),
                "n_estimators": trial.suggest_int("n_estimators", 20, 100),
            }

        elif self.classifier_name == "XGBClassifier":
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "min_child_weight": trial.suggest_float(
                    "min_child_weight", 1e-5, 1e2, log=True
                ),  # Min. sum of instance weight (hessian) needed in a child.
                "subsample": trial.suggest_float(
                    "subsample", 0.5, 1, log=False
                ),  # Subsample ratio of the training instances.
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.5, 1, log=False
                ),  # Subsample ratio of columns when constructing each tree.
                "reg_alpha": trial.suggest_float(
                    "reg_alpha", 0, 10, log=False
                ),  # L1 regularization term.
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 1, 10, log=False
                ),  # L2 regularization term.
                "gamma": trial.suggest_float(
                    "gamma", 1e-5, 1e2, log=True
                ),  # Minimum loss reduction required to make a further partition on a leaf node.
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-5, 1e2, log=True
                ),
                "n_estimators": trial.suggest_int("n_estimators", 20, 100),
            }

        return params

    def calculate_performance_metrics(
        self,
        true_class: ArrayLike,
        pred_class: ArrayLike,
    ) -> pd.DataFrame:
        """
        Calculates different performance metrics for binary classification models.

        Args:
            true_class (list): true class label.
            pred_class (list): predicted class label not probability.

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
        """Performs hyperparameters optimization for a specified model, where the
        search metric is fbeta score."""

        # Define parameters search space
        params = self.generate_search_space(trial=trial)

        # Fit model and calculate training score
        self.model.set_params(**params)
        self.model.fit(self.train_features_preprocessed, self.train_class)

        # Evaluate model on training and validation set
        # Note: default threshold of 0.5 is used for positive class but
        # other htreshold values can be used, which is problem-dependent.
        pred_train_preds = self.model.predict(self.train_features_preprocessed)
        pred_valid_preds = self.model.predict(self.valid_features_preprocessed)
        train_scores = self.calculate_performance_metrics(
            true_class=self.train_class,
            pred_class=pred_train_preds,
        )
        valid_scores = self.calculate_performance_metrics(
            true_class=self.valid_class,
            pred_class=pred_valid_preds,
        )

        train_score = train_scores.loc[
            train_scores["Metric"] == f"f_{self.fbeta_score_beta}_score", "Score"
        ]
        valid_score = valid_scores.loc[
            valid_scores["Metric"] == f"f_{self.fbeta_score_beta}_score", "Score"
        ]

        self.comet_exp.log_metric(name="training_score", value=train_score)
        self.comet_exp.log_metric(name="validation_score", value=valid_score)

        # Return the validation score to ensure it's used for model selsection
        return valid_score

    def tune_model(
        self,
        max_search_iters: int = 100,
        model_opt_timeout_secs: int = 180,
    ):
        """Tunes model hyperparameters using Optuna package."""

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
                print(
                    f"""\n
                    Trial {int(frozen_trial.number)} finished,
                    best value: {frozen_trial.value}
                    hyperparameters: {frozen_trial.params}."""
                )

        sampler = optuna.samplers.TPESampler(
            n_startup_trials=int(0.1 * max_search_iters),
            warn_independent_sampling=False,
            multivariate=True,
            constant_liar=True,
        )
        study = optuna.create_study(sampler=sampler)

        print(
            f"""\n
        ----------------------------------------------------------------
        --- Hyperparameter Optimization of {self.classifier_name} Starts ...
        ----------------------------------------------------------------\n"""
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
    ):
        """Tunes model hyperparameters using Optuna package."""

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
            direction="maximize",
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
        """Creates a pipeline including data prep steps and fitted model."""

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
        """Fits a pipeline including model with data preprocessing steps."""

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
    by ModelOptimizer class."""

    def __init__(
        self,
        comet_exp: Experiment,
        pipeline: Pipeline,
        train_features: pd.DataFrame,
        train_class: np.ndarray,
        valid_features: pd.DataFrame,
        valid_class: np.ndarray,
        fbeta_score_beta: float = 0.5,
        is_voting_ensemble: bool = False,
    ) -> None:
        super().__init__(
            self,
            train_features_preprocessed=None,
            train_class=None,
            valid_features_preprocessed=None,
            valid_class=None,
            n_features=None,
            model=pipeline.named_steps["classifier"]
            if is_voting_ensemble
            else pipeline.named_steps["classifier"],
            is_voting_ensemble=is_voting_ensemble,
        )

        self.comet_exp = comet_exp
        self.pipeline = pipeline
        self.train_features = train_features
        self.train_class = train_class
        self.valid_features = valid_features
        self.valid_class = valid_class
        self.fbeta_score_beta = fbeta_score_beta
        self.is_voting_ensemble = is_voting_ensemble

    def plot_feature_importance(
        self,
        feature_importance_scores: np.ndarray,
        feature_names: list,
        figure_obj: Figure,
        n_top_features: int = 30,
        font_size: int = 10,
        fig_size: tuple = (8, 12),
    ) -> None:
        """
        Plots top feature importance with their encoded names. It requires
        an empty figure object (figure_obj) to add plot to it and return
        plot as a figure object that can be logged.
        """
        try:
            feat_importances = pd.Series(feature_importance_scores, index=feature_names)
            feat_importances = feat_importances.nlargest(n_top_features, keep="all")
            feat_importances.sort_values(ascending=True, inplace=True)

            feat_importances.plot(
                kind="barh", fontsize=font_size, legend=None, figsize=fig_size
            )
            plt.title(f"Top {n_top_features} important features")
            plt.show()

        except ValueError as e:
            print(f"Error plotting feature importance --> {e}")

        return figure_obj

    def extract_feature_importance(
        self,
        pipeline: Pipeline,
        num_feature_names: Optional[list] = None,
        cat_feature_names: Optional[list] = None,
        n_top_features: int = 30,
        figure_size: tuple = (24, 36),
        font_size: float = 10.0,
    ) -> None:
        """Extracts feature importance and returns figure object and
        column names from fitted pipeline."""

        # Catch any error raised in this function to prevent experiment
        # from registering a model as it's not worth failing experiment for
        # an error in this function.
        try:
            # Get feature names
            if num_feature_names is None and cat_feature_names is not None:
                col_names = list(
                    pipeline.named_steps["preprocessor"]
                    .transformers_[1][1]
                    .named_steps["onehot_encoder"]
                    .get_feature_names_out(cat_feature_names)
                )

            elif num_feature_names is not None and cat_feature_names is None:
                col_names = num_feature_names

            elif num_feature_names is not None and cat_feature_names is not None:
                col_names = num_feature_names + list(
                    pipeline.named_steps["preprocessor"]
                    .transformers_[1][1]
                    .named_steps["onehot_encoder"]
                    .get_feature_names_out(cat_feature_names)
                )

            else:
                raise ValueError(
                    f"{num_feature_names} and/or {cat_feature_names} must be provided."
                )

            # Extract transformed feature names
            col_names = [
                i
                for (i, v) in zip(
                    col_names,
                    list(pipeline.named_steps["selector"].get_support()),
                )
                if v
            ]

            print(
                f"No. of features including encoded categorical features: {len(col_names)}"
            )

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
                self.comet_exp.log_figure(
                    figure_name="Feature Importance",
                    figure=feature_importance_fig,
                    overwrite=True,
                )

        except Exception as e:  # pylint: disable=W0718
            print(f"Feature importance extraction error --> {e}")
            feature_importance_fig, col_names = None, None

    @staticmethod
    def plot_roc_curve(
        y_true: np.array,
        y_pred: np.array,
        fig_size: tuple = (6, 6),
    ) -> None:
        """
        Plots receiver operating characteristic (ROC) curve for a binary classification model.
        It shows the trade-off between the true positive rate (TPR) and the false positive rate
        (FPR) for different probability thresholds.

        Args:
            y_true (np.ndarray): true labels of the data, either 0 or 1.
            y_pred (np.ndarray): predicted probabilities of the positive class by the model.

        Returns
            ax.get_figure(): figure object that can be logged and saved.
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
    ) -> None:
        """
        Plots plots the precision-recall curve for a binary classification model.
        It shows the trade-off between the precision and the recall for different
        probability thresholds.

        Args:
            y_true (np.ndarray): true labels of the data, either 0 or 1.
            y_pred (np.ndarray): predicted probabilities of the positive class by the model.

        Returns
            ax.get_figure(): figure object that can be logged and saved.
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
    ) -> None:
        """
        Plots the cumulative gains curve for a binary classification model. It
        shows the percentage of positive cases captured by the model as a function
        of the percentage of the sample that is predicted as positive.

        Args:
            y_true (np.ndarray): true labels of the data, either 0 or 1.
            y_pred (np.ndarray): predicted probabilities of the positive class by the model.

        Returns:
            ax.get_figure(): figure object that can be logged and saved.
        """

        ax = skplt.metrics.plot_cumulative_gain(y_true, y_pred, figsize=fig_size)
        plt.title("Cumulative Gains Curve")
        plt.xlabel("Percentage of sample")
        plt.ylabel("Percentage of positive outcomes")
        plt.show()

        return ax.get_figure()

    @staticmethod
    def plot_lift_curve(
        y_true: np.array, y_pred: np.array, fig_size: tuple = (6, 6)
    ) -> None:
        """
        This function plots the lift curve for a binary classification model.
        It shows the ratio of the positive cases captured by the model to the
        baseline (random) model as a function of the percentage of the sample
        that is predicted as positive.

        Args:
            y_true (np.ndarray): true labels of the data, either 0 or 1.
            y_pred (np.ndarray): predicted probabilities of the positive class by the model.

        Returns
            ax.get_figure(): figure object that can be logged and saved.
        """

        ax = skplt.metrics.plot_lift_curve(y_true, y_pred, figsize=fig_size)
        plt.title("Lift Curve")
        plt.xlabel("Percentage of samples")
        plt.ylabel("Lift")
        plt.show()

        return ax.get_figure()

    @staticmethod
    def convert_metrics_from_df_to_dict(
        scores: pd.DataFrame,
        prefix: Optional[str] = None,
    ) -> dict:
        """ "Converts scors on training and validation sets from dataframe to
        dictionary for results logging."""

        metrics_values = dict(scores.set_index("Metric").iloc[:, 0])

        # Add prefix to metric names to distinguish train scores from valid scores
        if prefix is not None:
            metrics_values = {prefix + k: v for k, v in metrics_values.items()}

        return metrics_values

    def evaluate_model_perf(
        self,
        class_encoder: Optional[LabelEncoder] = None,
        pos_class_label_thresh: float = 0.5,
    ) -> Union[pd.DataFrame, pd.DataFrame, Pipeline, list]:
        """Evaluates the best model returned by hyperparameters optimization procedure
        on both training and validation set.
        """

        # Generate class labels for validation set based on decision threshold value (0.5)
        pred_train_probs = self.pipeline.predict_proba(self.train_features)
        pred_train_class = np.where(
            pred_train_probs[:, self.encoded_pos_class_label] > pos_class_label_thresh,
            1,
            0,
        )
        pred_valid_probs = self.pipeline.predict_proba(self.valid_features)
        pred_valid_class = np.where(
            pred_valid_probs[:, self.encoded_pos_class_label] > pos_class_label_thresh,
            1,
            0,
        )

        # Calculate performance metrics on train and validation sets
        train_scores = self.calculate_performance_metrics(
            true_class=self.train_class,
            pred_class=pred_train_class,
        )
        valid_scores = self.calculate_performance_metrics(
            true_class=self.valid_class,
            pred_class=pred_valid_class,
        )

        # Extract original class label names, which can be expressive, i.e., not encoded.
        if class_encoder is None:  # Class labels are already encoded
            original_class_labels = self.pipeline.classes_

            # Extract expressive class names for confusion matrix
            original_train_class = self.train_class
            original_valid_class = self.valid_class
            pred_original_train_class = pred_train_class
            pred_original_valid_class = pred_valid_class

        else:
            original_class_labels = list(
                class_encoder.inverse_transform(self.pipeline.classes_)
            )

            # Extract expressive class names for confusion matrix
            original_train_class = class_encoder.inverse_transform(self.train_class)
            original_valid_class = class_encoder.inverse_transform(self.valid_class)
            pred_original_train_class = class_encoder.inverse_transform(
                pred_train_class
            )
            pred_original_valid_class = class_encoder.inverse_transform(
                pred_valid_class
            )

        train_cm = confusion_matrix(
            y_true=original_train_class,
            y_pred=pred_original_train_class,
            labels=original_class_labels,
            normalize=None,
        )

        self.comet_exp.log_confusion_matrix(
            matrix=train_cm,
            title="Train Set Confusion Matrix",
            file_name="Train Set Confusion Matrix.json",
        )

        train_cm_norm = confusion_matrix(
            y_true=original_train_class,
            y_pred=pred_original_train_class,
            labels=original_class_labels,
            normalize="true",
        )
        self.comet_exp.log_confusion_matrix(
            matrix=train_cm_norm,
            title="Train Set Normalized Confusion Matrix",
            file_name="Train Set Normalized Confusion Matrix.json",
        )

        valid_cm = confusion_matrix(
            y_true=original_valid_class,
            y_pred=pred_original_valid_class,
            labels=original_class_labels,
            normalize=None,
        )
        self.comet_exp.log_confusion_matrix(
            matrix=valid_cm,
            title="Validation Set Confusion Matrix",
            file_name="Validation Set Confusion Matrix.json",
        )

        valid_cm_norm = confusion_matrix(
            y_true=original_valid_class,
            y_pred=pred_original_valid_class,
            labels=original_class_labels,
            normalize="true",
        )
        self.comet_exp.log_confusion_matrix(
            matrix=valid_cm_norm,
            title="Validation Set Normalized Confusion Matrix",
            file_name="Validation Set Normalized Confusion Matrix.json",
        )

        # Plot claibration curve
        calib_curve = CalibrationDisplay.from_predictions(
            self.valid_class,
            pred_valid_probs[:, self.encoded_pos_class_label],
            n_bins=10,
        )

        self.comet_exp.log_figure(
            figure_name="Calibration Curve", figure=calib_curve.figure_, overwrite=True
        )

        # Plot ROC and precision-recall curves
        roc_curve_fig = self.plot_roc_curve(
            y_true=self.valid_class,
            y_pred=pred_valid_probs[:, self.encoded_pos_class_label],
            fig_size=(6, 6),
        )
        self.comet_exp.log_figure(
            figure_name="ROC Curve", figure=roc_curve_fig, overwrite=True
        )

        prec_recall_fig = self.plot_precision_recall_curve(
            y_true=self.valid_class,
            y_pred=pred_valid_probs[:, self.encoded_pos_class_label],
            fig_size=(6, 6),
        )
        self.comet_exp.log_figure(
            figure_name="Precision-Recall Curve", figure=prec_recall_fig, overwrite=True
        )

        # Plot cumulative gain and lift plot
        cum_gain_fig = self.plot_cumulative_gains(
            y_true=self.valid_class, y_pred=pred_valid_probs, fig_size=(6, 6)
        )
        self.comet_exp.log_figure(
            figure_name="Cumulative Gain", figure=cum_gain_fig, overwrite=True
        )

        lift_curve_fig = self.plot_lift_curve(
            y_true=self.valid_class, y_pred=pred_valid_probs, fig_size=(6, 6)
        )
        self.comet_exp.log_figure(
            figure_name="Lift Curve", figure=lift_curve_fig, overwrite=True
        )

        return (train_scores, valid_scores)

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


class PrepChampModel:
    """A class to select the best (champion) model, calibrates it, and
    registers it in workspace."""

    def select_best_performer(
        self,
        comet_project_name: str,
        comet_workspace_name: str,
        comparison_metric: str,
        comet_exp_keys: dict,
    ) -> str:
        """Selects the best performer from all challenger models. The comet_exp_keys
        is a dictionary of model names as keys and their corresponding experiment
        objects as values. It returns the name of the best challenger model."""

        comet_api = API()
        exp_scores = {}
        for i in range(comet_exp_keys.shape[0]):
            experiment = comet_api.get_experiment(
                project_name=comet_project_name,
                workspace=comet_workspace_name,
                experiment=comet_exp_keys.iloc[i, 1],
            )
            exp_metric_score = float(
                experiment.get_metrics(comparison_metric)[0]["metricValue"]
            )
            exp_scores.update(**{f"{comet_exp_keys.iloc[i, 0]}": exp_metric_score})

        # Select the best performer
        best_challenger_name = max(exp_scores, key=exp_scores.get)

        return best_challenger_name

    def calibrate_pipeline(
        self,
        train_features: pd.DataFrame,
        train_class: np.ndarray,
        preprocessor_step: ColumnTransformer,
        selector_step: VarianceThreshold,
        model: Callable,
        cv_folds: int = 5,
    ) -> Pipeline:
        """Calibrates a model within sklearn pipelines. It ."""

        # Fit a pipeline with a calibrated model
        calib_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor_step),
                ("selector", selector_step),
                (
                    "classifier",
                    CalibratedClassifierCV(
                        estimator=model,
                        method="isotonic" if len(train_class) > 1000 else "sigmoid",
                        cv=cv_folds,
                    ),
                ),
            ]
        )

        # Fit pipelines
        calib_pipeline.fit(train_features, train_class)

        return calib_pipeline

    def log_and_register_champ_model(
        self,
        local_path: str,
        champ_model_name: str,
        pipeline: Pipeline,
        exp_obj: ExistingExperiment,
    ) -> None:
        """Logs and registers champion model in workspace. It returns None."""

        if not os.path.exists(local_path):
            os.makedirs(local_path)
        joblib.dump(pipeline, f"{local_path}/{champ_model_name}.pkl")
        exp_obj.log_model(
            name=champ_model_name,
            file_or_folder=f"{local_path}/{champ_model_name}.pkl",
            overwrite=False,
        )
        exp_obj.register_model(model_name=champ_model_name)
        exp_obj.end()
        print(f"Champion model {champ_model_name} was registered in workspace.")
