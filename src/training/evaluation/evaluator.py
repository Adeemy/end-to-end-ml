"""
Model evaluation utilities. It defines an abstract base class for model evaluation
and a concrete implementation for classification model evaluation.
"""

from abc import ABC, abstractmethod
from pathlib import PosixPath
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.training.evaluation import metrics
from src.training.evaluation.visualizer import ModelVisualizer
from src.training.tracking.experiment_tracker import ExperimentTracker
from src.utils.logger import get_console_logger

module_name: str = PosixPath(__file__).stem
logger = get_console_logger(module_name)


class ModelEvaluator(ABC):
    """Abstract base class for model evaluation.

    This class defines the interface for evaluating machine learning models.
    Concrete implementations should handle specific evaluation strategies
    for binary classification, multi-class classification, etc.

    Attributes:
        tracker (ExperimentTracker): Experiment tracker for logging.
        visualizer (ModelVisualizer): Visualizer for creating plots.
        pipeline (Pipeline): fitted pipeline.
        train_features (pd.DataFrame): train features.
        train_class (np.ndarray): train class labels.
        valid_features (pd.DataFrame): validation features.
        valid_class (np.ndarray): validation class labels.
        fbeta_score_beta (float): beta value for fbeta score.
        encoded_pos_class_label (int): encoded positive class label.
        is_voting_ensemble (bool): whether the model is a voting ensemble or not.
    """

    def __init__(
        self,
        tracker: ExperimentTracker,
        pipeline: Pipeline,
        train_features: pd.DataFrame,
        train_class: np.ndarray,
        valid_features: pd.DataFrame,
        valid_class: np.ndarray,
        fbeta_score_beta: float = 0.5,
        encoded_pos_class_label: int = 1,
        is_voting_ensemble: bool = False,
        visualizer: Optional[ModelVisualizer] = None,
    ) -> None:
        """Creates a ModelEvaluator instance.

        Args:
            tracker: Experiment tracker for logging.
            pipeline: Fitted pipeline.
            train_features: Train features.
            train_class: Train class labels.
            valid_features: Validation features.
            valid_class: Validation class labels.
            fbeta_score_beta: Beta value for fbeta score.
            encoded_pos_class_label: Encoded positive class label.
            is_voting_ensemble: Whether the model is a voting ensemble.
            visualizer: Optional visualizer instance. If None, creates a new one.
        """
        self.tracker = tracker
        self.visualizer = visualizer if visualizer is not None else ModelVisualizer()
        self.pipeline = pipeline
        self.train_features = train_features
        self.train_class = train_class
        self.valid_features = valid_features
        self.valid_class = valid_class
        self.fbeta_score_beta = fbeta_score_beta
        self.encoded_pos_class_label = encoded_pos_class_label
        self.is_voting_ensemble = is_voting_ensemble

    @abstractmethod
    def calc_perf_metrics(
        self,
        true_class: ArrayLike,
        pred_class: ArrayLike,
    ) -> pd.DataFrame:
        """Calculates performance metrics for classification models.

        Args:
            true_class (ArrayLike): true class label.
            pred_class (ArrayLike): predicted class label not probability.

        Returns:
            performance_metrics (pd.DataFrame): a dataframe with metric name and score columns.
        """
        raise NotImplementedError

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

            logger.info(
                "No. of features including encoded categorical features: %d",
                len(col_names),
            )

            # Note: there is no feature_importances_ attribute for LogisticRegression, hence,
            # this if statement is needed to LR coefficients instead.
            classifier_name = pipeline.named_steps["classifier"].__class__.__name__
            feature_importances = (
                None  # To avoid pylint error E0606 (using variable before assignment)
            )
            if classifier_name == "LogisticRegression":
                feature_importances = pipeline.named_steps["classifier"].coef_[0]

            if classifier_name not in [
                "LogisticRegression",
                "VotingClassifier",
            ]:
                feature_importances = pipeline.named_steps[
                    "classifier"
                ].feature_importances_

            if classifier_name != "VotingClassifier":
                self._log_feature_importance_fig(
                    classifier_name=classifier_name,
                    feature_importance_scores=feature_importances,
                    col_names=col_names,
                    n_top_features=n_top_features,
                    figure_size=figure_size,
                    font_size=font_size,
                )

        except Exception as e:  # pylint: disable=W0718
            logger.info("Feature importance extraction error --> %s", e)

    def _log_feature_importance_fig(
        self,
        classifier_name: str,
        feature_importance_scores: np.ndarray,
        col_names: list,
        n_top_features: int = 30,
        figure_size: tuple = (24, 36),
        font_size: float = 10.0,
        fig_name: str = "Feature Importance",
    ) -> None:
        """Plots feature importance figure given feature importance scores and
        column names and logs it to experiment tracker.

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
            feature_importance_fig = self.visualizer.plot_feature_importance(
                feature_importance_scores=feature_importance_scores,
                feature_names=col_names,
                n_top_features=n_top_features,
                font_size=int(font_size),
                fig_size=figure_size,
            )
            self.tracker.log_figure(
                figure_name=fig_name,
                figure=feature_importance_fig,
                overwrite=True,
            )

            logger.info("Feature importance figure was logged to workspace.")

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

        # More robust conversion - handle different DataFrame structures
        if "Metric" not in scores.columns:
            raise ValueError(
                f"DataFrame must have 'Metric' column. Found columns: {scores.columns.tolist()}"
            )

        if scores.shape[1] < 2:
            raise ValueError(
                f"DataFrame must have at least 2 columns. Found shape: {scores.shape}"
            )

        # Set Metric as index and get the first value column
        indexed_scores = scores.set_index("Metric")

        # Use the first column after setting index (should be Score column)
        if indexed_scores.shape[1] == 1:
            # Single column case - get as Series
            metrics_values = indexed_scores.squeeze().to_dict()
        else:
            # Multiple columns case - take first column
            metrics_values = indexed_scores.iloc[:, 0].to_dict()

        # Add prefix to metric names to distinguish train scores from valid scores
        if prefix is not None:
            metrics_values = {prefix + k: v for k, v in metrics_values.items()}

        return metrics_values

    @abstractmethod
    def evaluate_model_perf(
        self,
        class_encoder: Optional[LabelEncoder] = None,
        pos_class_label_thresh: float = 0.5,
    ) -> Union[pd.DataFrame, pd.DataFrame]:
        """Evaluates the best model returned by hyperparameters optimization procedure
        on both training and validation set.

        Args:
            class_encoder (LabelEncoder): class encoder object.
            pos_class_label_thresh (float): decision threshold value for positive class.

        Returns:
            train_scores (pd.DataFrame): training scores.
            valid_scores (pd.DataFrame): validation scores.
        """

        raise NotImplementedError

    @abstractmethod
    def evaluate_test_set_only(
        self,
        class_encoder: LabelEncoder,
        pos_class_label_thresh: float = 0.5,
    ) -> pd.DataFrame:
        """Evaluates model on test set only and logs test-specific artifacts.

        Args:
            class_encoder (LabelEncoder): class encoder object for confusion matrix labels.
            pos_class_label_thresh (float): decision threshold value for positive class.

        Returns:
            test_scores (pd.DataFrame): test set scores.
        """

        raise NotImplementedError

    def _get_pred_class(self, pred_probs: np.ndarray, threshold: float) -> np.ndarray:
        """Returns predicted class labels based on decision threshold value.

        Classification-specific; concrete classification evaluators override it.

        Args:
            pred_probs (np.ndarray): predicted probabilities.
            threshold (float): decision threshold value.

        Returns:
            pred_class (np.ndarray): predicted class labels.
        """

        raise NotImplementedError

    def _safe_extract_pos_probs(
        self, pred_probs: np.ndarray, pos_class_index: int
    ) -> np.ndarray:
        """Safely extract positive class probabilities from predict_proba output.

        Args:
            pred_probs (np.ndarray): predicted probabilities from predict_proba.
            pos_class_index (int): index of positive class.

        Returns:
            np.ndarray: probabilities for positive class.
        """
        if pred_probs.ndim == 1:
            # 1D array case (binary classification edge case)
            return pred_probs
        elif pred_probs.ndim == 2:
            # 2D array case (normal case)
            return pred_probs[:, pos_class_index]
        else:
            raise ValueError(f"Unexpected predict_proba shape: {pred_probs.shape}")

    def _get_original_class_labels(
        self,
        true_class: np.ndarray,
        pred_class: np.ndarray,
        class_encoder: LabelEncoder,
    ) -> tuple:
        """Returns original class labels from encoded class labels.

        Args:
            true_class (np.ndarray): true class labels.
            pred_class (np.ndarray): predicted class labels.
            class_encoder (LabelEncoder): class encoder object.

        Returns:
            tuple: true_original_class, pred_original_class, original_class_labels
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
        original_train_class: np.ndarray,
        pred_original_train_class: np.ndarray,
        original_valid_class: np.ndarray,
        pred_original_valid_class: np.ndarray,
        original_class_labels: list,
        log_train_set: bool = True,
        valid_set_label: str = "Validation Set",
    ) -> None:
        """Logs confusion matrices (normalized and non-normalized) for the best model on
        both the training and validation sets using expressive labels, e.g., Y/N, instead
        of encoded class labels.

        Args:
            original_train_class (np.ndarray): true class labels for training set (expressive labels).
            pred_original_train_class (np.ndarray): predicted class labels for training set (expressive labels).
            original_valid_class (np.ndarray): true class labels for validation set (expressive labels).
            pred_original_valid_class (np.ndarray): predicted class labels for validation set (expressive labels).
            original_class_labels (list): list of expressive class labels.
            log_train_set (bool): whether to log training set confusion matrices (default True).
            valid_set_label (str): label to use for validation/test set confusion matrices (default "Validation Set").
        """

        if log_train_set:
            train_cm = confusion_matrix(
                y_true=original_train_class,
                y_pred=pred_original_train_class,
                labels=original_class_labels,
                normalize=None,
            )
            self.tracker.log_confusion_matrix(
                matrix=train_cm,
                title="Train Set Confusion Matrix",
                labels=original_class_labels,
                file_name="Train Set Confusion Matrix.json",
            )

            train_cm_norm = confusion_matrix(
                y_true=original_train_class,
                y_pred=pred_original_train_class,
                labels=original_class_labels,
                normalize="true",
            )
            self.tracker.log_confusion_matrix(
                matrix=train_cm_norm,
                title="Train Set Normalized Confusion Matrix",
                labels=original_class_labels,
                file_name="Train Set Normalized Confusion Matrix.json",
            )

        valid_cm = confusion_matrix(
            y_true=original_valid_class,
            y_pred=pred_original_valid_class,
            labels=original_class_labels,
            normalize=None,
        )
        self.tracker.log_confusion_matrix(
            matrix=valid_cm,
            title=f"{valid_set_label} Confusion Matrix",
            file_name=f"{valid_set_label} Confusion Matrix.json",
            labels=original_class_labels,
        )

        valid_cm_norm = confusion_matrix(
            y_true=original_valid_class,
            y_pred=pred_original_valid_class,
            labels=original_class_labels,
            normalize="true",
        )
        self.tracker.log_confusion_matrix(
            matrix=valid_cm_norm,
            title=f"{valid_set_label} Normalized Confusion Matrix",
            file_name=f"{valid_set_label} Normalized Confusion Matrix.json",
            labels=original_class_labels,
        )

    # The curve and ECE helpers below are classification-specific. They are kept
    # concrete (not abstract) so non-classification evaluators (e.g. regression)
    # can be instantiated without implementing them; the classification
    # evaluators override them.
    def _log_calibration_curve(self, pred_probs: np.ndarray) -> None:
        """Logs calibration curve for the best model on the validation set."""
        raise NotImplementedError

    def _log_roc_curve(
        self, pred_probs: np.ndarray, encoded_pos_class_label: int
    ) -> None:
        """Logs ROC curve for the best model on the validation set."""
        raise NotImplementedError

    def _log_precision_recall_curve(
        self, pred_probs: np.ndarray, encoded_pos_class_label: int
    ) -> None:
        """Logs precision-recall curve for the best model on the validation set."""
        raise NotImplementedError

    def _log_cumulative_gains(
        self, pred_probs: np.ndarray, valid_class: np.ndarray
    ) -> None:
        """Logs cumulative gains curve for the best model on the validation set."""
        raise NotImplementedError

    def _log_lift_curve(self, pred_probs: np.ndarray, valid_class: np.ndarray) -> None:
        """Logs lift curve for the best model on the validation set."""
        raise NotImplementedError

    def calc_expected_calibration_error(
        self,
        pred_probs: np.ndarray,
        true_labels: np.ndarray,
        decision_thresh_val: float = 0.5,
        nbins: int = 5,
    ) -> float:
        """Calculates Expected Calibration Error (ECE) for a classification model.

        Args:
            pred_probs (np.ndarray): predicted probabilities.
            true_labels (np.ndarray): actual class label (encoded).
            decision_thresh_val (float): decision threshold value.
            nbins (int): number of bins.

        Returns:
            ece (float): ECE value ([0, 1])
        """
        raise NotImplementedError


class BinaryClassificationEvaluator(ModelEvaluator):
    """Concrete implementation of ModelEvaluator for binary classification tasks.

    Provides evaluation metrics specific to binary classification problems.
    """

    def calc_perf_metrics(
        self,
        true_class: ArrayLike,
        pred_class: ArrayLike,
        pred_proba: ArrayLike = None,
    ) -> pd.DataFrame:
        """Calculates performance metrics for binary classification models.

        Args:
            true_class (ArrayLike): true class label.
            pred_class (ArrayLike): predicted class label not probability.
            pred_proba (ArrayLike): positive-class probabilities. Required to compute
                ROC-AUC correctly; if omitted, ROC-AUC is skipped.

        Returns:
            performance_metrics (pd.DataFrame): a dataframe with metric name and score columns.
        """

        rows = metrics.binary_classification_metrics(
            y_true=true_class,
            y_pred=pred_class,
            fbeta_beta=self.fbeta_score_beta,
            pos_proba=pred_proba,
        )
        return metrics.rows_to_dataframe(rows)

    def evaluate_model_perf(
        self,
        class_encoder: Optional[LabelEncoder] = None,
        pos_class_label_thresh: float = 0.5,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Evaluates the best model returned by hyperparameters optimization procedure
        on both training and validation set.

        Args:
            class_encoder (LabelEncoder): class encoder object.
            pos_class_label_thresh (float): decision threshold value for positive class.

        Returns:
            train_scores (pd.DataFrame): training scores.
            valid_scores (pd.DataFrame): validation scores.
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
            pred_proba=self._safe_extract_pos_probs(
                pred_train_probs, self.encoded_pos_class_label
            ),
        )
        valid_scores = self.calc_perf_metrics(
            true_class=self.valid_class,
            pred_class=pred_valid_class,
            pred_proba=self._safe_extract_pos_probs(
                pred_valid_probs, self.encoded_pos_class_label
            ),
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
                original_train_class=original_train_class,
                pred_original_train_class=pred_original_train_class,
                original_valid_class=original_valid_class,
                pred_original_valid_class=pred_original_valid_class,
                original_class_labels=original_class_labels,
            )

            logger.info("Confusion matrices were logged to workspace.")

        # Log calibration curve and plots
        self._log_calibration_curve(pred_valid_probs)
        logger.info("Calibration curve was logged to workspace.")

        self._log_roc_curve(pred_valid_probs, self.encoded_pos_class_label)
        logger.info("ROC curve was logged to workspace.")

        self._log_precision_recall_curve(pred_valid_probs, self.encoded_pos_class_label)
        logger.info("Precision-Recall curve was logged to workspace.")

        self._log_cumulative_gains(pred_valid_probs, self.valid_class)
        logger.info("Cumulative gains curve was logged to workspace.")

        self._log_lift_curve(pred_valid_probs, self.valid_class)
        logger.info("Lift curve was logged to workspace.")

        return train_scores, valid_scores

    def _get_pred_class(self, pred_probs: np.ndarray, threshold: float) -> np.ndarray:
        """Returns predicted class labels based on decision threshold value.

        Args:
            pred_probs (np.ndarray): predicted probabilities of the positive class.
            threshold (float): decision threshold value for positive class.

        Returns:
            pred_class (np.ndarray): predicted class labels.
        """

        # Use helper method to safely extract positive class probabilities
        pos_probs = self._safe_extract_pos_probs(
            pred_probs, self.encoded_pos_class_label
        )

        # Emit the model's own class labels rather than literal 1/0, so the
        # result is correct even if the encoded positive label is not 1 or the
        # classes are not {0, 1}.
        classes = list(self.pipeline.classes_)
        pos_label = (
            self.encoded_pos_class_label
            if self.encoded_pos_class_label in classes
            else classes[-1]
        )
        neg_label = next(c for c in classes if c != pos_label)
        # Use >= to match the serving decision rule (positive_class_predictions)
        # and threshold tuning, so reported metrics reflect the deployed model.
        pred_class = np.where(pos_probs >= threshold, pos_label, neg_label)

        return pred_class

    def _log_calibration_curve(self, pred_probs: np.ndarray) -> None:
        """Logs calibration curve for the best model on the validation set.

        Args:
            pred_probs (np.ndarray): predicted probabilities of the positive class
            of the best model on the validation set.
        """

        pos_probs = self._safe_extract_pos_probs(
            pred_probs, self.encoded_pos_class_label
        )
        calib_curve = CalibrationDisplay.from_predictions(
            self.valid_class,
            pos_probs,
            n_bins=10,
        )
        self.tracker.log_figure(
            figure_name="Calibration Curve", figure=calib_curve.figure_, overwrite=True
        )

    def _log_roc_curve(
        self, pred_probs: np.ndarray, encoded_pos_class_label: int
    ) -> None:
        """Logs ROC curve for the best model on the validation set.

        Args:
            pred_probs (np.ndarray): predicted probabilities of the positive class.
            encoded_pos_class_label (int): encoded positive class label.
        """

        pos_probs = self._safe_extract_pos_probs(pred_probs, encoded_pos_class_label)
        roc_curve_fig = self.visualizer.plot_roc_curve(
            y_true=self.valid_class,
            y_pred=pos_probs,
            fig_size=(6, 6),
        )
        self.tracker.log_figure(
            figure_name="ROC Curve", figure=roc_curve_fig, overwrite=True
        )

    def _log_precision_recall_curve(
        self, pred_probs: np.ndarray, encoded_pos_class_label: int
    ) -> None:
        """Logs precision-recall curve for the best model on the validation set.

        Args:
            pred_probs (np.ndarray): predicted probabilities of the positive class.
            encoded_pos_class_label (int): encoded positive class label.
        """

        pos_probs = self._safe_extract_pos_probs(pred_probs, encoded_pos_class_label)
        prec_recall_fig = self.visualizer.plot_precision_recall_curve(
            y_true=self.valid_class,
            y_pred=pos_probs,
            fig_size=(6, 6),
        )
        self.tracker.log_figure(
            figure_name="Precision-Recall Curve", figure=prec_recall_fig, overwrite=True
        )

    def _log_cumulative_gains(
        self, pred_probs: np.ndarray, valid_class: np.ndarray
    ) -> None:
        """Logs cumulative gains curve for the best model on the validation set.

        Args:
            pred_probs (1-D np.ndarray): predicted probabilities of the positive class.
            valid_class (np.ndarray): validation class labels.
        """

        pos_probs = self._safe_extract_pos_probs(
            pred_probs, self.encoded_pos_class_label
        )
        cum_gain_fig = self.visualizer.plot_cumulative_gains(
            y_true=valid_class,
            y_pred=pos_probs,
            fig_size=(6, 6),
        )
        self.tracker.log_figure(
            figure_name="Cumulative Gain", figure=cum_gain_fig, overwrite=True
        )

    def _log_lift_curve(self, pred_probs: np.ndarray, valid_class: np.ndarray) -> None:
        """Logs lift curve for the best model on the validation set.

        Args:
            pred_probs (1-D np.ndarray): predicted probabilities of the positive class.
            valid_class (np.ndarray): validation class labels.
        """

        pos_probs = self._safe_extract_pos_probs(
            pred_probs, self.encoded_pos_class_label
        )
        lift_curve_fig = self.visualizer.plot_lift_curve(
            y_true=valid_class,
            y_pred=pos_probs,
            fig_size=(6, 6),
        )
        self.tracker.log_figure(
            figure_name="Lift Curve", figure=lift_curve_fig, overwrite=True
        )

    def calc_expected_calibration_error(
        self,
        pred_probs: np.ndarray,
        true_labels: np.ndarray,
        decision_thresh_val: float = 0.5,
        nbins: int = 5,
    ) -> float:
        """Calculates Expected Calibration Error (ECE) for binary classification.

        Args:
            pred_probs (np.ndarray): predicted probability of positive class.
            true_labels (np.ndarray): actual class label (encoded).
            decision_thresh_val (float): probabilities greater than these value will
                                            be considered positive and negative otherwise.
            nbins (int): number of bins (bins values between 0 and 1 into nbins equal
                         sized bins).

        Returns:
            ece (float): ECE value ([0, 1])
        """

        # Confidence is the positive-class probability; a prediction is "correct"
        # when the threshold decision matches the true label. Binning/gap logic
        # is shared with the multi-class evaluator via metrics.expected_calibration_error.
        confidences = self._safe_extract_pos_probs(
            pred_probs, self.encoded_pos_class_label
        )
        pred_label = (confidences > decision_thresh_val).astype(float)
        correct = (pred_label == np.asarray(true_labels)).astype(float)
        return metrics.expected_calibration_error(
            confidences=confidences, correct=correct, nbins=nbins
        )

    def evaluate_test_set_only(
        self,
        class_encoder: LabelEncoder,
        pos_class_label_thresh: float = 0.5,
    ) -> pd.DataFrame:
        """Evaluates model on test set only and logs test-specific artifacts.

        Args:
            class_encoder (LabelEncoder): class encoder object for confusion matrix labels.
            pos_class_label_thresh (float): decision threshold value for positive class.
                For binary models this is the operating threshold to apply (so the
                reported test metrics reflect the deployed operating point).

        Returns:
            test_scores (pd.DataFrame): test set scores.
        """

        # Generate predictions for test set (valid_features/valid_class are actually test data)
        pred_test_probs = self.pipeline.predict_proba(self.valid_features)
        pred_test_class = self._get_pred_class(pred_test_probs, pos_class_label_thresh)

        # Generate predictions for train set (needed for confusion matrix comparison)
        pred_train_probs = self.pipeline.predict_proba(self.train_features)
        pred_train_class = self._get_pred_class(
            pred_train_probs, pos_class_label_thresh
        )

        # Calculate test scores
        test_scores = self.calc_perf_metrics(
            true_class=self.valid_class,  # This is actually test data
            pred_class=pred_test_class,
            pred_proba=self._safe_extract_pos_probs(
                pred_test_probs, self.encoded_pos_class_label
            ),
        )

        # Get original class labels for confusion matrix
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
            original_test_class,
            pred_original_test_class,
            _,
        ) = self._get_original_class_labels(
            true_class=self.valid_class,  # This is actually test data
            pred_class=pred_test_class,
            class_encoder=class_encoder,
        )

        # Log only test set confusion matrices
        self._log_confusion_matrix(
            original_train_class=original_train_class,
            pred_original_train_class=pred_original_train_class,
            original_valid_class=original_test_class,
            pred_original_valid_class=pred_original_test_class,
            original_class_labels=original_class_labels,
            log_train_set=False,  # Skip training set confusion matrices
            valid_set_label="Test Set",  # Use "Test Set" label
        )
        logger.info("Test set confusion matrices were logged to workspace.")

        # Log evaluation curves using test data
        self._log_calibration_curve(pred_test_probs)
        logger.info("Calibration curve was logged to workspace.")

        self._log_roc_curve(pred_test_probs, self.encoded_pos_class_label)
        logger.info("ROC curve was logged to workspace.")

        self._log_precision_recall_curve(pred_test_probs, self.encoded_pos_class_label)
        logger.info("Precision-Recall curve was logged to workspace.")

        self._log_cumulative_gains(
            pred_test_probs, self.valid_class
        )  # This is actually test data
        logger.info("Cumulative gains curve was logged to workspace.")

        self._log_lift_curve(
            pred_test_probs, self.valid_class
        )  # This is actually test data
        logger.info("Lift curve was logged to workspace.")

        return test_scores


class MultiClassificationEvaluator(ModelEvaluator):
    """Concrete implementation of ModelEvaluator for multi-class classification tasks.

    Provides evaluation metrics specific to multi-class classification problems.
    """

    def calc_perf_metrics(
        self,
        true_class: ArrayLike,
        pred_class: ArrayLike,
        pred_proba: ArrayLike = None,
    ) -> pd.DataFrame:
        """Calculates performance metrics for multi-class classification models.

        Args:
            true_class (ArrayLike): true class label.
            pred_class (ArrayLike): predicted class label not probability.
            pred_proba (ArrayLike): full predict_proba matrix (n_samples, n_classes).
                Required to compute ROC-AUC; if omitted, ROC-AUC is skipped.

        Returns:
            performance_metrics (pd.DataFrame): a dataframe with metric name and score columns.
        """

        rows = metrics.multiclass_classification_metrics(
            y_true=true_class,
            y_pred=pred_class,
            fbeta_beta=self.fbeta_score_beta,
            proba=pred_proba,
        )
        return metrics.rows_to_dataframe(rows)

    def evaluate_model_perf(
        self,
        class_encoder: Optional[LabelEncoder] = None,
        pos_class_label_thresh: float = 0.5,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Evaluates the best model returned by hyperparameters optimization procedure
        on both training and validation set for multi-class classification.

        Args:
            class_encoder (LabelEncoder): class encoder object.
            pos_class_label_thresh (float): decision threshold value (not used in multi-class).

        Returns:
            train_scores (pd.DataFrame): training scores.
            valid_scores (pd.DataFrame): validation scores.
        """

        # For multi-class, use argmax to get predicted classes
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
            pred_proba=pred_train_probs,
        )
        valid_scores = self.calc_perf_metrics(
            true_class=self.valid_class,
            pred_class=pred_valid_class,
            pred_proba=pred_valid_probs,
        )

        # Extract original class label names
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
                original_train_class=original_train_class,
                pred_original_train_class=pred_original_train_class,
                original_valid_class=original_valid_class,
                pred_original_valid_class=pred_original_valid_class,
                original_class_labels=original_class_labels,
            )

            logger.info("Confusion matrices were logged to workspace.")

        # Log calibration curve and plots (multi-class specific)
        self._log_calibration_curve(pred_valid_probs)
        logger.info("Calibration curve was logged to workspace.")

        # For multi-class, we'll skip ROC and PR curves that are binary-specific
        # or implement them differently
        try:
            self._log_roc_curve(pred_valid_probs, self.encoded_pos_class_label)
            logger.info("ROC curve was logged to workspace.")
        except Exception as e:  # pylint: disable=W0718
            logger.info("ROC curve logging skipped for multi-class: %s", e)

        try:
            self._log_precision_recall_curve(
                pred_valid_probs, self.encoded_pos_class_label
            )
            logger.info("Precision-Recall curve was logged to workspace.")
        except Exception as e:  # pylint: disable=W0718
            logger.info("Precision-Recall curve logging skipped for multi-class: %s", e)

        try:
            self._log_cumulative_gains(pred_valid_probs, self.valid_class)
            logger.info("Cumulative gains curve was logged to workspace.")
        except Exception as e:  # pylint: disable=W0718
            logger.info("Cumulative gains curve logging skipped for multi-class: %s", e)

        try:
            self._log_lift_curve(pred_valid_probs, self.valid_class)
            logger.info("Lift curve was logged to workspace.")
        except Exception as e:  # pylint: disable=W0718
            logger.info("Lift curve logging skipped for multi-class: %s", e)

        return train_scores, valid_scores

    def _get_pred_class(self, pred_probs: np.ndarray, threshold: float) -> np.ndarray:
        """Returns predicted class labels using argmax for multi-class classification.

        Args:
            pred_probs (np.ndarray): predicted probabilities for all classes.
            threshold (float): decision threshold value (not used in multi-class).

        Returns:
            pred_class (np.ndarray): predicted class labels.
        """
        # For multi-class, use argmax to get the class with highest probability
        pred_class = np.argmax(pred_probs, axis=1)
        return pred_class

    def _log_calibration_curve(self, pred_probs: np.ndarray) -> None:
        """Logs calibration curve for multi-class classification.

        Args:
            pred_probs (np.ndarray): predicted probabilities for all classes.
        """
        # For multi-class, we'll use the maximum probability across all classes
        max_probs = np.max(pred_probs, axis=1)
        pred_classes = np.argmax(pred_probs, axis=1)

        # Check if predictions match true classes for calibration
        correct_predictions = (pred_classes == self.valid_class).astype(int)

        try:
            calib_curve = CalibrationDisplay.from_predictions(
                correct_predictions,
                max_probs,
                n_bins=10,
            )
            self.tracker.log_figure(
                figure_name="Calibration Curve (Multi-class)",
                figure=calib_curve.figure_,
                overwrite=True,
            )
        except Exception as e:  # pylint: disable=W0718
            logger.info("Multi-class calibration curve could not be created: %s", e)

    def _log_roc_curve(
        self, pred_probs: np.ndarray, encoded_pos_class_label: int
    ) -> None:
        """Logs ROC curve for multi-class classification (simplified version).

        Args:
            pred_probs (np.ndarray): predicted probabilities for all classes.
            encoded_pos_class_label (int): encoded positive class label (not used in multi-class).
        """
        # For multi-class, ROC curves are more complex
        # This is a simplified implementation
        logger.info(
            "Multi-class ROC curve visualization not implemented in this version."
        )

    def _log_precision_recall_curve(
        self, pred_probs: np.ndarray, encoded_pos_class_label: int
    ) -> None:
        """Logs precision-recall curve for multi-class classification (simplified version).

        Args:
            pred_probs (np.ndarray): predicted probabilities for all classes.
            encoded_pos_class_label (int): encoded positive class label (not used in multi-class).
        """
        # For multi-class, PR curves are more complex
        # This is a simplified implementation
        logger.info(
            "Multi-class Precision-Recall curve visualization not implemented in this version."
        )

    def _log_cumulative_gains(
        self, pred_probs: np.ndarray, valid_class: np.ndarray
    ) -> None:
        """Logs cumulative gains curve for multi-class classification (simplified version).

        Args:
            pred_probs (np.ndarray): predicted probabilities for all classes.
            valid_class (np.ndarray): validation class labels.
        """
        # For multi-class, cumulative gains are more complex
        logger.info(
            "Multi-class cumulative gains curve visualization not implemented in this version."
        )

    def _log_lift_curve(self, pred_probs: np.ndarray, valid_class: np.ndarray) -> None:
        """Logs lift curve for multi-class classification (simplified version).

        Args:
            pred_probs (np.ndarray): predicted probabilities for all classes.
            valid_class (np.ndarray): validation class labels.
        """
        # For multi-class, lift curves are more complex
        logger.info(
            "Multi-class lift curve visualization not implemented in this version."
        )

    def calc_expected_calibration_error(
        self,
        pred_probs: np.ndarray,
        true_labels: np.ndarray,
        decision_thresh_val: float = 0.5,
        nbins: int = 5,
    ) -> float:
        """Calculates Expected Calibration Error (ECE) for multi-class classification.

        Args:
            pred_probs (np.ndarray): predicted probabilities for all classes.
            true_labels (np.ndarray): actual class label (encoded).
            decision_thresh_val (float): decision threshold value (not used in multi-class).
            nbins (int): number of bins.

        Returns:
            ece (float): ECE value ([0, 1])
        """

        # Top-label ECE: confidence is the max class probability and a prediction
        # is correct when the argmax class matches the true label. Binning is
        # shared with the binary path via metrics.expected_calibration_error.
        confidences = np.max(pred_probs, axis=1)
        pred_label = np.argmax(pred_probs, axis=1)
        correct = (pred_label == np.asarray(true_labels)).astype(float)
        return metrics.expected_calibration_error(
            confidences=confidences, correct=correct, nbins=nbins
        )

    def evaluate_test_set_only(
        self,
        class_encoder: LabelEncoder,
        pos_class_label_thresh: float = 0.5,
    ) -> pd.DataFrame:
        """Evaluates model on test set only and logs test-specific artifacts.

        Args:
            class_encoder (LabelEncoder): class encoder object for confusion matrix labels.
            pos_class_label_thresh (float): decision threshold value for positive class.

        Returns:
            test_scores (pd.DataFrame): test set scores.
        """

        # Generate predictions for test set (valid_features/valid_class are actually test data)
        pred_test_probs = self.pipeline.predict_proba(self.valid_features)
        pred_test_class = self._get_pred_class(pred_test_probs, pos_class_label_thresh)

        # Generate predictions for train set (needed for confusion matrix comparison)
        pred_train_probs = self.pipeline.predict_proba(self.train_features)
        pred_train_class = self._get_pred_class(
            pred_train_probs, pos_class_label_thresh
        )

        # Calculate test scores
        test_scores = self.calc_perf_metrics(
            true_class=self.valid_class,  # This is actually test data
            pred_class=pred_test_class,
            pred_proba=pred_test_probs,
        )

        # Get original class labels for confusion matrix
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
            original_test_class,
            pred_original_test_class,
            _,
        ) = self._get_original_class_labels(
            true_class=self.valid_class,  # This is actually test data
            pred_class=pred_test_class,
            class_encoder=class_encoder,
        )

        # Log only test set confusion matrices
        self._log_confusion_matrix(
            original_train_class=original_train_class,
            pred_original_train_class=pred_original_train_class,
            original_valid_class=original_test_class,
            pred_original_valid_class=pred_original_test_class,
            original_class_labels=original_class_labels,
            log_train_set=False,  # Skip training set confusion matrices
            valid_set_label="Test Set",  # Use "Test Set" label
        )
        logger.info("Test set confusion matrices were logged to workspace.")

        # Log evaluation curves using test data (multi-class specific)
        self._log_calibration_curve(pred_test_probs)
        logger.info("Calibration curve was logged to workspace.")

        self._log_roc_curve(pred_test_probs, self.encoded_pos_class_label)
        logger.info("ROC curve was logged to workspace.")

        self._log_precision_recall_curve(pred_test_probs, self.encoded_pos_class_label)
        logger.info("Precision-Recall curve was logged to workspace.")

        self._log_cumulative_gains(
            pred_test_probs, self.valid_class
        )  # This is actually test data
        logger.info("Cumulative gains curve was logged to workspace.")

        self._log_lift_curve(
            pred_test_probs, self.valid_class
        )  # This is actually test data
        logger.info("Lift curve was logged to workspace.")

        return test_scores


class RegressionEvaluator(ModelEvaluator):
    """Concrete ModelEvaluator for regression tasks.

    Provides regression metrics (MAE, RMSE, R2, MAPE) and a predicted-vs-actual
    plot. Classification-only machinery (probabilities, thresholds, calibration,
    label encoding, ROC/PR/lift curves) does not apply and is not implemented.

    Note: regression is supported here to keep the evaluator layer task-agnostic.
    The end-to-end regression training/serving path is intentionally left as a
    documented extension point (see docs/enhancement-plan.md).
    """

    def calc_perf_metrics(  # pylint: disable=unused-argument
        self,
        true_class: ArrayLike,
        pred_class: ArrayLike,
        pred_proba: ArrayLike = None,  # unused; kept for interface symmetry
    ) -> pd.DataFrame:
        """Computes regression metrics (``true_class``/``pred_class`` are values).

        Args:
            true_class (ArrayLike): true target values.
            pred_class (ArrayLike): predicted target values.
            pred_proba (ArrayLike): ignored for regression.

        Returns:
            performance_metrics (pd.DataFrame): metric name/score dataframe.
        """
        rows = metrics.regression_metrics(y_true=true_class, y_pred=pred_class)
        return metrics.rows_to_dataframe(rows)

    def _log_predicted_vs_actual(
        self, y_true: np.ndarray, y_pred: np.ndarray, set_label: str
    ) -> None:
        """Logs a predicted-vs-actual scatter to the tracker."""
        try:
            import matplotlib.pyplot as plt  # local import: optional plotting dep

            fig, axis = plt.subplots(figsize=(6, 6))
            axis.scatter(y_true, y_pred, s=8, alpha=0.5)
            lo = float(min(np.min(y_true), np.min(y_pred)))
            hi = float(max(np.max(y_true), np.max(y_pred)))
            axis.plot([lo, hi], [lo, hi], "r--", linewidth=1)
            axis.set_xlabel("Actual")
            axis.set_ylabel("Predicted")
            axis.set_title(f"{set_label}: Predicted vs Actual")
            self.tracker.log_figure(
                figure_name=f"{set_label} Predicted vs Actual",
                figure=fig,
                overwrite=True,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.info("Predicted-vs-actual plot skipped: %s", exc)

    def evaluate_model_perf(
        self,
        class_encoder: Optional[LabelEncoder] = None,
        pos_class_label_thresh: float = 0.5,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Evaluates the regressor on the training and validation sets.

        Args:
            class_encoder (LabelEncoder): unused for regression.
            pos_class_label_thresh (float): unused for regression.

        Returns:
            tuple: (train_scores, valid_scores).
        """
        pred_train = self.pipeline.predict(self.train_features)
        pred_valid = self.pipeline.predict(self.valid_features)
        train_scores = self.calc_perf_metrics(self.train_class, pred_train)
        valid_scores = self.calc_perf_metrics(self.valid_class, pred_valid)
        self._log_predicted_vs_actual(
            np.asarray(self.valid_class), np.asarray(pred_valid), "Validation Set"
        )
        return train_scores, valid_scores

    def evaluate_test_set_only(
        self,
        class_encoder: LabelEncoder,
        pos_class_label_thresh: float = 0.5,
    ) -> pd.DataFrame:
        """Evaluates the regressor on the held-out test set.

        Note: as elsewhere in this module, ``valid_features``/``valid_class`` hold
        the test data in the test-evaluation orchestrator.

        Args:
            class_encoder (LabelEncoder): unused for regression.
            pos_class_label_thresh (float): unused for regression.

        Returns:
            test_scores (pd.DataFrame): test set scores.
        """
        pred_test = self.pipeline.predict(self.valid_features)
        test_scores = self.calc_perf_metrics(self.valid_class, pred_test)
        self._log_predicted_vs_actual(
            np.asarray(self.valid_class), np.asarray(pred_test), "Test Set"
        )
        return test_scores


def create_model_evaluator(
    tracker: ExperimentTracker,
    pipeline: Pipeline,
    train_features: pd.DataFrame,
    train_class: np.ndarray,
    valid_features: pd.DataFrame,
    valid_class: np.ndarray,
    fbeta_score_beta: float = 0.5,
    encoded_pos_class_label: int = 1,
    is_voting_ensemble: bool = False,
    visualizer: Optional[ModelVisualizer] = None,
    task_type: Optional[str] = None,
) -> ModelEvaluator:
    """Factory function to create the appropriate ModelEvaluator implementation.

    Dispatch is driven by the explicit ``task_type`` when provided
    (``binary``/``multiclass``/``regression``). When ``task_type`` is None it
    falls back to inferring binary vs multi-class from the number of unique
    classes (backward-compatible), and cross-checks an explicit classification
    ``task_type`` against the observed class count.

    Args:
        tracker: Experiment tracker for logging.
        pipeline: Fitted pipeline.
        train_features: Train features.
        train_class: Train class labels (or target values for regression).
        valid_features: Validation features.
        valid_class: Validation class labels (or target values for regression).
        fbeta_score_beta: Beta value for fbeta score.
        encoded_pos_class_label: Encoded positive class label.
        is_voting_ensemble: Whether the model is a voting ensemble.
        visualizer: Optional visualizer instance. If None, creates a new one.
        task_type: Explicit task type; overrides the class-count heuristic.

    Returns:
        ModelEvaluator: Appropriate concrete implementation for the task.

    Raises:
        ValueError: If task_type is provided but not supported.
    """
    common_kwargs = {
        "tracker": tracker,
        "pipeline": pipeline,
        "train_features": train_features,
        "train_class": train_class,
        "valid_features": valid_features,
        "valid_class": valid_class,
        "fbeta_score_beta": fbeta_score_beta,
        "encoded_pos_class_label": encoded_pos_class_label,
        "is_voting_ensemble": is_voting_ensemble,
        "visualizer": visualizer,
    }

    if task_type == metrics.REGRESSION:
        return RegressionEvaluator(**common_kwargs)

    # Classification: trust an explicit task_type, else infer from class count.
    n_classes = len(np.unique(np.concatenate([train_class, valid_class])))
    if task_type == metrics.MULTICLASS:
        is_binary = False
    elif task_type == metrics.BINARY:
        is_binary = True
        if n_classes > 2:
            logger.warning(
                "task_type='binary' but %d classes observed; using the "
                "multi-class evaluator to avoid incorrect metrics.",
                n_classes,
            )
            is_binary = False
    elif task_type is None:
        is_binary = n_classes <= 2
    else:
        raise ValueError(
            f"Unsupported task_type: {task_type}. "
            f"Supported: {sorted(metrics.SUPPORTED_TASKS)}"
        )

    if is_binary:
        return BinaryClassificationEvaluator(**common_kwargs)
    return MultiClassificationEvaluator(**common_kwargs)
