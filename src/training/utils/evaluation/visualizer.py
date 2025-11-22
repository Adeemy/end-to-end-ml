"""
Model visualization utilities for plotting performance metrics.
"""

import kds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


class ModelVisualizer:
    """Stateless visualizer for model evaluation plots."""

    @staticmethod
    def plot_feature_importance(
        feature_importance_scores: np.ndarray,
        feature_names: list,
        n_top_features: int = 30,
        font_size: int = 10,
        fig_size: tuple = (8, 12),
    ) -> Figure:
        """Plots top feature importance with their names.

        Args:
            feature_importance_scores: Feature importance scores.
            feature_names: List of feature names.
            n_top_features: Number of top features to plot.
            font_size: Font size for labels.
            fig_size: Figure size.

        Returns:
            Matplotlib figure object.
        """
        fig = plt.figure(figsize=fig_size)

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
            plt.tight_layout()

        except ValueError as e:
            plt.text(
                0.5,
                0.5,
                f"Error plotting feature importance: {e}",
                ha="center",
                va="center",
            )

        return fig

    @staticmethod
    def plot_roc_curve(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        fig_size: tuple = (6, 6),
    ) -> Figure:
        """Plots ROC curve for a binary classification model.

        Args:
            y_true: True labels (0 or 1).
            y_pred: Predicted probabilities of the positive class.
            fig_size: Figure size.

        Returns:
            Matplotlib figure object.
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)

        fig = plt.figure(figsize=fig_size)
        plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {auc:.2f})")
        plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Random guess")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()

        return fig

    @staticmethod
    def plot_precision_recall_curve(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        fig_size: tuple = (6, 6),
    ) -> Figure:
        """Plots precision-recall curve for a binary classification model.

        Args:
            y_true: True labels (0 or 1).
            y_pred: Predicted probabilities of the positive class.
            fig_size: Figure size.

        Returns:
            Matplotlib figure object.
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)

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
        plt.tight_layout()

        return fig

    @staticmethod
    def plot_cumulative_gains(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        fig_size: tuple = (6, 6),
    ) -> Figure:
        """Plots cumulative gains curve.

        Args:
            y_true: True labels (0 or 1).
            y_pred: Predicted probabilities of the positive class.
            fig_size: Figure size.

        Returns:
            Matplotlib figure object.
        """
        kds.metrics.plot_cumulative_gain(y_true, y_pred, figsize=fig_size)
        fig = plt.gcf()
        plt.tight_layout()

        return fig

    @staticmethod
    def plot_lift_curve(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        fig_size: tuple = (6, 6),
    ) -> Figure:
        """Plots lift curve.

        Args:
            y_true: True labels (0 or 1).
            y_pred: Predicted probabilities of the positive class.
            fig_size: Figure size.

        Returns:
            Matplotlib figure object.
        """
        kds.metrics.plot_lift(y_true, y_pred, figsize=fig_size)
        fig = plt.gcf()
        plt.tight_layout()

        return fig
