"""
Tests for the shared, task-aware metrics module (src/training/evaluation/metrics.py).

These lock the metric-name contracts that champion selection and the deployment
gate depend on, the probability-gating of ROC-AUC/PR-AUC/Brier/log-loss, the
regression metric set, the selection-metric resolver, and the ECE helper.
"""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.training.evaluation import metrics


@pytest.fixture(name="binary_arrays")
def binary_arrays_fixture():
    """Balanced binary labels with separable positive-class probabilities."""
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 50 + [1] * 50)
    pos_proba = np.concatenate([rng.uniform(0, 0.5, 50), rng.uniform(0.5, 1.0, 50)])
    y_pred = (pos_proba >= 0.5).astype(int)
    return y_true, y_pred, pos_proba


def test_binary_metrics_gate_probability_metrics(binary_arrays):
    """ROC-AUC/PR-AUC/Brier/log-loss appear only when probabilities are given."""
    y_true, y_pred, pos_proba = binary_arrays

    without = dict(metrics.binary_classification_metrics(y_true, y_pred, 0.5))
    for prob_metric in ("roc_auc", "average_precision", "brier_score", "log_loss"):
        assert prob_metric not in without
    # Threshold-based metrics are always present.
    for base in ("accuracy", "precision", "recall", "f1", "f_0.5_score"):
        assert base in without

    with_proba = dict(
        metrics.binary_classification_metrics(y_true, y_pred, 0.5, pos_proba)
    )
    assert with_proba["roc_auc"] == pytest.approx(roc_auc_score(y_true, pos_proba))
    assert with_proba["average_precision"] == pytest.approx(
        average_precision_score(y_true, pos_proba)
    )
    assert "brier_score" in with_proba and "log_loss" in with_proba


def test_binary_metrics_skip_probs_when_single_class():
    """Probability metrics are skipped when y_true has a single class (undefined)."""
    y_true = np.zeros(10, dtype=int)
    y_pred = np.zeros(10, dtype=int)
    proba = np.linspace(0, 1, 10)
    result = dict(metrics.binary_classification_metrics(y_true, y_pred, 0.5, proba))
    assert "roc_auc" not in result


def test_multiclass_metric_names_and_auc():
    """Multi-class emits macro/micro/weighted names + roc_auc_macro + log_loss."""
    rng = np.random.default_rng(1)
    y_true = np.array([0] * 20 + [1] * 20 + [2] * 20)
    proba = rng.dirichlet(np.ones(3), size=60)
    # Make the proba weakly informative of the true class.
    for i, label in enumerate(y_true):
        proba[i, label] += 1.0
    proba = proba / proba.sum(axis=1, keepdims=True)
    y_pred = np.argmax(proba, axis=1)

    result = dict(metrics.multiclass_classification_metrics(y_true, y_pred, 0.5, proba))
    for name in (
        "precision_macro",
        "precision_micro",
        "precision_weighted",
        "f1_macro",
        "f_0.5_score_macro",
        "roc_auc_macro",
        "log_loss",
    ):
        assert name in result
    assert result["roc_auc_macro"] == pytest.approx(
        roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
    )


def test_regression_metrics_and_mape_guard():
    """Regression emits MAE/RMSE/R2; MAPE is skipped when a target is zero."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    result = dict(metrics.regression_metrics(y_true, y_pred))
    assert {"mae", "rmse", "r2", "mape"}.issubset(result)
    assert result["rmse"] >= result["mae"]  # RMSE >= MAE always

    with_zero = dict(
        metrics.regression_metrics(np.array([0.0, 2.0]), np.array([0.1, 1.9]))
    )
    assert "mape" not in with_zero  # undefined when a true value is zero


def test_selection_metric_row_name_by_task():
    """The resolver maps config metric names to the emitted row names per task."""
    assert (
        metrics.selection_metric_row_name("fbeta_score", "binary", 0.5) == "f_0.5_score"
    )
    assert metrics.selection_metric_row_name("roc_auc", "binary", 0.5) == "roc_auc"
    assert (
        metrics.selection_metric_row_name("fbeta_score", "multiclass", 0.5)
        == "f_0.5_score_macro"
    )
    assert (
        metrics.selection_metric_row_name("roc_auc", "multiclass", 0.5)
        == "roc_auc_macro"
    )
    assert metrics.selection_metric_row_name("f1", "multiclass", 0.5) == "f1_macro"
    assert metrics.selection_metric_row_name("rmse", "regression", 0.5) == "rmse"


def test_compute_metrics_dispatch_and_validation():
    """compute_metrics dispatches by task and rejects unknown task types."""
    y = np.array([0, 1, 0, 1])
    assert (
        dict(metrics.compute_metrics("binary", y, y, fbeta_beta=0.5))["accuracy"] == 1.0
    )
    assert "rmse" in dict(metrics.compute_metrics("regression", [1.0, 2.0], [1.0, 2.0]))
    with pytest.raises(ValueError):
        metrics.compute_metrics("ranking", y, y)


def test_expected_calibration_error_bounds():
    """ECE is 0 for a perfectly calibrated/confident correct set and in [0, 1]."""
    confidences = np.array([1.0, 1.0, 1.0, 1.0])
    correct = np.array([1.0, 1.0, 1.0, 1.0])
    assert metrics.expected_calibration_error(
        confidences, correct, nbins=5
    ) == pytest.approx(0.0)

    # Confident but always wrong -> ECE close to 1.
    wrong = np.zeros(4)
    ece = metrics.expected_calibration_error(confidences, wrong, nbins=5)
    assert 0.0 <= ece <= 1.0 and ece == pytest.approx(1.0)
