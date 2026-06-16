"""Shared, task-aware performance metrics.

Single source of truth for the metric computations used by the hyperparameter
optimizer (``core/optimizer.py``) and the model evaluators
(``evaluation/evaluator.py``), replacing three near-identical copies of
``calc_perf_metrics``.

Metric *names* are stable contracts. Champion selection
(``orchestrator._score_pipeline_on_valid``) and the deployment gate look up rows
by these exact names, so do not rename existing keys without updating
``selection_metric_row_name`` and the tests in ``tests/test_training``.

Functions return ``List[Tuple[str, float]]`` (name, value) rows so callers can
wrap them in the ``["Metric", "Score"]`` DataFrame the rest of the code expects.
Probability-based metrics (ROC-AUC, PR-AUC, Brier, log loss) are only emitted
when probabilities are supplied AND at least two classes are present in
``y_true``; otherwise they are undefined and skipped rather than raising.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

MetricRows = List[Tuple[str, float]]

# Task types understood across the training/evaluation code.
BINARY = "binary"
MULTICLASS = "multiclass"
REGRESSION = "regression"
CLASSIFICATION_TASKS = frozenset({BINARY, MULTICLASS})
SUPPORTED_TASKS = frozenset({BINARY, MULTICLASS, REGRESSION})

# Metrics for which a *lower* value is better. Used to derive the Optuna study
# direction so a lower-is-better metric is not optimized backwards.
LOWER_IS_BETTER_METRICS: frozenset = frozenset(
    {"log_loss", "brier_score", "brier", "mae", "rmse", "mape"}
)


def fbeta_row_name(fbeta_beta: float) -> str:
    """Returns the metric-row name for an F-beta score (e.g. ``f_0.5_score``)."""
    return f"f_{fbeta_beta}_score"


def _two_classes_present(y_true: ArrayLike) -> bool:
    """True if ``y_true`` contains at least two distinct labels."""
    return len(np.unique(np.asarray(y_true))) >= 2


def binary_classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    fbeta_beta: float = 0.5,
    pos_proba: Optional[ArrayLike] = None,
) -> MetricRows:
    """Computes binary-classification metrics.

    Args:
        y_true: True (encoded) labels.
        y_pred: Hard predicted labels.
        fbeta_beta: Beta for the F-beta score.
        pos_proba: Positive-class probabilities (1-D). Required for the
            probability-based metrics; if omitted they are skipped.

    Returns:
        MetricRows: ``(name, value)`` pairs. Probability-based rows
        (``roc_auc``, ``average_precision``, ``brier_score``, ``log_loss``) are
        included only when ``pos_proba`` is given and two classes are present.
    """
    rows: MetricRows = [
        ("accuracy", accuracy_score(y_true, y_pred)),
        ("precision", precision_score(y_true, y_pred, zero_division=0)),
        ("recall", recall_score(y_true, y_pred, zero_division=0)),
        ("f1", f1_score(y_true, y_pred, zero_division=0)),
        (
            fbeta_row_name(fbeta_beta),
            fbeta_score(y_true, y_pred, beta=fbeta_beta, zero_division=0),
        ),
    ]

    # ROC-AUC, PR-AUC, Brier and log loss must come from probabilities, not hard
    # labels. roc_auc_score(true, hard_labels) silently degenerates into a
    # balanced-accuracy proxy, so only add them when probabilities are provided.
    if pos_proba is not None and _two_classes_present(y_true):
        pos_proba = np.asarray(pos_proba, dtype=float)
        rows.extend(
            [
                ("roc_auc", roc_auc_score(y_true, pos_proba)),
                ("average_precision", average_precision_score(y_true, pos_proba)),
                ("brier_score", brier_score_loss(y_true, pos_proba)),
                ("log_loss", log_loss(y_true, pos_proba, labels=[0, 1])),
            ]
        )

    return rows


def multiclass_classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    fbeta_beta: float = 0.5,
    proba: Optional[ArrayLike] = None,
) -> MetricRows:
    """Computes multi-class metrics with macro/micro/weighted averaging.

    Args:
        y_true: True (encoded) labels.
        y_pred: Hard predicted labels.
        fbeta_beta: Beta for the F-beta score.
        proba: Full ``(n_samples, n_classes)`` probability matrix. Required for
            the probability-based metrics; if omitted they are skipped.

    Returns:
        MetricRows: ``(name, value)`` pairs.
    """
    fbeta = fbeta_row_name(fbeta_beta)
    rows: MetricRows = [
        ("accuracy", accuracy_score(y_true, y_pred)),
        (
            "precision_macro",
            precision_score(y_true, y_pred, average="macro", zero_division=0),
        ),
        (
            "precision_micro",
            precision_score(y_true, y_pred, average="micro", zero_division=0),
        ),
        (
            "precision_weighted",
            precision_score(y_true, y_pred, average="weighted", zero_division=0),
        ),
        (
            "recall_macro",
            recall_score(y_true, y_pred, average="macro", zero_division=0),
        ),
        (
            "recall_micro",
            recall_score(y_true, y_pred, average="micro", zero_division=0),
        ),
        (
            "recall_weighted",
            recall_score(y_true, y_pred, average="weighted", zero_division=0),
        ),
        ("f1_macro", f1_score(y_true, y_pred, average="macro", zero_division=0)),
        ("f1_micro", f1_score(y_true, y_pred, average="micro", zero_division=0)),
        ("f1_weighted", f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        (
            f"{fbeta}_macro",
            fbeta_score(
                y_true, y_pred, beta=fbeta_beta, average="macro", zero_division=0
            ),
        ),
        (
            f"{fbeta}_micro",
            fbeta_score(
                y_true, y_pred, beta=fbeta_beta, average="micro", zero_division=0
            ),
        ),
        (
            f"{fbeta}_weighted",
            fbeta_score(
                y_true, y_pred, beta=fbeta_beta, average="weighted", zero_division=0
            ),
        ),
    ]

    # ROC-AUC (one-vs-rest, macro) and log loss need the full probability matrix.
    if proba is not None and _two_classes_present(y_true):
        proba = np.asarray(proba, dtype=float)
        rows.append(
            (
                "roc_auc_macro",
                roc_auc_score(y_true, proba, multi_class="ovr", average="macro"),
            )
        )
        rows.append(("log_loss", log_loss(y_true, proba)))

    return rows


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> MetricRows:
    """Computes regression metrics: MAE, RMSE, R2 and MAPE.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.

    Returns:
        MetricRows: ``(name, value)`` pairs. MAPE is skipped when any target is
        zero (undefined).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    rows: MetricRows = [
        ("mae", mean_absolute_error(y_true, y_pred)),
        ("rmse", rmse),
        ("r2", r2_score(y_true, y_pred)),
    ]
    if np.all(y_true != 0):
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)))
        rows.append(("mape", mape))
    return rows


def compute_metrics(
    task_type: str,
    y_true: ArrayLike,
    y_pred: ArrayLike,
    proba: Optional[ArrayLike] = None,
    fbeta_beta: float = 0.5,
) -> MetricRows:
    """Dispatches to the metric set for ``task_type``.

    Args:
        task_type: One of ``binary``, ``multiclass``, ``regression``.
        y_true: True labels/targets.
        y_pred: Hard predictions (classification) or predicted values (regression).
        proba: Positive-class probabilities (binary) or the full probability
            matrix (multiclass); ignored for regression.
        fbeta_beta: Beta for the F-beta score (classification only).

    Returns:
        MetricRows: ``(name, value)`` pairs.

    Raises:
        ValueError: If ``task_type`` is not supported.
    """
    if task_type == BINARY:
        return binary_classification_metrics(y_true, y_pred, fbeta_beta, proba)
    if task_type == MULTICLASS:
        return multiclass_classification_metrics(y_true, y_pred, fbeta_beta, proba)
    if task_type == REGRESSION:
        return regression_metrics(y_true, y_pred)
    raise ValueError(
        f"Unsupported task_type: {task_type}. Supported: {sorted(SUPPORTED_TASKS)}"
    )


def rows_to_dataframe(rows: MetricRows) -> pd.DataFrame:
    """Wraps metric rows in the ``['Metric', 'Score']`` DataFrame used elsewhere."""
    return pd.DataFrame(rows, columns=["Metric", "Score"])


def selection_metric_row_name(
    metric: str, task_type: str, fbeta_beta: float = 0.5
) -> str:
    """Resolves a configured metric to its row name for ``task_type``.

    The config exposes task-agnostic names (``fbeta_score``, ``roc_auc``,
    ``average_precision``, ``f1``, ...). This maps them to the row names actually
    produced by :func:`compute_metrics`, applying macro averaging for multi-class
    so champion selection and the deployment gate find the row.

    Args:
        metric: Configured metric name (e.g. ``fbeta_score``, ``roc_auc``).
        task_type: One of ``binary``, ``multiclass``, ``regression``.
        fbeta_beta: Beta used to build the F-beta row name.

    Returns:
        str: The metric name as it appears in the metrics dataframe.
    """
    if task_type == REGRESSION:
        return metric

    if task_type == BINARY:
        return fbeta_row_name(fbeta_beta) if metric == "fbeta_score" else metric

    # Multi-class: apply macro averaging to match the emitted row names.
    if metric == "fbeta_score":
        return f"{fbeta_row_name(fbeta_beta)}_macro"
    if metric == "roc_auc":
        return "roc_auc_macro"
    if metric == "average_precision":
        return "average_precision_macro"
    if metric in ("precision", "recall", "f1"):
        return f"{metric}_macro"
    return metric


def expected_calibration_error(
    confidences: ArrayLike,
    correct: ArrayLike,
    nbins: int = 10,
) -> float:
    """Expected Calibration Error from per-sample confidences and correctness.

    Bins predictions into ``nbins`` equal-width confidence bins and accumulates
    the gap between average confidence and empirical accuracy in each bin,
    weighted by bin population. This is the top-label ECE definition and works
    for both binary (confidence = P(positive)) and multi-class (confidence =
    max class probability) once the caller supplies the matching ``correct`` mask.

    Args:
        confidences: Per-sample confidence in the predicted label, in [0, 1].
        correct: Per-sample boolean/0-1 array, 1 where the prediction is correct.
        nbins: Number of equal-width bins.

    Returns:
        float: ECE in [0, 1].
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    bin_boundaries = np.linspace(0, 1, nbins + 1)
    ece = 0.0
    n = len(confidences)
    if n == 0:
        return 0.0
    for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > lower) & (confidences <= upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            acc_in_bin = correct[in_bin].mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += abs(conf_in_bin - acc_in_bin) * prop_in_bin
    return float(ece)
