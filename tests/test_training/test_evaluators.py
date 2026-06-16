"""
Coverage for the concrete evaluators' end-to-end perf methods.

Exercises evaluate_model_perf (binary/multi-class/regression), the label-aware
binary decision rule, and the shared ECE helper through the evaluator API, all
against a MagicMock tracker so no experiment backend is required.
"""

# pylint: disable=protected-access

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.training.evaluation.evaluator import (
    BinaryClassificationEvaluator,
    MultiClassificationEvaluator,
    RegressionEvaluator,
)


def _binary(n=120, seed=0, sep=1.5):
    rng = np.random.default_rng(seed)
    half = n // 2
    features = pd.DataFrame(
        {
            "f1": np.concatenate(
                [rng.normal(0, 1, half), rng.normal(sep, 1, n - half)]
            ),
            "f2": np.concatenate(
                [rng.normal(0, 1, half), rng.normal(sep, 1, n - half)]
            ),
        }
    )
    labels = np.array([0] * half + [1] * (n - half))
    return features, labels


def _multiclass(n=150, seed=1, sep=2.5):
    rng = np.random.default_rng(seed)
    third = n // 3
    features = pd.DataFrame(
        {
            "f1": np.concatenate(
                [
                    rng.normal(0, 1, third),
                    rng.normal(sep, 1, third),
                    rng.normal(2 * sep, 1, n - 2 * third),
                ]
            ),
            "f2": np.concatenate(
                [
                    rng.normal(0, 1, third),
                    rng.normal(sep, 1, third),
                    rng.normal(2 * sep, 1, n - 2 * third),
                ]
            ),
        }
    )
    labels = np.array([0] * third + [1] * third + [2] * (n - 2 * third))
    return features, labels


def _clf(features, labels):
    return Pipeline(
        steps=[
            ("preprocessor", StandardScaler()),
            ("selector", VarianceThreshold(0.0)),
            ("classifier", LogisticRegression(max_iter=200)),
        ]
    ).fit(features, labels)


def test_binary_evaluate_model_perf_returns_train_and_valid_scores():
    """Binary evaluate_model_perf returns train/valid scores and logs curves."""
    features, labels = _binary()
    pipeline = _clf(features, labels)
    tracker = MagicMock()
    evaluator = BinaryClassificationEvaluator(
        tracker=tracker,
        pipeline=pipeline,
        train_features=features,
        train_class=labels,
        valid_features=features,
        valid_class=labels,
    )
    train_scores, valid_scores = evaluator.evaluate_model_perf(
        class_encoder=LabelEncoder().fit(labels)
    )
    for name in ("accuracy", "precision", "recall", "f1", "f_0.5_score", "roc_auc"):
        assert name in set(train_scores["Metric"])
        assert name in set(valid_scores["Metric"])
    # Curves are logged to the tracker (calibration, ROC, PR, gains, lift).
    assert tracker.log_figure.called
    # Confusion matrices are logged when a class encoder is supplied.
    assert tracker.log_confusion_matrix.called


def test_binary_get_pred_class_is_label_aware_and_threshold_respecting():
    """_get_pred_class returns the model's class labels and honours the threshold."""
    features, labels = _binary()
    pipeline = _clf(features, labels)
    evaluator = BinaryClassificationEvaluator(
        tracker=MagicMock(),
        pipeline=pipeline,
        train_features=features,
        train_class=labels,
        valid_features=features,
        valid_class=labels,
    )
    probs = pipeline.predict_proba(features)
    # A near-zero threshold predicts (almost) all positive; a near-one threshold
    # predicts (almost) all negative. Output uses the model's own labels {0, 1}.
    high_recall = evaluator._get_pred_class(probs, 0.01)
    low_recall = evaluator._get_pred_class(probs, 0.99)
    assert set(np.unique(high_recall)).issubset(set(pipeline.classes_))
    assert high_recall.sum() >= low_recall.sum()


def test_binary_expected_calibration_error_in_unit_interval():
    """Binary ECE is a float in [0, 1]."""
    features, labels = _binary()
    pipeline = _clf(features, labels)
    evaluator = BinaryClassificationEvaluator(
        tracker=MagicMock(),
        pipeline=pipeline,
        train_features=features,
        train_class=labels,
        valid_features=features,
        valid_class=labels,
    )
    probs = pipeline.predict_proba(features)
    ece = evaluator.calc_expected_calibration_error(probs, labels, nbins=10)
    assert 0.0 <= float(ece) <= 1.0


def test_multiclass_evaluate_model_perf_emits_macro_metrics():
    """Multi-class evaluate_model_perf returns macro/micro/weighted scores."""
    features, labels = _multiclass()
    pipeline = _clf(features, labels)
    evaluator = MultiClassificationEvaluator(
        tracker=MagicMock(),
        pipeline=pipeline,
        train_features=features,
        train_class=labels,
        valid_features=features,
        valid_class=labels,
    )
    _, valid_scores = evaluator.evaluate_model_perf(
        class_encoder=LabelEncoder().fit(labels)
    )
    names = set(valid_scores["Metric"])
    assert {"f1_macro", "f_0.5_score_macro", "roc_auc_macro"}.issubset(names)


def test_multiclass_expected_calibration_error_in_unit_interval():
    """Multi-class top-label ECE is a float in [0, 1]."""
    features, labels = _multiclass()
    pipeline = _clf(features, labels)
    evaluator = MultiClassificationEvaluator(
        tracker=MagicMock(),
        pipeline=pipeline,
        train_features=features,
        train_class=labels,
        valid_features=features,
        valid_class=labels,
    )
    probs = pipeline.predict_proba(features)
    ece = evaluator.calc_expected_calibration_error(probs, labels, nbins=10)
    assert 0.0 <= float(ece) <= 1.0


def test_regression_evaluate_model_perf_returns_scores():
    """Regression evaluate_model_perf returns train/valid regression metrics."""
    rng = np.random.default_rng(0)
    features = pd.DataFrame({"f1": rng.normal(size=100), "f2": rng.normal(size=100)})
    target = 2.0 * features["f1"] - features["f2"] + rng.normal(0, 0.1, 100)
    pipeline = Pipeline(
        steps=[("preprocessor", StandardScaler()), ("regressor", LinearRegression())]
    ).fit(features, target)
    evaluator = RegressionEvaluator(
        tracker=MagicMock(),
        pipeline=pipeline,
        train_features=features,
        train_class=target.to_numpy(),
        valid_features=features,
        valid_class=target.to_numpy(),
    )
    train_scores, valid_scores = evaluator.evaluate_model_perf()
    assert {"mae", "rmse", "r2"}.issubset(set(train_scores["Metric"]))
    assert {"mae", "rmse", "r2"}.issubset(set(valid_scores["Metric"]))
    # A good linear fit on linear data has high R2.
    r2 = valid_scores.loc[valid_scores["Metric"] == "r2", "Score"].iloc[0]
    assert r2 > 0.9
