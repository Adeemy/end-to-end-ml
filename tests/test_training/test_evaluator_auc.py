"""
Regression guards for ROC-AUC in the evaluators.

    #1  The binary evaluator computes ROC-AUC from probabilities, not hard labels.
    #4  The multiclass evaluator emits a (one-vs-rest, macro) ROC-AUC from the
        full probability matrix.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.training.evaluation.evaluator import (
    BinaryClassificationEvaluator,
    MultiClassificationEvaluator,
)


def _fitted_pipeline(features, labels):
    """Fits a preprocessor/selector/classifier pipeline on the given data."""
    pipeline = Pipeline(
        steps=[
            ("preprocessor", StandardScaler()),
            ("selector", VarianceThreshold(0.0)),
            ("classifier", LogisticRegression(max_iter=200)),
        ]
    )
    pipeline.fit(features, labels)
    return pipeline


def test_binary_evaluator_auc_requires_probabilities():
    """Binary evaluator's AUC is probability-based and skipped without probs."""
    rng = np.random.default_rng(0)
    n = 80
    features = pd.DataFrame(
        {
            "f1": np.concatenate([rng.normal(0, 1, n), rng.normal(1.2, 1, n)]),
            "f2": np.concatenate([rng.normal(0, 1, n), rng.normal(1.2, 1, n)]),
        }
    )
    labels = np.array([0] * n + [1] * n)
    pipeline = _fitted_pipeline(features, labels)

    evaluator = BinaryClassificationEvaluator(
        tracker=MagicMock(),
        pipeline=pipeline,
        train_features=features,
        train_class=labels,
        valid_features=features,
        valid_class=labels,
    )
    proba = pipeline.predict_proba(features)[:, 1]
    pred = pipeline.predict(features)

    assert "roc_auc" not in set(
        evaluator.calc_perf_metrics(true_class=labels, pred_class=pred)["Metric"]
    )
    scores = evaluator.calc_perf_metrics(
        true_class=labels, pred_class=pred, pred_proba=proba
    )
    auc = scores.loc[scores["Metric"] == "roc_auc", "Score"].iloc[0]
    assert auc == pytest.approx(roc_auc_score(labels, proba))


def test_multiclass_evaluator_auc_from_proba_matrix():
    """Multiclass evaluator emits roc_auc_macro only with the proba matrix."""
    rng = np.random.default_rng(1)
    n = 60
    features = pd.DataFrame(
        {
            "f1": np.concatenate(
                [rng.normal(0, 1, n), rng.normal(3, 1, n), rng.normal(6, 1, n)]
            ),
            "f2": np.concatenate(
                [rng.normal(0, 1, n), rng.normal(3, 1, n), rng.normal(6, 1, n)]
            ),
        }
    )
    labels = np.array([0] * n + [1] * n + [2] * n)
    pipeline = _fitted_pipeline(features, labels)

    evaluator = MultiClassificationEvaluator(
        tracker=MagicMock(),
        pipeline=pipeline,
        train_features=features,
        train_class=labels,
        valid_features=features,
        valid_class=labels,
    )
    pred = pipeline.predict(features)
    proba = pipeline.predict_proba(features)

    assert "roc_auc_macro" not in set(
        evaluator.calc_perf_metrics(true_class=labels, pred_class=pred)["Metric"]
    )
    scores = evaluator.calc_perf_metrics(
        true_class=labels, pred_class=pred, pred_proba=proba
    )
    auc = scores.loc[scores["Metric"] == "roc_auc_macro", "Score"].iloc[0]
    assert auc == pytest.approx(
        roc_auc_score(labels, proba, multi_class="ovr", average="macro")
    )
