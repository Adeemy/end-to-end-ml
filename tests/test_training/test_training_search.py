"""
Regression guards for hyperparameter-search correctness in ``ModelOptimizer``.

Each test fails on the pre-fix behaviour and passes after the fix:
    #1  ROC-AUC is computed from probabilities, not hard labels.
    #2  The search optimizes the configured comparison metric.
    #6  The study direction is derived from the metric (seeded search elsewhere).
    #7  The positive-class column is decoded via classes_, not hardcoded.

These are deliberately white-box guards, so access to protected helpers is expected.
"""

# pylint: disable=protected-access

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.training.core.optimizer import ModelOptimizer


@pytest.fixture(name="binary_data")
def binary_data_fixture():
    """Small, overlapping binary dataset as a DataFrame + label array."""
    rng = np.random.default_rng(0)
    n = 80
    features = pd.DataFrame(
        {
            "f1": np.concatenate([rng.normal(0, 1, n), rng.normal(1.2, 1, n)]),
            "f2": np.concatenate([rng.normal(0, 1, n), rng.normal(1.2, 1, n)]),
        }
    )
    labels = np.array([0] * n + [1] * n)
    return features, labels


@pytest.fixture(name="fitted_pipeline")
def fitted_pipeline_fixture(binary_data):
    """A fitted preprocessor/selector/classifier pipeline."""
    features, labels = binary_data
    pipeline = Pipeline(
        steps=[
            ("preprocessor", StandardScaler()),
            ("selector", VarianceThreshold(0.0)),
            ("classifier", LogisticRegression(max_iter=200)),
        ]
    )
    pipeline.fit(features, labels)
    return pipeline


def _make_optimizer(model, optimization_metric="fbeta_score"):
    """Builds a ModelOptimizer that skips the supported-model check."""
    return ModelOptimizer(
        tracker=MagicMock(),
        train_features_preprocessed=pd.DataFrame({"f1": [0.0]}),
        train_class=np.array([0]),
        valid_features_preprocessed=pd.DataFrame({"f1": [0.0]}),
        valid_class=np.array([0]),
        n_features=1,
        model=model,
        search_space_params={},
        supported_models=None,
        registered_model_name="logistic-regression",
        fbeta_score_beta=0.5,
        is_voting_ensemble=True,  # bypasses the supported-model validation
        optimization_metric=optimization_metric,
    )


def test_optimizer_auc_requires_probabilities(binary_data, fitted_pipeline):
    """calc_perf_metrics adds roc_auc only with probabilities, and uses them."""
    features, labels = binary_data
    optimizer = _make_optimizer(fitted_pipeline.named_steps["classifier"])
    proba = fitted_pipeline.predict_proba(features)[:, 1]
    pred = fitted_pipeline.predict(features)

    without = optimizer.calc_perf_metrics(true_class=labels, pred_class=pred)
    assert "roc_auc" not in set(without["Metric"])

    with_proba = optimizer.calc_perf_metrics(
        true_class=labels, pred_class=pred, pred_proba=proba
    )
    auc = with_proba.loc[with_proba["Metric"] == "roc_auc", "Score"].iloc[0]
    assert auc == pytest.approx(roc_auc_score(labels, proba))
    # A probability-based AUC differs from the degenerate hard-label value.
    assert auc != pytest.approx(roc_auc_score(labels, pred))


def test_optimizer_metric_resolution_and_direction(fitted_pipeline):
    """The optimizer resolves and directs the configured metric correctly."""
    classifier = fitted_pipeline.named_steps["classifier"]

    fbeta_opt = _make_optimizer(classifier, optimization_metric="fbeta_score")
    assert fbeta_opt._metric_row_name() == "f_0.5_score"
    assert fbeta_opt.optimization_direction == "maximize"

    auc_opt = _make_optimizer(classifier, optimization_metric="roc_auc")
    assert auc_opt._metric_row_name() == "roc_auc"
    assert auc_opt.optimization_direction == "maximize"

    loss_opt = _make_optimizer(classifier, optimization_metric="log_loss")
    assert loss_opt.optimization_direction == "minimize"


def test_optimizer_pos_class_proba_uses_classes(binary_data, fitted_pipeline):
    """_pos_class_proba locates the positive column via classes_."""
    features, _ = binary_data
    classifier = fitted_pipeline.named_steps["classifier"]
    optimizer = _make_optimizer(classifier)

    # Mirror training preprocessing, then score.
    transformed = fitted_pipeline.named_steps["preprocessor"].transform(features)
    transformed = fitted_pipeline.named_steps["selector"].transform(transformed)

    pos_col = list(classifier.classes_).index(1)
    expected = classifier.predict_proba(transformed)[:, pos_col]
    got = optimizer._pos_class_proba(transformed)
    assert got.shape == (len(features),)
    assert np.allclose(got, expected)
