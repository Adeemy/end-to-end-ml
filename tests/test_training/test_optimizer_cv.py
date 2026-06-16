"""
Guard for the cross-validated objective in ModelOptimizer.

With cv_folds > 1 the objective scores each trial as the mean of the
optimization metric over stratified folds (variance-aware tuning) and records
the per-trial standard error for the 1-SE selection rule.
"""

from unittest.mock import MagicMock

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.training.core.optimizer import ModelOptimizer


@pytest.fixture(name="preprocessed_binary")
def preprocessed_binary_fixture():
    """A separable, already-preprocessed binary dataset split into train/valid."""
    rng = np.random.default_rng(0)
    n = 60
    features = pd.DataFrame(
        {
            "f1": np.concatenate([rng.normal(0, 1, n), rng.normal(2.5, 1, n)]),
            "f2": np.concatenate([rng.normal(0, 1, n), rng.normal(2.5, 1, n)]),
        }
    )
    labels = np.array([0] * n + [1] * n)
    order = rng.permutation(len(features))
    features, labels = features.iloc[order].reset_index(drop=True), labels[order]
    split = len(features) // 2
    return (
        features.iloc[:split].reset_index(drop=True),
        labels[:split],
        features.iloc[split:].reset_index(drop=True),
        labels[split:],
    )


def _make_cv_optimizer(preprocessed_binary, cv_folds):
    train_x, train_y, valid_x, valid_y = preprocessed_binary
    return ModelOptimizer(
        tracker=MagicMock(),
        train_features_preprocessed=train_x,
        train_class=train_y,
        valid_features_preprocessed=valid_x,
        valid_class=valid_y,
        n_features=train_x.shape[1],
        model=LogisticRegression(max_iter=200),
        search_space_params={},  # no params to sample; trials use the base model
        supported_models=None,
        registered_model_name="logistic-regression",
        fbeta_score_beta=0.5,
        is_voting_ensemble=True,  # bypass the supported-model validation
        optimization_metric="roc_auc",
        random_seed=42,
        cv_folds=cv_folds,
    )


def test_cv_objective_returns_finite_mean_and_records_std(preprocessed_binary):
    """A CV trial returns a finite mean ROC-AUC and stores the per-trial std."""
    optimizer = _make_cv_optimizer(preprocessed_binary, cv_folds=3)
    study = optuna.create_study(direction=optimizer.optimization_direction)
    study.optimize(optimizer.obj_func, n_trials=1)

    assert np.isfinite(study.best_value)
    assert 0.5 <= study.best_value <= 1.0  # separable data -> better than chance
    # The CV path records the per-fold standard error for the 1-SE rule.
    assert "cv_std" in study.best_trial.user_attrs
    assert study.best_trial.user_attrs["cv_std"] >= 0.0


def test_single_holdout_path_when_cv_disabled(preprocessed_binary):
    """cv_folds <= 1 uses the single-holdout objective (no cv_std attr)."""
    optimizer = _make_cv_optimizer(preprocessed_binary, cv_folds=1)
    study = optuna.create_study(direction=optimizer.optimization_direction)
    study.optimize(optimizer.obj_func, n_trials=1)

    assert np.isfinite(study.best_value)
    assert "cv_std" not in study.best_trial.user_attrs


def test_tune_model_serial_runs_and_returns_study(preprocessed_binary):
    """The serial tune_model runs the configured number of trials and returns a study."""
    optimizer = _make_cv_optimizer(preprocessed_binary, cv_folds=1)
    study = optimizer.tune_model(max_search_iters=3, model_opt_timeout_secs=60)
    assert len(study.trials) >= 1
    assert np.isfinite(study.best_value)


def test_proba_for_metrics_binary_and_regression(preprocessed_binary):
    """_proba_for_metrics returns 1-D probs for binary and None for regression."""
    train_x, train_y, _, _ = preprocessed_binary
    model = LogisticRegression(max_iter=200).fit(train_x, train_y)

    binary_opt = _make_cv_optimizer(preprocessed_binary, cv_folds=1)
    binary_opt.model = model
    proba = binary_opt._proba_for_metrics(train_x)  # pylint: disable=protected-access
    assert proba.ndim == 1

    reg_opt = _make_cv_optimizer(preprocessed_binary, cv_folds=1)
    reg_opt.task_type = "regression"
    assert (
        reg_opt._proba_for_metrics(train_x) is None
    )  # pylint: disable=protected-access


def test_proba_for_metrics_multiclass_returns_matrix():
    """_proba_for_metrics returns the full probability matrix for multi-class."""
    rng = np.random.default_rng(0)
    features = pd.DataFrame({"f1": rng.normal(size=60), "f2": rng.normal(size=60)})
    labels = np.array([0, 1, 2] * 20)
    model = LogisticRegression(max_iter=200).fit(features, labels)
    optimizer = ModelOptimizer(
        tracker=MagicMock(),
        train_features_preprocessed=features,
        train_class=labels,
        valid_features_preprocessed=features,
        valid_class=labels,
        n_features=2,
        model=model,
        search_space_params={},
        supported_models=None,
        registered_model_name="logistic-regression",
        is_voting_ensemble=True,
        optimization_metric="roc_auc",
        task_type="multiclass",
    )
    proba = optimizer._proba_for_metrics(features)  # pylint: disable=protected-access
    assert proba.ndim == 2 and proba.shape[1] == 3
