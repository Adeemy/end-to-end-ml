"""
Rigor guards for task-aware evaluation and champion selection.

Covers the fixes that make offline selection predict production behaviour:
  - the test set is scored at the deployed operating threshold (not 0.5);
  - calibration is applied before test evaluation;
  - multi-class selection resolves the macro metric (no silent fallback);
  - the 1-SE rule prefers the simpler model among statistical ties;
  - the evaluator factory dispatches by explicit task_type (incl. regression).

These exercise protected helpers directly, so protected-access is expected.
"""

# pylint: disable=protected-access

import json
from unittest.mock import MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.training.evaluation import metrics
from src.training.evaluation.champion import ModelChampionManager
from src.training.evaluation.evaluator import (
    BinaryClassificationEvaluator,
    MultiClassificationEvaluator,
    RegressionEvaluator,
    create_model_evaluator,
)
from src.training.evaluation.orchestrator import (
    TestSetEvaluationOrchestrator as EvalOrchestrator,
)
from src.training.evaluation.selector import ModelSelector


def _binary_frame(n, seed, sep):
    """Binary dataset with two numeric features."""
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


def _multiclass_frame(n, seed, sep):
    """Three-class dataset with two numeric features."""
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


def _fit_clf(features, labels):
    return Pipeline(
        steps=[
            ("preprocessor", StandardScaler()),
            ("selector", VarianceThreshold(0.0)),
            ("classifier", LogisticRegression(max_iter=200)),
        ]
    ).fit(features, labels)


def _orchestrator(tmp_path, task_type="binary", cv_folds=1, model_preference=None):
    train_x, train_y = _binary_frame(60, seed=9, sep=2.0)
    return EvalOrchestrator(
        tracker=MagicMock(),
        train_features=train_x,
        train_class=train_y,
        test_features=train_x,
        test_class=train_y,
        artifacts_path=str(tmp_path),
        fbeta_score_beta=0.5,
        voting_ensemble_name=None,
        decision_threshold=0.5,
        tune_decision_threshold=True,
        encoded_pos_class_label=1,
        task_type=task_type,
        cv_folds=cv_folds,
        model_preference=model_preference,
        random_seed=7,
    )


# --- factory dispatch by explicit task_type --------------------------------


def test_factory_dispatch_by_task_type():
    """create_model_evaluator honours explicit task_type, incl. regression."""
    features, labels = _binary_frame(40, seed=1, sep=2.0)
    pipeline = _fit_clf(features, labels)
    common = dict(
        tracker=MagicMock(),
        pipeline=pipeline,
        train_features=features,
        train_class=labels,
        valid_features=features,
        valid_class=labels,
    )
    assert isinstance(
        create_model_evaluator(task_type="binary", **common),
        BinaryClassificationEvaluator,
    )
    assert isinstance(
        create_model_evaluator(task_type="multiclass", **common),
        MultiClassificationEvaluator,
    )
    assert isinstance(
        create_model_evaluator(task_type="regression", **common), RegressionEvaluator
    )


def test_factory_binary_with_extra_classes_falls_back_to_multiclass():
    """task_type='binary' but 3 observed classes uses the multi-class evaluator."""
    features, labels = _multiclass_frame(60, seed=2, sep=3.0)
    pipeline = _fit_clf(features, labels)
    evaluator = create_model_evaluator(
        tracker=MagicMock(),
        pipeline=pipeline,
        train_features=features,
        train_class=labels,
        valid_features=features,
        valid_class=labels,
        task_type="binary",
    )
    assert isinstance(evaluator, MultiClassificationEvaluator)


# --- regression evaluator ---------------------------------------------------


def test_regression_evaluator_emits_regression_metrics():
    """RegressionEvaluator computes MAE/RMSE/R2 without probabilities/threshold."""
    rng = np.random.default_rng(0)
    features = pd.DataFrame({"f1": rng.normal(size=80), "f2": rng.normal(size=80)})
    target = 3.0 * features["f1"] - 2.0 * features["f2"] + rng.normal(0, 0.1, 80)
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
    scores = evaluator.evaluate_test_set_only(class_encoder=None)
    metric_names = set(scores["Metric"])
    assert {"mae", "rmse", "r2"}.issubset(metric_names)


# --- multi-class selection resolves the macro metric (no silent fallback) ---


def test_multiclass_selection_uses_macro_metric(tmp_path):
    """The macro metric row is found, so the better multi-class model is chosen."""
    train_x, train_y = _multiclass_frame(180, seed=3, sep=2.5)
    good = _fit_clf(train_x, train_y)
    rng = np.random.default_rng(0)
    bad = _fit_clf(train_x, rng.permutation(train_y))
    joblib.dump(good, tmp_path / "good.pkl")
    joblib.dump(bad, tmp_path / "bad.pkl")

    valid_x, valid_y = _multiclass_frame(180, seed=4, sep=2.5)
    orch = _orchestrator(tmp_path, task_type="multiclass")

    metric_name = metrics.selection_metric_row_name("roc_auc", "multiclass", 0.5)
    assert metric_name == "roc_auc_macro"

    # The bare 'roc_auc' name would not exist for multi-class (the old bug),
    # but the resolved macro name does and yields a finite score.
    assert orch._score_pipeline_on_valid(
        good, "good", valid_x, valid_y, "roc_auc"
    ) == float("-inf")
    assert np.isfinite(
        orch._score_pipeline_on_valid(good, "good", valid_x, valid_y, metric_name)
    )

    champion_name, champion_pipeline = orch._select_champion_in_process(
        candidates=[("good", None), ("bad", None)],
        valid_features=valid_x,
        valid_class=valid_y,
        comparison_metric_name=metric_name,
    )
    assert champion_name == "good"
    assert champion_pipeline is not None  # not the silent first-candidate fallback


# --- the test set is scored at the deployed operating threshold -------------


def test_test_eval_applies_operating_threshold(tmp_path):
    """Changing the operating threshold changes the reported test metrics."""
    train_x, train_y = _binary_frame(200, seed=5, sep=1.0)
    pipeline = _fit_clf(train_x, train_y)
    orch = _orchestrator(tmp_path, task_type="binary")
    orch.train_features, orch.train_class = train_x, train_y
    orch.test_features, orch.test_class = train_x, train_y

    low = orch.evaluate_on_test_set(pipeline, "m", decision_threshold=0.1)
    high = orch.evaluate_on_test_set(pipeline, "m", decision_threshold=0.9)

    # A lower threshold predicts the positive class more often -> higher recall.
    assert low["test_recall"] >= high["test_recall"]
    # The probability-based metric (ROC-AUC) is threshold-invariant -> unchanged.
    assert low["test_roc_auc"] == pytest.approx(high["test_roc_auc"])


def test_calibrate_then_resolve_returns_calibrated_pipeline(tmp_path):
    """Calibration runs before test eval and yields a calibrated pipeline + threshold."""
    train_x, train_y = _binary_frame(200, seed=6, sep=1.2)
    pipeline = _fit_clf(train_x, train_y)
    calib_x, calib_y = _binary_frame(200, seed=7, sep=1.2)
    orch = _orchestrator(tmp_path, task_type="binary")

    manager = ModelChampionManager(champ_model_name="champion")
    calibrated, threshold = orch.calibrate_and_resolve_threshold(
        model_pipeline=pipeline,
        model_name="m",
        calibration_features=calib_x,
        calibration_class=calib_y,
        champion_manager=manager,
    )
    assert isinstance(calibrated.named_steps["classifier"], CalibratedClassifierCV)
    assert 0.0 < threshold < 1.0


# --- 1-SE rule prefers the simpler/cheaper model among ties -----------------


def test_one_se_rule_prefers_preferred_model_on_tie(tmp_path):
    """With CV variance and a tie, model_preference decides the champion."""
    train_x, train_y = _binary_frame(160, seed=8, sep=1.5)
    pipeline = _fit_clf(train_x, train_y)
    # Identical model saved under two names -> identical means and SEs (a tie).
    joblib.dump(pipeline, tmp_path / "model_a.pkl")
    joblib.dump(pipeline, tmp_path / "model_b.pkl")
    valid_x, valid_y = _binary_frame(160, seed=8, sep=1.5)

    candidates = [("model_a", None), ("model_b", None)]
    metric_name = "roc_auc"

    orch_prefers_b = _orchestrator(
        tmp_path, cv_folds=3, model_preference=["model_b", "model_a"]
    )
    name_b, _ = orch_prefers_b._select_champion_in_process(
        candidates, valid_x, valid_y, metric_name
    )
    assert name_b == "model_b"

    orch_prefers_a = _orchestrator(
        tmp_path, cv_folds=3, model_preference=["model_a", "model_b"]
    )
    name_a, _ = orch_prefers_a._select_champion_in_process(
        candidates, valid_x, valid_y, metric_name
    )
    assert name_a == "model_a"


# --- full workflow: select -> calibrate -> test@threshold -> gate -> register ---


def _seed_two_candidates(tmp_path, sep=1.5):
    """Trains a good and a bad candidate, saves them, returns (orch, valid, calib)."""
    train_x, train_y = _binary_frame(200, seed=20, sep=sep)
    good = _fit_clf(train_x, train_y)
    bad = _fit_clf(train_x, np.random.default_rng(1).permutation(train_y))
    joblib.dump(good, tmp_path / "good.pkl")
    joblib.dump(bad, tmp_path / "bad.pkl")
    orch = _orchestrator(tmp_path, task_type="binary", model_preference=["good", "bad"])
    orch.test_features, orch.test_class = _binary_frame(200, seed=21, sep=sep)
    valid = _binary_frame(200, seed=22, sep=sep)
    calib = _binary_frame(200, seed=23, sep=sep)
    return orch, valid, calib


def _run_workflow(orch, valid, calib, deployment_threshold):
    """Drives run_evaluation_workflow with training-supplied experiment keys."""
    valid_x, valid_y = valid
    calib_x, calib_y = calib
    return orch.run_evaluation_workflow(
        model_selector=ModelSelector(
            project_name="p", workspace_name="w", comparison_metric="valid_roc_auc"
        ),
        valid_features=valid_x,
        valid_class=valid_y,
        calibration_features=calib_x,
        calibration_class=calib_y,
        champion_manager=ModelChampionManager(champ_model_name="champion"),
        comparison_metric_name="roc_auc",
        deployment_threshold=deployment_threshold,
        experiment_keys=pd.DataFrame([["good", "k_good"], ["bad", "k_bad"]]),
        experiment_tracker_type="mlflow",
    )


def test_run_evaluation_workflow_registers_calibrated_champion(tmp_path):
    """The workflow selects the better model and registers the CALIBRATED champion."""
    orch, valid, calib = _seed_two_candidates(tmp_path)
    name, test_metrics = _run_workflow(orch, valid, calib, deployment_threshold=0.0)

    assert name == "good"
    assert "test_roc_auc" in test_metrics
    assert (tmp_path / "champion.pkl").exists()
    assert (tmp_path / "champion_metadata.json").exists()
    saved = joblib.load(tmp_path / "champion.pkl")
    assert isinstance(saved.named_steps["classifier"], CalibratedClassifierCV)


def test_run_evaluation_workflow_gate_blocks_below_threshold(tmp_path):
    """A model below the deployment threshold raises and is NOT registered."""
    orch, valid, calib = _seed_two_candidates(tmp_path)
    with pytest.raises(ValueError, match="deployment"):
        _run_workflow(orch, valid, calib, deployment_threshold=1.5)
    assert not (tmp_path / "champion.pkl").exists()


def test_reported_test_metrics_match_saved_champion(tmp_path):
    """Headline guarantee: reported test metrics equal the deployed champion at its
    persisted threshold (calibrated pipeline + tuned operating point)."""
    orch, valid, calib = _seed_two_candidates(tmp_path)
    name, test_metrics = _run_workflow(orch, valid, calib, deployment_threshold=0.0)
    assert name == "good"

    saved = joblib.load(tmp_path / "champion.pkl")
    meta = json.loads((tmp_path / "champion_metadata.json").read_text())
    threshold = meta["decision_threshold"]
    test_x, test_y = orch.test_features, orch.test_class

    pos_col = list(saved.classes_).index(1)
    proba = saved.predict_proba(test_x)[:, pos_col]
    pred = (proba >= threshold).astype(int)

    # ROC-AUC (threshold-invariant) and recall (threshold-dependent) both match,
    # proving the report reflects the exact deployed artifact and operating point.
    assert test_metrics["test_roc_auc"] == pytest.approx(roc_auc_score(test_y, proba))
    assert test_metrics["test_recall"] == pytest.approx(recall_score(test_y, pred))
