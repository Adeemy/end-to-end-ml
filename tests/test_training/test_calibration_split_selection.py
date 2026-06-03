"""
Guards for the dedicated calibration split + in-process champion selection.

    Concern #1/#2 — calibration and threshold tuning use the dedicated calibration
        split (disjoint from the validation set used for selection).
    Concern #4 — champion selection recomputes the comparison metric in-process,
        without reading metrics back from the experiment tracker.

These exercise protected helpers directly, so protected-access is expected.
"""

# pylint: disable=protected-access

from unittest.mock import MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature.utils.data import TrainingDataPrep
from src.training.evaluation.champion import ModelChampionManager
from src.training.evaluation.orchestrator import (
    TestSetEvaluationOrchestrator as EvalOrchestrator,
)


def _make_frame(n, seed, sep):
    """Binary dataset with an ID column and two numeric features."""
    rng = np.random.default_rng(seed)
    half = n // 2
    features = {
        "f1": np.concatenate([rng.normal(0, 1, half), rng.normal(sep, 1, n - half)]),
        "f2": np.concatenate([rng.normal(0, 1, half), rng.normal(sep, 1, n - half)]),
    }
    frame = pd.DataFrame(features)
    frame["y"] = np.array([0] * half + [1] * (n - half))
    frame["ID"] = np.arange(n) + seed * 10_000
    return frame


def _fit_pipeline(features, labels):
    """Fits a preprocessor/selector/classifier pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", StandardScaler()),
            ("selector", VarianceThreshold(0.0)),
            ("classifier", LogisticRegression(max_iter=200)),
        ]
    ).fit(features, labels)


# --- #1/#2: a dedicated, disjoint calibration split ------------------------


def test_calibration_split_is_disjoint():
    """train / valid / calibration are pairwise disjoint and partition the dev set."""
    dev = _make_frame(200, seed=1, sep=2.0)
    test = _make_frame(50, seed=2, sep=2.0)

    prep = TrainingDataPrep(
        train_set=dev,
        test_set=test,
        primary_key="ID",
        class_col_name="y",
        numerical_feature_names=["f1", "f2"],
        categorical_feature_names=[],
    )
    dev_ids = set(dev["ID"])
    prep.create_validation_set(
        split_type="random", train_set_size=0.8, split_random_seed=42
    )
    prep.create_calibration_set(
        split_type="random", train_set_size=0.8, split_random_seed=42
    )

    train_ids = set(prep.train_set["ID"])
    valid_ids = set(prep.valid_set["ID"])
    calib_ids = set(prep.calib_set["ID"])

    assert prep.calib_set is not None and len(calib_ids) > 0
    # Pairwise disjoint.
    assert train_ids.isdisjoint(valid_ids)
    assert train_ids.isdisjoint(calib_ids)
    assert valid_ids.isdisjoint(calib_ids)
    # The three partition the original dev set exactly.
    assert train_ids | valid_ids | calib_ids == dev_ids


def test_create_calibration_set_guards_double_call():
    """Calling create_calibration_set twice raises rather than overwriting."""
    dev = _make_frame(120, seed=3, sep=2.0)
    test = _make_frame(40, seed=4, sep=2.0)
    prep = TrainingDataPrep(
        train_set=dev,
        test_set=test,
        primary_key="ID",
        class_col_name="y",
        numerical_feature_names=["f1", "f2"],
        categorical_feature_names=[],
    )
    prep.create_validation_set(train_set_size=0.8, split_random_seed=0)
    prep.create_calibration_set(train_set_size=0.8, split_random_seed=0)
    with pytest.raises(ValueError):
        prep.create_calibration_set(train_set_size=0.8, split_random_seed=0)


def _make_orchestrator(tmp_path, tune=False):
    """Minimal orchestrator with a mock tracker and tiny train/test data."""
    train = _make_frame(60, seed=9, sep=2.0)
    return EvalOrchestrator(
        tracker=MagicMock(),
        train_features=train[["f1", "f2"]],
        train_class=train["y"].to_numpy(),
        test_features=train[["f1", "f2"]],
        test_class=train["y"].to_numpy(),
        artifacts_path=str(tmp_path),
        fbeta_score_beta=0.5,
        voting_ensemble_name=None,
        decision_threshold=0.5,
        tune_decision_threshold=tune,
        encoded_pos_class_label=1,
    )


def test_threshold_resolved_from_calibration_data(tmp_path):
    """The persisted threshold is tuned on the calibration data passed in."""
    train = _make_frame(160, seed=5, sep=1.2)
    pipeline = _fit_pipeline(train[["f1", "f2"]], train["y"])
    calib = _make_frame(160, seed=6, sep=1.2)
    calib_x, calib_y = calib[["f1", "f2"]], calib["y"].to_numpy()

    orch = _make_orchestrator(tmp_path, tune=True)
    threshold = orch._resolve_decision_threshold(pipeline, calib_x, calib_y)

    pos_probs = pipeline.predict_proba(calib_x)[:, list(pipeline.classes_).index(1)]
    expected = ModelChampionManager.tune_decision_threshold(
        valid_class=calib_y, pos_class_proba=pos_probs, fbeta_beta=0.5
    )
    assert threshold == pytest.approx(expected)


# --- #4: in-process champion selection (no tracker reads) ------------------


def test_in_process_selection_picks_best_without_tracker(tmp_path):
    """Champion is chosen by recomputed valid metric; the tracker is never read."""
    # A "good" model trained on the real pattern, and a "bad" model trained on
    # shuffled labels (≈ random). The good one must win on validation AUC.
    train = _make_frame(200, seed=7, sep=1.5)
    good = _fit_pipeline(train[["f1", "f2"]], train["y"])

    rng = np.random.default_rng(0)
    bad = _fit_pipeline(train[["f1", "f2"]], rng.permutation(train["y"].to_numpy()))

    joblib.dump(good, tmp_path / "good.pkl")
    joblib.dump(bad, tmp_path / "bad.pkl")

    valid = _make_frame(200, seed=8, sep=1.5)
    valid_x, valid_y = valid[["f1", "f2"]], valid["y"].to_numpy()

    orch = _make_orchestrator(tmp_path)
    champion_name, champion_pipeline = orch._select_champion_in_process(
        candidates=[("good", None), ("bad", None)],
        valid_features=valid_x,
        valid_class=valid_y,
        comparison_metric_name="roc_auc",
    )

    # Sanity: the good model really does score higher on valid AUC.
    good_auc = roc_auc_score(valid_y, good.predict_proba(valid_x)[:, 1])
    bad_auc = roc_auc_score(valid_y, bad.predict_proba(valid_x)[:, 1])
    assert good_auc > bad_auc

    assert champion_name == "good"
    # The returned pipeline is the (reloaded) good model — verify behaviourally,
    # since joblib.load returns a deserialized copy (not the same object).
    assert champion_pipeline is not None
    assert np.allclose(
        champion_pipeline.predict_proba(valid_x), good.predict_proba(valid_x)
    )
    # Selection must NOT consult the tracker for metrics.
    orch.tracker.get_metric.assert_not_called()
    orch.tracker.get_metrics.assert_not_called()


def test_score_pipeline_matches_sklearn_auc(tmp_path):
    """_score_pipeline_on_valid reproduces sklearn's ROC-AUC on the positive class."""
    train = _make_frame(160, seed=11, sep=1.3)
    pipeline = _fit_pipeline(train[["f1", "f2"]], train["y"])
    valid = _make_frame(160, seed=12, sep=1.3)
    valid_x, valid_y = valid[["f1", "f2"]], valid["y"].to_numpy()

    orch = _make_orchestrator(tmp_path)
    score = orch._score_pipeline_on_valid(
        pipeline, "model", valid_x, valid_y, "roc_auc"
    )
    expected = roc_auc_score(valid_y, pipeline.predict_proba(valid_x)[:, 1])
    assert score == pytest.approx(expected)
