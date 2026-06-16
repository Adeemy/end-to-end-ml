"""
Regression guards for champion calibration and decision-threshold handling.

    #3  Calibration keeps the train-fitted model (cv="prefit"), not a refit.
    #5  The decision threshold is tuned on validation and persisted as metadata.
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.training.evaluation.champion import ModelChampionManager


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


def test_calibrate_pipeline_keeps_trained_model_frozen(binary_data, fitted_pipeline):
    """Calibration must keep the trained model fixed (FrozenEstimator), not refit it."""
    features, labels = binary_data
    trained_classifier = fitted_pipeline.named_steps["classifier"]
    original_coef = trained_classifier.coef_.copy()

    calibrated = ModelChampionManager.calibrate_pipeline(
        valid_features=features,
        valid_class=labels,
        fitted_pipeline=fitted_pipeline,
    )
    calibrator = calibrated.named_steps["classifier"]
    assert isinstance(calibrator, CalibratedClassifierCV)
    # The trained model is wrapped frozen so only the calibration map is fit
    # (the supported replacement for the removed cv="prefit").
    assert isinstance(calibrator.estimator, FrozenEstimator)
    # The frozen estimator is the same trained object -> coefficients unchanged.
    assert np.allclose(trained_classifier.coef_, original_coef)


def test_tune_decision_threshold_returns_valid_threshold(binary_data, fitted_pipeline):
    """Threshold tuning returns a usable in-range threshold."""
    features, labels = binary_data
    proba = fitted_pipeline.predict_proba(features)[:, 1]
    threshold = ModelChampionManager.tune_decision_threshold(
        valid_class=labels, pos_class_proba=proba, fbeta_beta=0.5
    )
    assert 0.0 < threshold < 1.0


def test_save_model_metadata_roundtrip(tmp_path):
    """Serving metadata is persisted with threshold and positive class."""
    manager = ModelChampionManager(champ_model_name="champion")
    path = manager.save_model_metadata(
        local_path=str(tmp_path),
        decision_threshold=0.42,
        encoded_pos_class_label=1,
    )
    with open(path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["decision_threshold"] == pytest.approx(0.42)
    assert metadata["encoded_pos_class_label"] == 1


def test_select_best_performer_picks_highest_metric():
    """select_best_performer returns the model with the highest tracked metric."""
    from unittest.mock import MagicMock  # pylint: disable=import-outside-toplevel

    lr_tracker = MagicMock()
    lr_tracker.get_metric.return_value = 0.70
    rf_tracker = MagicMock()
    rf_tracker.get_metric.return_value = 0.85
    manager = ModelChampionManager(champ_model_name="champion")

    best = manager.select_best_performer(
        trackers={"lr": lr_tracker, "rf": rf_tracker},
        comparison_metric="valid_roc_auc",
    )
    assert best == "rf"


def test_select_best_performer_raises_without_scores():
    """A metric absent from every tracker raises rather than guessing."""
    from unittest.mock import MagicMock  # pylint: disable=import-outside-toplevel

    tracker = MagicMock()
    tracker.get_metric.return_value = None
    manager = ModelChampionManager(champ_model_name="champion")
    with pytest.raises(ValueError):
        manager.select_best_performer(
            trackers={"lr": tracker}, comparison_metric="valid_roc_auc"
        )


def test_log_and_register_champ_model_writes_and_registers(tmp_path, fitted_pipeline):
    """log_and_register_champ_model dumps the pkl and calls the tracker once each."""
    from unittest.mock import MagicMock  # pylint: disable=import-outside-toplevel

    tracker = MagicMock()
    manager = ModelChampionManager(champ_model_name="champion", tracker=tracker)
    manager.log_and_register_champ_model(
        local_path=str(tmp_path), pipeline=fitted_pipeline
    )

    assert (tmp_path / "champion.pkl").exists()
    tracker.log_model.assert_called_once()
    tracker.register_model.assert_called_once()
    tracker.end.assert_called_once()


def test_log_and_register_requires_tracker(tmp_path, fitted_pipeline):
    """Registration without a configured tracker raises."""
    manager = ModelChampionManager(champ_model_name="champion", tracker=None)
    with pytest.raises(ValueError):
        manager.log_and_register_champ_model(
            local_path=str(tmp_path), pipeline=fitted_pipeline
        )
