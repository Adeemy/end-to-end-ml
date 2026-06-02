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


def test_calibrate_pipeline_uses_prefit(binary_data, fitted_pipeline):
    """Calibration must keep the trained model (cv='prefit'), not refit it."""
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
    assert calibrator.cv == "prefit"
    # The prefit estimator is the same trained object — coefficients unchanged.
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
