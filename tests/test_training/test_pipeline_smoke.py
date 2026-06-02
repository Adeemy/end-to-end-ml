"""
End-to-end smoke test for the evaluation -> serving path.

Runs in-process on synthetic data (no experiment tracker or secrets required) so
it can execute in CI via ``make test``. It exercises the chain that the review
fixes touch: train a pipeline -> calibrate it (prefit) -> tune + persist the
decision threshold -> load it back at inference and apply the persisted
threshold and positive-class index.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.inference.utils import helpers
from src.training.evaluation.champion import ModelChampionManager


def _make_dataset():
    """Synthetic, slightly overlapping binary dataset."""
    rng = np.random.default_rng(42)
    n = 120
    features = pd.DataFrame(
        {
            "f1": np.concatenate([rng.normal(0, 1, n), rng.normal(1.5, 1, n)]),
            "f2": np.concatenate([rng.normal(0, 1, n), rng.normal(1.5, 1, n)]),
        }
    )
    labels = np.array([0] * n + [1] * n)
    return features, labels


def test_calibrate_threshold_and_serve(tmp_path, monkeypatch):
    """Calibrate, persist the tuned threshold, then serve with it applied."""
    features, labels = _make_dataset()

    # Shuffle so both splits contain both classes (the raw data is class-ordered).
    order = np.random.default_rng(7).permutation(len(features))
    features, labels = features.iloc[order].reset_index(drop=True), labels[order]

    split = len(features) // 2
    train_x, valid_x = features.iloc[:split], features.iloc[split:]
    train_y, valid_y = labels[:split], labels[split:]

    # 1. Train the full pipeline on the training split.
    pipeline = Pipeline(
        steps=[
            ("preprocessor", StandardScaler()),
            ("selector", VarianceThreshold(0.0)),
            ("classifier", LogisticRegression(max_iter=200)),
        ]
    )
    pipeline.fit(train_x, train_y)

    # 2. Calibrate on validation while keeping the train-fitted model (prefit).
    manager = ModelChampionManager(champ_model_name="smoke_champion")
    calibrated = manager.calibrate_pipeline(
        valid_features=valid_x, valid_class=valid_y, fitted_pipeline=pipeline
    )

    # 3. Tune the decision threshold on validation and persist serving metadata.
    pos_col = list(calibrated.classes_).index(1)
    valid_proba = calibrated.predict_proba(valid_x)[:, pos_col]
    threshold = manager.tune_decision_threshold(
        valid_class=valid_y, pos_class_proba=valid_proba, fbeta_beta=0.5
    )
    manager.save_model_metadata(
        local_path=str(tmp_path),
        decision_threshold=threshold,
        encoded_pos_class_label=1,
    )

    # 4. Serve: point inference at the temp artifacts dir and predict.
    monkeypatch.setattr(helpers, "ARTIFACTS_DIR", tmp_path)
    metadata = helpers.load_serving_metadata("smoke_champion")
    assert metadata["decision_threshold"] == pytest.approx(threshold)

    pos_proba, pred_class, applied_threshold = helpers.positive_class_predictions(
        calibrated, valid_x, "smoke_champion"
    )

    # The serving path must reproduce the calibrated probabilities, apply the
    # persisted threshold, and produce a usable, non-degenerate classifier.
    assert applied_threshold == pytest.approx(threshold)
    assert np.allclose(pos_proba, valid_proba)
    assert np.array_equal(pred_class, (pos_proba >= threshold).astype(int))
    assert set(np.unique(pred_class)).issubset({0, 1})
