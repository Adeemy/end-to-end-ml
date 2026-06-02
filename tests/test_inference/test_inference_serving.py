"""
Regression guards for inference-time serving metadata and positive-class decode.

    #5  The persisted decision threshold is loaded and applied at inference.
    #7  The positive-class column is decoded via classes_, not hardcoded index 1.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.inference.utils import helpers
from src.inference.utils.model import (
    _positive_class_index,  # pylint: disable=protected-access
)


def test_positive_class_index_decodes_via_classes():
    """_positive_class_index uses classes_, so a reversed order still resolves."""

    class _Stub:
        classes_ = np.array([1, 0])  # positive label 1 lives in column 0

    assert _positive_class_index(_Stub(), pos_label=1) == 0

    class _Normal:
        classes_ = np.array([0, 1])

    assert _positive_class_index(_Normal(), pos_label=1) == 1


def test_load_serving_metadata_defaults(monkeypatch, tmp_path):
    """Missing metadata falls back to default threshold / positive label."""
    monkeypatch.setattr(helpers, "ARTIFACTS_DIR", tmp_path)
    metadata = helpers.load_serving_metadata("missing_model")
    assert metadata["decision_threshold"] == helpers.DEFAULT_DECISION_THRESHOLD
    assert (
        metadata["encoded_pos_class_label"] == helpers.DEFAULT_ENCODED_POS_CLASS_LABEL
    )


def test_positive_class_predictions_applies_persisted_threshold(monkeypatch, tmp_path):
    """Inference reads the persisted threshold and decodes the positive class."""
    rng = np.random.default_rng(0)
    n = 60
    features = pd.DataFrame(
        {
            "f1": np.concatenate([rng.normal(0, 1, n), rng.normal(1.5, 1, n)]),
            "f2": np.concatenate([rng.normal(0, 1, n), rng.normal(1.5, 1, n)]),
        }
    )
    labels = np.array([0] * n + [1] * n)
    model = Pipeline(
        steps=[
            ("preprocessor", StandardScaler()),
            ("selector", VarianceThreshold(0.0)),
            ("classifier", LogisticRegression(max_iter=200)),
        ]
    ).fit(features, labels)

    # Persist a non-default threshold and point inference at the temp dir.
    monkeypatch.setattr(helpers, "ARTIFACTS_DIR", tmp_path)
    threshold = 0.7
    (tmp_path / "svc_metadata.json").write_text(
        '{"decision_threshold": 0.7, "encoded_pos_class_label": 1}',
        encoding="utf-8",
    )

    pos_proba, pred_class, applied = helpers.positive_class_predictions(
        model, features, "svc"
    )
    pos_col = list(model.classes_).index(1)
    expected_proba = model.predict_proba(features)[:, pos_col]

    assert applied == pytest.approx(threshold)
    assert np.allclose(pos_proba, expected_proba)
    assert np.array_equal(pred_class, (expected_proba >= threshold).astype(int))
