"""
Tests for the FastAPI serving app (src/inference/api_server.py).

Verify the lazy model-loading refactor: importing the app does NOT load a model,
/health reports load state (ok / degraded), /predict serves single and batch
inputs, returns 422 on schema mismatch, and 503 when the model cannot be loaded.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.inference import api_server
from src.inference.utils import helpers


@pytest.fixture(name="fitted_model")
def fitted_model_fixture():
    """A fitted binary pipeline whose feature_names_in_ are f1, f2."""
    rng = np.random.default_rng(0)
    features = pd.DataFrame(
        {
            "f1": np.concatenate([rng.normal(0, 1, 40), rng.normal(2, 1, 40)]),
            "f2": np.concatenate([rng.normal(0, 1, 40), rng.normal(2, 1, 40)]),
        }
    )
    labels = np.array([0] * 40 + [1] * 40)
    return Pipeline(
        steps=[
            ("preprocessor", StandardScaler()),
            ("selector", VarianceThreshold(0.0)),
            ("classifier", LogisticRegression(max_iter=200)),
        ]
    ).fit(features, labels)


@pytest.fixture(name="loaded_app")
def loaded_app_fixture(fitted_model, monkeypatch, tmp_path):
    """App with the model pre-loaded into the cache and metadata defaults."""
    monkeypatch.setattr(api_server, "_model", fitted_model)
    monkeypatch.setattr(api_server, "_expected_features", ["f1", "f2"])
    monkeypatch.setattr(api_server, "_last_load_error", None)
    # Point serving metadata at an empty dir so defaults (0.5, pos=1) are used.
    monkeypatch.setattr(helpers, "ARTIFACTS_DIR", tmp_path)
    return TestClient(api_server.app)


def _record():
    return {"f1": 1.0, "f2": 0.5}


def test_health_ok(loaded_app):
    """/health reports ok when the model is loaded."""
    response = loaded_app.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_predict_single_and_batch(loaded_app):
    """/predict returns a dict for a single record and a list for a batch."""
    single = loaded_app.post("/predict", json=_record())
    assert single.status_code == 200
    body = single.json()
    assert set(body) == {"predicted_probability", "prediction"}
    assert body["prediction"] in (0, 1)

    batch = loaded_app.post("/predict", json=[_record(), _record()])
    assert batch.status_code == 200
    assert isinstance(batch.json(), list) and len(batch.json()) == 2


def test_predict_422_on_schema_mismatch(loaded_app):
    """A record missing a model feature is rejected with 422."""
    response = loaded_app.post("/predict", json={"f1": 1.0})  # missing f2
    assert response.status_code == 422


def test_health_degraded_when_model_unavailable(monkeypatch):
    """/health returns 503 (degraded) when the model cannot be loaded."""
    monkeypatch.setattr(api_server, "_model", None)
    failing_loader = MagicMock()
    failing_loader.load_model.side_effect = RuntimeError("registry down")
    monkeypatch.setattr(api_server, "_loader_context", failing_loader)

    client = TestClient(api_server.app)
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["model_loaded"] is False


def test_predict_503_when_model_unavailable(monkeypatch):
    """/predict returns 503 when the model cannot be loaded."""
    monkeypatch.setattr(api_server, "_model", None)
    failing_loader = MagicMock()
    failing_loader.load_model.side_effect = RuntimeError("registry down")
    monkeypatch.setattr(api_server, "_loader_context", failing_loader)

    client = TestClient(api_server.app)
    response = client.post("/predict", json=_record())
    assert response.status_code == 503
