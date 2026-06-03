"""
Guards for two latent bugs found in review:

- `ClassifierEnsembleOrchestrator.get_base_models` used `model` before it was
  assigned, so base-estimator names were wrong (and the first pipeline failed
  silently). It must return each classifier paired with its own class name.
- The training-config builder read model-section keys that did not match the
  YAML (`logistic_regression` vs `logisticregression`, `lightgbm` vs `lgbm`), so
  those config sections silently loaded empty.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.training.core.ensemble import ClassifierEnsembleOrchestrator
from src.training.schemas import Config, build_training_config
from src.utils.config_loader import load_config
from src.utils.path import PARENT_DIR


def _fitted_pipeline(classifier):
    """Fits a minimal preprocessor + classifier pipeline on toy data."""
    features = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0], "f2": [1.0, 0.0, 1.0, 0.0]})
    labels = np.array([0, 1, 0, 1])
    return Pipeline(
        steps=[("preprocessor", StandardScaler()), ("classifier", classifier)]
    ).fit(features, labels)


def test_get_base_models_uses_each_classifier_name():
    """Each base model is paired with its OWN classifier class name."""
    pipelines = [
        _fitted_pipeline(LogisticRegression(max_iter=100)),
        _fitted_pipeline(RandomForestClassifier(n_estimators=5)),
    ]
    orchestrator = ClassifierEnsembleOrchestrator(
        experiment_manager=MagicMock(),
        train_features=pd.DataFrame({"f1": [0.0]}),
        valid_features=pd.DataFrame({"f1": [0.0]}),
        train_class=np.array([0]),
        valid_class=np.array([0]),
        class_encoder=MagicMock(),
        artifacts_path="",
        supported_models=MagicMock(),
        base_pipelines=pipelines,
    )

    base_models = orchestrator.get_base_models()
    names = [name for name, _ in base_models]
    assert names == ["LogisticRegression", "RandomForestClassifier"]
    # Each tuple pairs the right name with the right estimator instance.
    for name, model in base_models:
        assert type(model).__name__ == name


def test_config_model_sections_load():
    """The per-model config sections actually populate (no key drift)."""
    config_path = f"{str(PARENT_DIR)}/config/training-config.yml"
    cfg = load_config(Config, build_training_config, config_path)

    assert cfg.logistic_regression.params is not None
    assert cfg.random_forest.params is not None
    assert cfg.lightgbm.params is not None
    # A representative value is read through, not just a non-empty dict.
    assert cfg.lightgbm.params.get("objective") == "binary"
