"""Scoring script for the champion managed online endpoint.

Azure ML calls ``init()`` once on startup and ``run(raw_data)`` per request. The
champion is an MLflow pyfunc model (a calibrated pipeline including preprocessing),
so it takes raw feature rows directly - the same contract as the local FastAPI
service.

Real-time feature-store option: instead of expecting full feature rows in the
request, ``run`` could take entity ids and fetch online features via the feature
store's online retrieval before predicting. Kept request-carries-features here to
match the existing serving contract; see docs/azure-ml-refactor-plan.md.
"""

import json
import os

import mlflow.pyfunc
import pandas as pd

_model = None


def init() -> None:
    """Loads the champion model from the deployment's model directory."""
    global _model  # pylint: disable=global-statement
    model_dir = os.environ["AZUREML_MODEL_DIR"]
    # The registered MLflow model is under AZUREML_MODEL_DIR; if it sits in a
    # subfolder, locate the directory that contains an MLmodel file.
    model_path = model_dir
    for root, _dirs, files in os.walk(model_dir):
        if "MLmodel" in files:
            model_path = root
            break
    _model = mlflow.pyfunc.load_model(model_path)


def run(raw_data: str):
    """Scores a JSON payload (a record or list of records) and returns predictions."""
    payload = json.loads(raw_data)
    records = payload if isinstance(payload, list) else [payload]
    frame = pd.DataFrame(records)
    predictions = _model.predict(frame)
    return (
        predictions.tolist() if hasattr(predictions, "tolist") else list(predictions)
    )
