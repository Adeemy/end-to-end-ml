"""
FastAPI-based model serving service for real-time predictions.

This module provides a REST API service that loads trained ML models from
experiment tracking backends (MLflow/Comet ML) and serves predictions via
HTTP endpoints. Supports both champion model loading and direct model
specification for production inference.

Endpoints:
    GET /: Health check and service information
    POST /predict: Generate predictions from input features

The service automatically handles model loading, feature preprocessing,
and returns JSON-formatted prediction results. The champion model is loaded
from tracking system Model Registry, with a local fallback to the path
./src/training/artifacts/champion_model.pkl only if the registry is unavailable.
"""

import os
from pathlib import PosixPath

import pandas as pd
from dotenv import load_dotenv
from fastapi import Body, FastAPI
from fastapi.responses import HTMLResponse

from src.inference.utils.model import ModelLoaderManager
from src.utils.logger import get_console_logger
from src.utils.path import ARTIFACTS_DIR, PARENT_DIR

load_dotenv()


module_name: str = PosixPath(__file__).stem
logger = get_console_logger(module_name)

########################################################
# # Sample of prod data for testing
# data = {
#     "BMI": 29.0,
#     "PhysHlth": 0,
#     "Age": "65 to 69",
#     "HighBP": "0",
#     "HighChol": "1",
#     "CholCheck": "0",
#     "Smoker": "1",
#     "Stroke": "1",
#     "HeartDiseaseorAttack": "0",
#     "PhysActivity": "1",
#     "Fruits": "1",
#     "Veggies": "1",
#     "HvyAlcoholConsump": "1",
#     "AnyHealthcare": "1",
#     "NoDocbcCost": "1",
#     "GenHlth": "Poor",
#     "MentHlth": "1",
#     "DiffWalk": "1",
#     "Sex": "1",
#     "Education": "1",
#     "Income": "7"
# }

# Extracts config params
load_model = ModelLoaderManager(comet_api_key=os.environ["COMET_API_KEY"])
(
    tracker_type,
    comet_ws,
    champ_model_name,
    *_,
) = load_model.get_config_params(
    config_yaml_abs_path=f"{str(PARENT_DIR.parent)}/src/config/training-config.yml"
)

# Load champion model
model = load_model.load_model(
    tracker_type=tracker_type,
    model_name=champ_model_name,
    artifacts_path=ARTIFACTS_DIR,
    workspace_name=comet_ws,
)

# Root to ./src/inference and run "uvicorn --host 0.0.0.0 main:app"
app = FastAPI()


@app.get("/")
def root():
    return HTMLResponse("<h1>Predict pre-diabetes/diabetes.</h1>")


@app.post("/predict")
def predict(data: dict = Body(...)):
    """Predicts the probability of having a heart disease or stroke.

    Args:
        data (dict): dictionary containing the input data.

    Returns:
        dict: dictionary containing the predicted probability.
    """

    # Convert input dictionary to data frame required by the model
    data_df = pd.json_normalize(data)

    # Predict the probability using the model
    preds = model.predict_proba(data_df)[0]
    return {"Predicted Probability": round(preds[1], 3)}
