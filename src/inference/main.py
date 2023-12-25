"""
This script wraps an API endpoint around a model to score
production data via API calls.
"""

import os

import pandas as pd
from dotenv import load_dotenv  # pylint: disable=W0611
from fastapi import Body, FastAPI
from fastapi.responses import HTMLResponse
from utils import _download_model, get_config_params

from training.utils.path import ARTIFACTS_DIR, PARENT_DIR

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
(
    comet_ws,
    champ_model_name,
    *_,
) = get_config_params(
    config_yaml_abs_path=str(PARENT_DIR.parent) + "/config/training/config.yml"
)

# Download champion model
model = _download_model(
    comet_workspace=comet_ws,
    comet_api_key=os.environ["COMET_API_KEY"],
    model_name=champ_model_name,
    artifacts_path=ARTIFACTS_DIR,
)

# Root to ./src/inference and run "uvicorn --host 0.0.0.0 main:app"
app = FastAPI()


@app.get("/")
def root():
    return HTMLResponse("<h1>Predict pre-diabetes/diabetes.</h1>")


@app.post("/predict")
def predict(data: dict = Body(...)):
    # Convert input dictionary to data frame required by the model
    data_df = pd.json_normalize(data)

    # Predict the probability using the model
    preds = model.predict_proba(data_df)[0]
    return {"Predicted Probability": round(preds[1], 3)}
