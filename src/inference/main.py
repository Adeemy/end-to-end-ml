"""
This script wraps an API endpoint around a model to score
production data via API calls.
"""

import os

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI
from fastapi.responses import HTMLResponse
from pydantic import create_model
from utils import _download_model, get_config_params

from training.utils.path import ARTIFACTS_DIR, PARENT_DIR

# ########################################################
# # Comet API key, project and workspace names
# # Note: importing Comet API keu from local file is
# # only needed during dev on a local machine but not
# # in CI/CD setting, e.g., GitHub Actions.
# import json
# from pathlib import Path

# with open(
#     Path(__file__).parent.resolve().parent.parent / ".comet.json", encoding="utf-8"
# ) as file:
#     contents = json.load(file)
# os.environ["COMET_API_KEY"] = contents["COMET_API_KEY"]

# # Import prod data for scoring
# from datasets import load_dataset
# hf_data_source = "Bena345/cdc-diabetes-health-indicators"
# dataset = load_dataset(hf_data_source, data_files = "inference.parquet")
# prod_data = dataset["train"].to_pandas()
# input_data = prod_data[0:][:1].to_json(orient="records")
data = {
    "BMI": 29.0,
    "PhysHlth": 0,
    "Age": "65 to 69",
    "HighBP": "0",
    "HighChol": "1",
    "CholCheck": "0",
    "Smoker": "1",
    "Stroke": "1",
    "HeartDiseaseorAttack": "0",
    "PhysActivity": "1",
    "Fruits": "1",
    "Veggies": "1",
    "HvyAlcoholConsump": "1",
    "AnyHealthcare": "1",
    "NoDocbcCost": "1",
    "GenHlth": "Poor",
    "MentHlth": "1",
    "DiffWalk": "1",
    "Sex": "1",
    "Education": "1",
    "Income": "7"
}

# Extracts config params
(
    comet_ws,
    champ_model_name,
    num_feature_names,
    cat_feature_names,
    hf_data_source,
) = get_config_params(
    config_yaml_abs_path=str(PARENT_DIR.parent) + "/config/training/config.yml"
)

# Define data model for the input received from the request body
# Note: this setup if flexible as it takes into account dynamic list
# of numerical and categorical features.
num_field_names = {feature: (float, ...) for feature in num_feature_names}
cat_field_names = {feature: (str, ...) for feature in cat_feature_names}
feature_names = {**num_field_names, **cat_field_names}
Data = create_model("Data", **feature_names)

# Data.schema_json(indent=2)

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

# @app.post("/predict")
# def predict(data: Data = Body(...)):
    
#     data = np.array(data).reshape(1, -1)
#     preds = model.predict_proba(data)[0]
    
#     return {"Predicted Probability": round(preds[1], 2)}

@app.post("/predict")
def predict(data: dict = Body(...)):
    
    # Convert the dictionary values to a list
    data_list = list(data.values())
    
    # Reshape the list to a 2D array
    data_array = np.array(data_list).reshape(1, -1)
    
    # Convert to pandas dataframe
    data_df = pd.DataFrame(data_array)
    
    data_df.columns = [
        "BMI",
        "PhysHlth",
        "Age",
        "HighBP",
        "HighChol",
        "CholCheck",
        "Smoker",
        "Stroke",
        "HeartDiseaseorAttack",
        "PhysActivity",
        "Fruits",
        "Veggies",
        "HvyAlcoholConsump",
        "AnyHealthcare",
        "NoDocbcCost",
        "GenHlth",
        "MentHlth",
        "DiffWalk",
        "Sex",
        "Education",
        "Income",
    ]
        
    # Predict the probability using the model
    preds = model.predict_proba(data_df)[0]
    return {"Predicted Probability": round(preds[1], 2)}