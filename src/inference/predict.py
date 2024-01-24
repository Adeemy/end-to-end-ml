"""
This script wraps an API endpoint around a model to score
production data via API calls.
"""

import os

import pandas as pd
from dotenv import load_dotenv
from fastapi import Body

from src.config.path import ARTIFACTS_DIR, PARENT_DIR
from src.inference.utils.model import ModelLoader

load_dotenv()

########################################################
# Extracts config params
load_model = ModelLoader()
(
    comet_ws,
    champ_model_name,
    *_,
) = load_model.get_config_params(
    config_yaml_abs_path=str(PARENT_DIR.parent) + "/src/config/training/config.yml"
)

# Download champion model
model = load_model.download_model(
    comet_workspace=comet_ws,
    comet_api_key=os.environ["COMET_API_KEY"],
    model_name=champ_model_name,
    artifacts_path=ARTIFACTS_DIR,
)


def predict(data: dict = Body(...)):
    # Convert input dictionary to data frame required by the model
    data_df = pd.json_normalize(data)

    # Predict the probability using the model
    preds = model.predict_proba(data_df)[0]
    return {"Predicted Probability": round(preds[1], 3)}


if __name__ == "__main__":
    # Sample of prod data for testing
    sample_data = {
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
        "Income": "7",
    }

    prediction = predict(sample_data)
    print(f"\n\n\n{prediction=}\n\n\n")
