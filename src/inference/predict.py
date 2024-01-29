"""
This script wraps an API endpoint around a model to score
production data via API calls.
"""

import argparse
import os

from dotenv import load_dotenv

from src.inference.utils.model import ModelLoader, predict
from src.utils.path import ARTIFACTS_DIR

load_dotenv()

########################################################


def main(config_yaml_path: str, api_key: str, input_data: dict) -> None:
    """Loads the champion model and scores the input data.

    Args:
        config_yaml_path (str): path to the config yaml file.
        api_key (str): Comet API key.
        input_data (dict): dictionary containing the input data.

    Returns:
        None.
    """

    # Extracts config params
    load_model = ModelLoader()
    (
        comet_ws,
        champ_model_name,
        *_,
    ) = load_model.get_config_params(config_yaml_abs_path=config_yaml_path)

    # Download champion model
    model = load_model.download_model(
        comet_workspace=comet_ws,
        comet_api_key=api_key,
        model_name=champ_model_name,
        artifacts_path=ARTIFACTS_DIR,
    )

    prediction = predict(model, input_data)
    print(f"\n\n\n{prediction=}\n\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml_path",
        type=str,
        default="./config.yml",
        help="Path to the config yaml file.",
    )

    args = parser.parse_args()

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

    main(
        config_yaml_path=args.config_yaml_path,
        api_key=os.environ["COMET_API_KEY"],
        input_data=sample_data,
    )
