"""
FastAPI-based model serving service for real-time predictions.

This module provides a REST API service that loads trained ML models from
experiment tracking backends (e.g., MLflow/Comet ML) and serves predictions via
HTTP endpoints. Supports both champion model loading and direct model
specification for production inference.

Endpoints:
    GET /: Health check and service information
    POST /predict: Generate predictions from input features (single sample or batch)

The service automatically handles model loading, feature preprocessing,
and returns JSON-formatted prediction results. Accepts both single samples
and batch data in JSON format.

Usage Examples:
    # Single sample via CLI (run from project root)
    python src/inference/api_server.py --input_data '{"BMI": 29.0, "PhysHlth": 0, "Age": "65 to 69", "HighBP": "0", "HighChol": "1", "CholCheck": "0", "Smoker": "1", "Stroke": "1", "HeartDiseaseorAttack": "0", "PhysActivity": "1", "Fruits": "1", "Veggies": "1", "HvyAlcoholConsump": "1", "AnyHealthcare": "1", "NoDocbcCost": "1", "GenHlth": "Poor", "MentHlth": "1", "DiffWalk": "1", "Sex": "1", "Education": "1", "Income": "7"}'

    # Batch samples via CLI
    python src/inference/api_server.py --input_data '[{"BMI": 29.0, "PhysHlth": 0, ...}, {"BMI": 25.0, "PhysHlth": 2, ...}]'

    # Start API server
    uvicorn src.inference.api_server:app --host 0.0.0.0 --port 8000

    # API endpoint (single sample)
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"BMI": 29.0, "PhysHlth": 0, "Age": "65 to 69", ...}'

    # API endpoint (batch samples)
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '[{"BMI": 29.0, ...}, {"BMI": 25.0, ...}]'
"""

import argparse
import json
import os
from pathlib import PosixPath
from typing import List, Union

import pandas as pd
from dotenv import load_dotenv
from fastapi import Body, FastAPI
from fastapi.responses import HTMLResponse

from src.inference.utils.helpers import (
    LocalModelLoader,
    ModelLoadingContext,
    RegistryModelLoader,
    extract_model_config,
)
from src.utils.logger import get_console_logger
from src.utils.path import PARENT_DIR

load_dotenv()


module_name: str = PosixPath(__file__).stem
logger = get_console_logger(module_name)

# Load champion model at startup using strategy pattern
CONFIG_PARAMS = extract_model_config(
    config_yaml_path=f"{str(PARENT_DIR.parent)}/src/config/training-config.yml"
)

# Required API keys for model loading
COMET_API_KEY = os.environ.get("COMET_API_KEY")

# Create loading strategies with fallback
primary_strategy = RegistryModelLoader()
fallback_strategy = LocalModelLoader()
loader_context = ModelLoadingContext(primary_strategy, fallback_strategy)

model = loader_context.load_model(
    model_name=CONFIG_PARAMS["model_name"],
    config_params=CONFIG_PARAMS,
    comet_api_key=COMET_API_KEY,
    logger=logger,
)

# FastAPI app
app = FastAPI()


@app.get("/")
def root():
    return HTMLResponse("<h1>Predict pre-diabetes/diabetes.</h1>")


@app.post("/predict")
def predict(data: Union[dict, List[dict]] = Body(...)):
    """Predicts the probability of having a heart disease or stroke.

    Args:
        data (Union[dict, List[dict]]): Single sample dict or list of sample dicts.

    Returns:
        Union[dict, List[dict]]: Single prediction or list of predictions.
    """
    # Handle both single sample and batch processing
    is_single_sample = isinstance(data, dict)

    # Convert to list format for uniform processing
    if is_single_sample:
        data = [data]

    # Convert input to DataFrame
    data_df = pd.DataFrame(data)

    # Generate predictions
    predictions = model.predict_proba(data_df)

    # Format results
    results = []
    for _, pred in enumerate(predictions):
        result = {
            "predicted_probability": round(pred[1], 3),
            "prediction": 1 if pred[1] > 0.5 else 0,
        }
        results.append(result)

    # Return single result or batch results based on input
    return results[0] if is_single_sample else results


def cli_predict(input_data: Union[str, dict, List[dict]]) -> Union[dict, List[dict]]:
    """CLI interface for predictions.

    Args:
        input_data: JSON string, single dict, or list of dicts containing sample data.

    Returns:
        Prediction results.
    """

    # Parse input data if it's a JSON string
    if isinstance(input_data, str):
        try:
            input_data = json.loads(input_data)
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON input: %s", e)
            raise ValueError(f"Invalid JSON input: {e}") from e

    # Process prediction
    results = predict(input_data)

    logger.info("Input Data: %s", input_data)
    logger.info("Prediction Results: %s", results)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FastAPI model serving with CLI prediction support. Use --input_data for direct CLI predictions, or run without arguments to start the API server."
    )
    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="JSON string containing input data (single sample or array of samples).",
    )

    args = parser.parse_args()

    # Run CLI prediction and print formatted results
    preds = cli_predict(args.input_data)
    print("\nPrediction Results:")
    print("=" * 50)
    print(json.dumps(preds, indent=2))
