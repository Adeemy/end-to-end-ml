# Inference Module

This module handles model serving for trained machine learning models. It provides a REST API for making predictions on new data using the champion model.

## Overview

The inference module provides:
- REST API for real-time predictions
- Model loading from Comet ML registry
- Input validation and preprocessing
- Dockerized deployment support

## Workflow

### 1. Model Deployment
```bash
# Local development
cd src/inference
uvicorn --host 0.0.0.0 main:app

# Docker deployment
docker build -t ml-inference -f src/inference/Dockerfile .
docker run -p 8000:8000 ml-inference
```

### 2. Making Predictions
```bash
# Root endpoint (health check)
curl http://localhost:8000/

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## Directory Structure

```
inference/
├── README.md              # This file
├── main.py               # FastAPI application entry point
├── predict.py            # Command-line prediction script
├── Dockerfile            # Docker container configuration
└── utils/                # Inference utilities
    ├── __init__.py
    └── model.py          # Model loading and prediction utilities
```

## Key Components

### FastAPI Application (`main.py`)
Main application server providing REST API for predictions.

**Endpoints**:
- `GET /` - Root endpoint returning HTML health message
- `POST /predict` - Single instance prediction

**Features**:
- Downloads champion model from Comet ML on startup
- Converts input JSON to DataFrame for model prediction
- Returns probability prediction for positive class

### Command-line Prediction (`predict.py`)
Standalone script for making predictions without starting a server.

**Usage**:
```python
python src/inference/predict.py \
  --config_yaml_path ./src/config/training-config.yml \
  --api_key YOUR_COMET_API_KEY \
  --input_data '{"BMI": 29.0, "Age": "65 to 69", ...}'
```

### Model Management (`utils/model.py`)
Handles model downloading from Comet ML registry and prediction logic.

**Key Classes**:
- `ModelLoader`: Downloads champion models from Comet ML
- `predict()`: Function for making predictions with loaded model

## API Reference

### Root Endpoint
**Endpoint**: `GET /`

**Response**:
```html
<h1>Predict pre-diabetes/diabetes.</h1>
```

### Prediction Endpoint
**Endpoint**: `POST /predict`

**Request Body** (all fields required):
```json
{
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
```

**Response**:
```json
{
  "Predicted Probability": 0.856
}
```

## Configuration

Inference behavior is controlled through environment variables and configuration files:

```bash
# Required environment variable
export COMET_API_KEY=your_comet_api_key

# Configuration file
CONFIG_PATH=./src/config/training-config.yml
```

The system loads configuration from `training-config.yml` to extract:
- Comet ML workspace name
- Champion model name
- Feature column definitions

## Usage Examples

### Loading and Using Model Programmatically
```python
from src.inference.utils.model import ModelLoader, predict

# Initialize model loader
loader = ModelLoader(comet_api_key="your_api_key")

# Get configuration
workspace, model_name, *_ = loader.get_config_params(
    config_yaml_abs_path="./src/config/training-config.yml"
)

# Download model
model = loader.download_model(
    comet_workspace=workspace,
    model_name=model_name,
    artifacts_path="./artifacts"
)

# Make prediction
input_data = {
    "BMI": 29.0,
    "Age": "65 to 69",
    # ... other features
}

prediction = predict(model, input_data)
print(prediction)  # {"Predicted Probability": 0.856}
```

### Command-line Prediction
```python
from src.inference.predict import main
from src.utils.logger import get_console_logger

logger = get_console_logger("inference")

input_data = {
    "BMI": 29.0,
    "PhysHlth": 0,
    "Age": "65 to 69",
    # ... complete feature set
}

main(
    config_yaml_path="./src/config/training-config.yml",
    api_key="your_comet_api_key",
    input_data=input_data,
    logger=logger
)
```

## Deployment

### Local Development
```bash
# Install dependencies (ensure you have requirements.txt)
pip install -r requirements.txt

# Set API key
export COMET_API_KEY=your_comet_api_key

# Navigate to inference directory and run
cd src/inference
uvicorn --host 0.0.0.0 main:app
```

### Docker Deployment
```bash
# Build image (from project root)
docker build -t ml-inference -f src/inference/Dockerfile .

# Run container
docker run -d \
  --name diabetes-predictor \
  -p 8000:8000 \
  -e COMET_API_KEY=your_comet_api_key \
  ml-inference
```

### Model Requirements
The inference system requires:
1. A champion model registered in Comet ML
2. Model saved as `champion_model.pkl`
3. Valid Comet ML API key with access to the workspace
4. Configuration file with model registry settings

## Dependencies

Key dependencies for the inference module:
- **fastapi**: Web framework for API endpoints
- **uvicorn**: ASGI server for running FastAPI
- **pandas**: Data manipulation for input processing
- **scikit-learn**: Model pipeline support
- **joblib**: Model serialization/deserialization
- **comet-ml**: Model registry integration
- **python-dotenv**: Environment variable management

## Model Loading Process

1. **Configuration Loading**: Reads training config to get model registry details
2. **Comet ML Connection**: Creates API connection using provided key
3. **Model Download**: Downloads latest version of champion model
4. **Model Loading**: Loads pickled model using joblib
5. **Ready for Inference**: Model available for predictions

## Input Format

The prediction endpoint expects input data matching the training feature schema:
- **Numerical features**: `BMI`, `PhysHlth`, `MentHlth`
- **Categorical features**: `Age`, `HighBP`, `HighChol`, etc.
- **String values**: Categorical features should be strings matching training categories
- **Complete feature set**: All training features must be provided

## Error Handling

Common errors and solutions:
- **Comet ML authentication failure**: Check API key validity
- **Model not found**: Verify champion model exists in registry
- **Invalid input format**: Ensure all required features are provided
- **Prediction errors**: Check input data types and ranges

## Troubleshooting

### Common Issues
1. **Model Loading Errors**: Verify Comet ML credentials and model registry
2. **Missing Features**: Ensure input contains all required feature columns
3. **Type Errors**: Check that categorical features are strings, numerical are numbers
4. **API Connection**: Verify network connectivity to Comet ML

### Debugging Tips
- Check logs for model loading status
- Validate input data format against training schema
- Test Comet ML connection independently
- Verify model file exists in artifacts directory



## Usage Examples

### Loading and Using Model
```python
from src.inference.utils.model import ModelManager

# Initialize model manager
model_manager = ModelManager(
    model_path="artifacts/lightgbm.pkl"
)

# Load model
model = model_manager.load_model()

# Make prediction
from src.inference.predict import make_prediction

prediction = make_prediction(
    features=[1.0, 2.0, 3.0, 4.0],
    model=model
)
```



## Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export MODEL_PATH=src/training/artifacts/lightgbm.pkl
export HOST=0.0.0.0
export PORT=8000

# Run application
python src/inference/main.py
```

### Docker Deployment
```bash
# Build image
docker build -t ml-inference -f src/inference/Dockerfile .

# Run container
docker run -d \
  --name ml-api \
  -p 8000:8000 \
  -v $(pwd)/src/training/artifacts:/app/models \
  -e MODEL_PATH=/app/models/lightgbm.pkl \
  ml-inference

# Check logs
docker logs ml-api
```



## Troubleshooting

### Common Issues
1. **Model Loading Errors**: Check file paths and permissions
2. **Memory Issues**: Monitor memory usage and model size
3. **Timeout Errors**: Adjust timeout settings or optimize preprocessing
4. **Input Validation**: Ensure input format matches training data

### Debugging Tips
- Check logs for model loading status
- Validate input data format against training schema
- Test Comet ML connection independently
- Verify model file exists in artifacts directory
