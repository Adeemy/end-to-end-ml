# Inference Module

REST API for serving trained ML models with real-time predictions.

## Quick Start

### Local Deployment in Host Machine

```bash
# Local development (run from project root)
export COMET_API_KEY=your_api_key
uvicorn src.inference.main:app --host 0.0.0.0 --port 8000

# Test
curl http://localhost:8000/
```

### Docker Deployment (Host Machine Only)

**Note**: Docker is not available inside devcontainers. Run these commands on your host machine:

```bash
# Exit devcontainer first, then run on host:
docker build -t ml-inference -f src/inference/Dockerfile .
docker run -p 8000:8000 -e COMET_API_KEY=your_api_key ml-inference
```

## API Usage

### Health Check

```bash
curl http://localhost:8000/
# Response: {"message": "Diabetes Prediction API"}
```

### Make Predictions

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "BMI": 29.0,
    "PhysHlth": 0,
    "Age": "65 to 69",
    "HighBP": "1",
    "HighChol": "1",
    "CholCheck": "1",
    "Smoker": "0",
    "Stroke": "0",
    "HeartDiseaseorAttack": "0",
    "PhysActivity": "1",
    "Fruits": "1",
    "Veggies": "1",
    "HvyAlcoholConsump": "0",
    "AnyHealthcare": "1",
    "NoDocbcCost": "0",
    "GenHlth": "Fair",
    "MentHlth": 5,
    "DiffWalk": "0",
    "Sex": "Female",
    "Education": "High School",
    "Income": "2"
  }'
```

## Configuration

Set required environment variables:

```bash
export COMET_API_KEY=your_comet_api_key
```

The system loads model registry details from `./src/config/training-config.yml`.

## Troubleshooting

**Common Issues:**

- **Config file not found**: Run from project root (`/workspaces/end-to-end-ml`)
- **COMET_API_KEY not set**: Export the environment variable
- **Model loading errors**: Verify Comet ML credentials and champion model exists
