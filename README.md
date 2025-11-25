[![CI/CD](https://github.com/Adeemy/end-to-end-ml/actions/workflows/main.yml/badge.svg)](https://github.com/Adeemy/end-to-end-ml/actions/workflows/main.yml)[![python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.99.1-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![codecov](https://codecov.io/gh/Adeemy/end-to-end-ml/graph/badge.svg?token=LO67YZIGXR)](https://codecov.io/gh/Adeemy/end-to-end-ml)

# End-to-end ML

An end-to-end ML project for tabular data that incorporates software engineering principles in machine learning. It spans the whole lifecycle of a ML model, from data exploration, preprocessing, feature engineering, model selection, training, evaluation, to deployment.

The project leverages the Diabetes Health Indicators public dataset from [UCI](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators). The dataset comprises various information about patients, such as demographics, lab results, and self-reported health history. The goal is to develop a classifier that can discern whether a patient has diabetes, is pre-diabetic, or healthy.

The project adheres to best practices of machine learning engineering, such as modular code, documentation, testing, logging, configuration, and version control. The project also demonstrates how to utilize various tools and frameworks, such as pandas, scikit-learn, [feast](https://feast.dev), [optuna](https://optuna.org), experiment tracking using [MLflow](https://mlflow.org) (default) and [Comet](https://www.comet.com/site/), and Docker, to facilitate the ML workflow and enhance the model performance.

## Key Features

- **Configurable Pipeline**: YAML-based configuration for easy parameter tuning
- **Hyperparameter Optimization**: Automated tuning using Optuna
- **Experiment Tracking**: Complete experiment management with MLflow (default) and Comet ML support
- **Feature Engineering**: Feast-based feature store for consistent data processing
- **Model Evaluation**: F-beta score optimization with configurable precision/recall weighting
- **Containerized Deployment**: Docker-based model serving with REST API
- **CI/CD Integration**: GitHub Actions for automated testing and deployment
- **Reproducible Environment**: Dev containers and dependency management with uv

## Project Structure

- **`notebooks/`**: Exploratory data analysis and baseline model development (only directory for notebooks)
- **`src/config/`**: YAML configuration files for data processing and training
- **`src/feature/`**: Data ingestion, preprocessing, and feature engineering
- **`src/training/`**: Model training, hyperparameter optimization, and evaluation
- **`src/inference/`**: Model serving with CLI batch scoring (`predict.py`) and REST API service (`api_server.py`)
- **`src/utils/`**: Shared utilities, logging, and configuration management
- **`tests/`**: Unit tests for ML components and pipeline validation

### ML Pipeline Flow

1. **Data Preparation** (`split_data.py`): Load from feature store → create train/validation/test splits
2. **Training** (`train.py`): Hyperparameter optimization → model training → experiment tracking
3. **Evaluation** (`evaluate.py`): Model selection based on validation set → testign set evaluation → champion registration if meets deployment criteria
4. **Deployment**: Containerized model serving via REST API

Below is the repo structure.

        end-to-end-ml
        ├── LICENSE
        ├── Makefile
        ├── README.md
        ├── dist
        │   ├── end_to_end_ml-0.1.0-py3-none-any.whl
        │   └── end_to_end_ml-0.1.0.tar.gz
        ├── docs
        ├── examples
        ├── img
        │   └── feast_workflow.png
        ├── notebooks
        │   ├── eda.ipynb
        │   └── utils.py
        ├── pyproject.toml
        ├── pytest.ini
        ├── scripts
        ├── src
        │   ├── __init__.py
        │   ├── config
        │   │   ├── feature-store-config.yml
        │   │   ├── logging.conf
        │   │   └── training-config.yml
        │   ├── end_to_end_ml.egg-info
        │   │   ├── PKG-INFO
        │   │   ├── SOURCES.txt
        │   │   ├── dependency_links.txt
        │   │   ├── requires.txt
        │   │   └── top_level.txt
        │   ├── feature
        │   │   ├── README.md
        │   │   ├── __init__.py
        │   │   ├── feature_repo
        │   │   │   ├── data
        │   │   │   │   ├── historical_data.parquet
        │   │   │   │   ├── inference.parquet
        │   │   │   │   ├── online_store.db
        │   │   │   │   ├── preprocessed_dataset_features.parquet
        │   │   │   │   ├── preprocessed_dataset_target.parquet
        │   │   │   │   ├── raw_dataset.parquet
        │   │   │   │   ├── registry.db
        │   │   │   │   ├── test.parquet
        │   │   │   │   ├── train.parquet
        │   │   │   │   └── validation.parquet
        │   │   │   ├── define_feature.py
        │   │   │   └── feature_store.yaml
        │   │   ├── generate_initial_data.py
        │   │   ├── prep_data.py
        │   │   ├── schemas.py
        │   │   └── utils
        │   │       ├── __init__.py
        │   │       ├── data.py
        │   │       └── prep.py
        │   ├── inference
        │   │   ├── Dockerfile
        │   │   ├── README.md
        │   │   ├── __init__.py
        │   │   ├── api_server.py
        │   │   ├── predict.py
        │   │   └── utils
        │   │       ├── __init__.py
        │   │       └── model.py
        │   ├── training
        │   │   ├── README.md
        │   │   ├── __init__.py
        │   │   ├── artifacts
        │   │   │   ├── champion_model.pkl
        │   │   │   └── study_LGBMClassifier.csv
        │   │   ├── core
        │   │   │   ├── __init__.py
        │   │   │   ├── ensemble.py
        │   │   │   ├── optimizer.py
        │   │   │   └── trainer.py
        │   │   ├── evaluate.py
        │   │   ├── evaluation
        │   │   │   ├── __init__.py
        │   │   │   ├── champion.py
        │   │   │   ├── evaluator.py
        │   │   │   ├── orchestrator.py
        │   │   │   ├── selector.py
        │   │   │   └── visualizer.py
        │   │   ├── schemas.py
        │   │   ├── split_data.py
        │   │   ├── tracking
        │   │   │   ├── __init__.py
        │   │   │   ├── experiment.py
        │   │   │   ├── experiment_tracker.py
        │   │   │   └── study_logger.py
        │   │   └── train.py
        │   └── utils
        │       ├── README.md
        │       ├── __init__.py
        │       ├── config_loader.py
        │       ├── logger.py
        │       └── path.py
        ├── tests
        │   ├── __init__.py
        │   ├── test_feature
        │   │   ├── test_data_preprocessor.py
        │   │   ├── test_data_splitter.py
        │   │   ├── test_data_transformer.py
        │   │   └── test_feature_store_config.py
        │   ├── test_inference
        │   │   └── test_inference_model.py
        │   ├── test_training
        │   │   ├── test_data_utils.py
        │   │   ├── test_job.py
        │   │   ├── test_training_config.py
        │   │   └── test_training_model.py
        │   └── test_utils.py
        └── uv.lock

### Environment setup & usage

This project uses [uv](https://github.com/astral-sh/uv) to manage dependencies, which are defined in the `pyproject.toml` file. A devcontainer is configured to set up a full-featured development environment and install required dependencies in addition to some useful VS Code extensions. It allows isolating the tools, libraries, and runtimes needed for working with this project codebase, and to use VS Code’s full feature set inside the container. A devcontainer requires [Docker](https://docs.docker.com/engine/install/) to be up and running. It's recommended to use the devcontainer for this project to ensure consistency and reproducibility across different machines and platforms, but if not desired, for whatever reason, you can create a virtual environment and install the python dependencies by running the following commands from the project root:

    python3.10 -m venv .venv
    source .venv/bin/activate
    make install

#### Environment Variables

**MLflow** is used as the default experiment tracker and requires no additional configuration.

When using **Comet ML as the experiment tracker**, set these environment variables:

    COMET_API_KEY=your_comet_api_key
    ENABLE_COMET_LOGGING=true

The `ENABLE_COMET_LOGGING` variable ensures proper import order for automatic logging. For MLflow, these variables are not needed.

Copy `.env_template` to `.env` and add your API keys:

    cp .env_template .env
    # Edit .env with your actual API keys

#### Pipeline Commands

The training and deployment pipelines can be run in GitHub Actions. You can also run the following commands in CLI to implement all steps from generating raw dataset to pulling packaged model:

- Import raw dataset from [UCI](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) and generate raw dataset for training and inference set (5% holdout) to simulate production data

        make gen_init_data

- Preprocess data before ingesting it by feature store

        make prep_data

- Setup feature store

        make setup_feast

- Split dataset extracted from feature store

        make split_data

- Submit training job

        make train

- Evaluate models

        make evaluate

**Champion Model Registration:**
The evaluation process automatically:
- Selects the best performing model based on validation metrics from both training and evaluation runs
- Registers it as "champion_model" in the experiment tracker's model registry with complete artifacts
- Saves it locally as `src/training/artifacts/champion_model.pkl` for deployment
- Includes all necessary files: model.pkl, MLmodel metadata, environment specs, and dependencies
- Ensures the model meets deployment criteria before registration
- Provides production-ready model accessible via `models:/champion_model/latest`
- Both training and evaluation runs log complete model artifacts for full traceability

- **Test champion model via CLI (batch scoring)**

        make test_model_cli

- **Start REST API server for real-time predictions**

        make start_api_server
        # Then visit http://localhost:8000/docs to test the API

- **Test API server with sample data**

        # Option 1: Test against running server
        make test_api_with_sample

        # Option 2: Full end-to-end test (start → test → stop)
        make test_api_full

        # Show all API testing options
        make help_api

- **Pull containerized model**

        docker pull ghcr.io/adeemy/end-to-end-ml:c35fb9610651e155d7a3799644e6ff64c1a5a2db

## Experiment Tracking

This project supports both Comet ML and MLflow for experiment tracking. You can switch between them by modifying the `experiment_tracker` parameter in `src/config/training-config.yml`.

### Using Comet ML

Set `experiment_tracker: "comet"` in the training configuration. Experiments will be tracked in your Comet workspace with the configured project name.

### Using MLflow

Set `experiment_tracker: "mlflow"` in the training configuration.

#### Viewing MLflow Dashboard

To view the MLflow web interface and track your experiments:

1. **Start MLflow UI server:**
   ```bash
   mlflow ui --host 0.0.0.0 --port 8080
   ```

2. **If the dashboard appears blank, try with additional security options:**
   ```bash
   mlflow ui --host 0.0.0.0 --port 8080 --allowed-hosts "*"
   ```

3. **For persistent background running:**
   ```bash
   nohup mlflow ui --host 0.0.0.0 --port 8080 --allowed-hosts "*" > mlflow.log 2>&1 &
   ```

4. **Access the dashboard:**
   - **Local development:** Open http://localhost:8080 in your browser
   - **Dev container/Codespaces:** Use VS Code's port forwarding or `"$BROWSER" http://localhost:8080`
   - **Remote server:** Replace `localhost` with your server's IP address

5. **Dashboard features:**
   - View all experiments and runs
   - Compare metrics and parameters across runs
   - Download model artifacts
   - Visualize experiment results and trends

**Troubleshooting blank dashboard:**
- **Check if server is running**: `curl -I http://localhost:8080` should return HTTP 200
- **JavaScript issues**: Ensure JavaScript is enabled in your browser
- **Browser compatibility**: Try different browsers (Chrome, Firefox, Edge)
- **Port forwarding**: In VS Code dev containers, use the "Ports" tab to forward port 8080
- **Try different port**: Use `--port 8088` if port 8080 has conflicts

The MLflow tracking server will automatically detect experiments logged to the default `./mlruns` directory.
