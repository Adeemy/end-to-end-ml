# Run this file in project root directory

# Prepend the virtual environment's bin directory to the PATH for all rules.
# This allows calling tools like `uv`, `python`, `isort` directly.
VENV_BIN := $(CURDIR)/ml_env/bin
export PATH := $(VENV_BIN):$(PATH)

# Project venv interpreter, used explicitly by script targets so they run in the
# project environment even when another virtualenv is active in the shell.
PYTHON := $(VENV_BIN)/python

# Put the project root on PYTHONPATH so `import src.*` resolves when running the
# scripts directly (CI and the devcontainer set this too; this makes `make`
# targets work in a plain shell without extra setup).
export PYTHONPATH := $(CURDIR):$(PYTHONPATH)

# Create a virtual environment and install uv. Primarily for CI.
setup:
	python -m venv ml_env
	python -m pip install -U uv

# Install project dependencies. Depends on the environment being set up.
# --python targets the project venv explicitly (uv otherwise defaults to .venv).
# The feature-store and distributed extras are included so the full pipeline and
# the lint of src/feature (which imports feast) work.
install: setup
	uv pip install --python $(VENV_BIN)/python --link-mode=copy -e '.[dev,feature-store,distributed]'

# Install dependencies for analysis tasks only in addition to base dependencies.
install_analysis: setup
	uv pip install --python $(VENV_BIN)/python --link-mode=copy -e '.[analysis]'

pre_commit:
	pre-commit install --hook-type pre-commit --hook-type pre-merge-commit

# Recipes call tools by their explicit venv path ($(VENV_BIN)/...) so the project
# environment is used deterministically even when another virtualenv is active
# in the shell (an activated foreign venv would otherwise shadow these tools).
isort:
	$(VENV_BIN)/isort --profile black --filter-files ./notebooks ./src ./tests

format:
	$(VENV_BIN)/black ./notebooks ./src ./tests

test:
	$(VENV_BIN)/coverage run -m pytest -vvv
	$(VENV_BIN)/coverage report -m

debug:
	$(VENV_BIN)/pytest -vvv --pdb

# Exclude W0511 (fixme) and E1120 (no value for argument in function call) in .pylintrc file
lint:
	$(VENV_BIN)/pylint ./src/feature ./src/training ./src/inference ./tests ./notebooks

all: install isort format test lint


# Import raw dataset from source
gen_init_data:
	$(PYTHON) ./src/feature/generate_initial_data.py --config_yaml_path ./src/config/feature-store-config.yml --logger_path ./src/config/logging.conf

# Preprocess and transform data before ingestion by feature store
prep_data:
	$(PYTHON) ./src/feature/prep_data.py --config_yaml_path ./src/config/feature-store-config.yml --logger_path ./src/config/logging.conf

# Setup feature store, view entities and feature views
teardown_feast:
	cd ./src/feature/feature_repo && $(VENV_BIN)/feast teardown

init_feast:
	cd ./src/feature/feature_repo && $(VENV_BIN)/feast apply

show_feast_entities:
	cd ./src/feature/feature_repo && $(VENV_BIN)/feast entities list

show_feast_views:
	cd ./src/feature/feature_repo && $(VENV_BIN)/feast feature-views list

show_feast_ui:
	cd ./src/feature/feature_repo && $(VENV_BIN)/feast ui

setup_feast: teardown_feast init_feast show_feast_entities show_feast_views


# Submit train experiment
# NOTE: If using Comet ML tracker, set ENABLE_COMET_LOGGING=true in environment
split_data:
	$(PYTHON) ./src/training/split_data.py --config_yaml_path ./src/config/training-config.yml --logger_path ./src/config/logging.conf

train:
	$(PYTHON) ./src/training/train.py --config_yaml_path ./src/config/training-config.yml

# Evaluates the most recent training run by default. To evaluate a specific run:
#   make evaluate RUN_ID=<mlflow_run_id>
evaluate:
	$(PYTHON) ./src/training/evaluate.py --config_yaml_path ./src/config/training-config.yml $(if $(RUN_ID),--run_id $(RUN_ID),)

submit_train: prep_data split_data train evaluate

# Launch the MLflow UI to browse training and evaluation runs. Points at the
# repo's local file store (./mlruns) so it shows the runs this project logs;
# open http://localhost:8080. Override the port with PORT=...
# MLFLOW_ALLOW_FILE_STORE opts into the file backend on MLflow 3.x (the `mlflow ui`
# CLI doesn't import the project code that sets this automatically for training).
view_mlflow:
	MLFLOW_ALLOW_FILE_STORE=true $(VENV_BIN)/mlflow ui --backend-store-uri "file://$(CURDIR)/mlruns" --host 0.0.0.0 --port $(or $(PORT),8080)

# Test champion model via CLI (batch scoring requires inference.parquet file)
test_model_cli:
	$(PYTHON) ./src/inference/predict.py --config_yaml_path ./src/config/training-config.yml --logger_path ./src/config/logging.conf

# Batch prediction on parquet file
predict_batch:
	@echo "Running batch prediction on inference.parquet..."
	$(PYTHON) ./src/inference/predict.py \
		--config_yaml_path ./src/config/training-config.yml \
		--logger_path ./src/config/logging.conf \
		--input_file ./src/feature/feature_repo/data/inference.parquet \
		--output_file ./src/inference/artifacts/batch_predictions.parquet

# Batch prediction with custom input/output files (uses config defaults if not specified)
predict_batch_custom:
	@echo "Usage: make predict_batch_custom [INPUT_FILE=path/to/input.parquet] [OUTPUT_FILE=path/to/output.parquet]"
	@echo "Note: If INPUT_FILE/OUTPUT_FILE not specified, uses paths from training-config.yml"
	$(PYTHON) ./src/inference/predict.py \
		--config_yaml_path ./src/config/training-config.yml \
		--logger_path ./src/config/logging.conf \
		$(if $(INPUT_FILE),--input_file $(INPUT_FILE),) \
		$(if $(OUTPUT_FILE),--output_file $(OUTPUT_FILE),)

# Test API prediction logic directly via CLI (no HTTP server needed)
test_api_cli:
	$(PYTHON) ./src/inference/api_server.py --input_data '[{"BMI": 29.0, "PhysHlth": 0, "Age": "65 to 69", "HighBP": "0", "HighChol": "1", "CholCheck": "0", "Smoker": "1", "Stroke": "1", "HeartDiseaseorAttack": "0", "PhysActivity": "1", "Fruits": "1", "Veggies": "1", "HvyAlcoholConsump": "1", "AnyHealthcare": "1", "NoDocbcCost": "1", "GenHlth": "Poor", "MentHlth": "1", "DiffWalk": "1", "Sex": "1", "Education": "1", "Income": "7"}, {"BMI": 25.0, "PhysHlth": 2, "Age": "35 to 39", "HighBP": "1", "HighChol": "0", "CholCheck": "1", "Smoker": "0", "Stroke": "0", "HeartDiseaseorAttack": "0", "PhysActivity": "1", "Fruits": "1", "Veggies": "1", "HvyAlcoholConsump": "0", "AnyHealthcare": "1", "NoDocbcCost": "0", "GenHlth": "Very Good", "MentHlth": "0", "DiffWalk": "0", "Sex": "0", "Education": "6", "Income": "8"}]'

# Start REST API server for real-time predictions (go to http://localhost:8000/docs to test)
start_api_server:
	cd ./src/inference && $(VENV_BIN)/uvicorn --host 0.0.0.0 api_server:app

# Test API server with two data samples (requires server to be running)
test_api_with_sample:
	@echo "Testing API server with multiple sample data points..."
	@curl -X POST http://localhost:8000/predict \
	  -H "Content-Type: application/json" \
	  -d '[{"BMI": 29.0, "PhysHlth": 0, "Age": "65 to 69", "HighBP": "0", "HighChol": "1", "CholCheck": "0", "Smoker": "1", "Stroke": "1", "HeartDiseaseorAttack": "0", "PhysActivity": "1", "Fruits": "1", "Veggies": "1", "HvyAlcoholConsump": "1", "AnyHealthcare": "1", "NoDocbcCost": "1", "GenHlth": "Poor", "MentHlth": "1", "DiffWalk": "1", "Sex": "1", "Education": "1", "Income": "7"}, {"BMI": 25.0, "PhysHlth": 2, "Age": "35 to 39", "HighBP": "1", "HighChol": "0", "CholCheck": "1", "Smoker": "0", "Stroke": "0", "HeartDiseaseorAttack": "0", "PhysActivity": "1", "Fruits": "1", "Veggies": "1", "HvyAlcoholConsump": "0", "AnyHealthcare": "1", "NoDocbcCost": "0", "GenHlth": "Very Good", "MentHlth": "0", "DiffWalk": "0", "Sex": "0", "Education": "6", "Income": "8"}]' || true

# Test API server end-to-end (start server in background, test, then stop)
test_api_full:
	@echo "Starting API server in background..."
	@cd ./src/inference && uvicorn --host 0.0.0.0 api_server:app & echo $$! > /tmp/api_server.pid
	@echo "Waiting for server to start..."
	@sleep 5
	@echo "Testing health endpoint..."
	@curl -s http://localhost:8000/ || true
	@echo "\nTesting prediction endpoint with sample data..."
	@curl -X POST http://localhost:8000/predict \
	  -H "Content-Type: application/json" \
	  -d '{"BMI": 29.0, "PhysHlth": 0, "Age": "65 to 69", "HighBP": "0", "HighChol": "1", "CholCheck": "0", "Smoker": "1", "Stroke": "1", "HeartDiseaseorAttack": "0", "PhysActivity": "1", "Fruits": "1", "Veggies": "1", "HvyAlcoholConsump": "1", "AnyHealthcare": "1", "NoDocbcCost": "1", "GenHlth": "Poor", "MentHlth": "1", "DiffWalk": "1", "Sex": "1", "Education": "1", "Income": "7"}' || true
	@echo "\nStopping API server..."
	@kill `cat /tmp/api_server.pid` 2>/dev/null || true
	@rm -f /tmp/api_server.pid
	@echo "API test completed."

# Show available API testing commands
help_api:
	@echo "API Testing Commands:"
	@echo "  make start_api_server     - Start FastAPI server (visit http://localhost:8000/docs)"
	@echo "  make test_api_with_sample - Send sample data to running API server"
	@echo "  make test_api_full        - End-to-end test (start→test→stop server)"
	@echo ""
	@echo "CLI Testing Commands:"
	@echo "  make test_model_cli       - Test model via CLI with sample data"
	@echo "  make test_model           - Alias for test_model_cli"
	@echo ""
	@echo "Batch Prediction Commands:"
	@echo "  make predict_batch        - Predict on inference.parquet file"
	@echo "  make predict_batch_custom INPUT_FILE=input.parquet OUTPUT_FILE=output.parquet"
	@echo "                            - Predict on custom parquet file"

# Print recent directory structure
print_tree:
	tree -I "__pycache__|*.pyc|.git|.venv|node_modules"
