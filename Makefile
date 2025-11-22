# Run this file in project root directory

# Prepend the virtual environment's bin directory to the PATH for all rules.
# This allows calling tools like `uv`, `python`, `isort` directly.
VENV_BIN := $(CURDIR)/.venv/bin
export PATH := $(VENV_BIN):$(PATH)

# Create a virtual environment and install uv. Primarily for CI.
setup:
	python -m venv .venv
	python -m pip install -U uv

# Install project dependencies. Depends on the environment being set up.
install: setup
	uv pip install --link-mode=copy -e '.[dev]'

# Install dependencies for analysis tasks only in addition to base dependencies.
install_analysis: setup
	uv pip install --link-mode=copy -e '.[analysis]'

pre_commit:
	pre-commit install --hook-type pre-commit --hook-type pre-merge-commit

isort:
	isort --profile black --filter-files ./notebooks ./src ./tests

format:
	black ./notebooks ./src ./tests

test:
	coverage run -m pytest -vvv
	coverage report -m

debug:
	pytest -vvv --pdb

# Exclude W0511 (fixme) and E1120 (no value for argument in function call) in .pylintrc file
lint:
	pylint ./src/feature_store ./src/training ./src/inference ./tests ./notebooks

all: install isort format test lint


# Import raw dataset from source
get_init_data:
	python ./src/feature_store/generate_initial_data.py --config_yaml_path ./src/config/feature-store-config.yml --logger_path ./src/config/logging.conf

# Preprocess and transform data before ingestion by feature store
prep_data:
	python ./src/feature_store/prep_data.py --config_yaml_path ./src/config/feature-store-config.yml --logger_path ./src/config/logging.conf

# Setup feature store, view entities and feature views
teardown_feast:
	cd ./src/feature_store/feature_repo && feast teardown

init_feast:
	cd ./src/feature_store/feature_repo && feast apply

show_feast_entities:
	cd ./src/feature_store/feature_repo && feast entities list

show_feast_views:
	cd ./src/feature_store/feature_repo && feast feature-views list

show_feast_ui:
	cd ./src/feature_store/feature_repo && feast ui

setup_feast: teardown_feast init_feast show_feast_entities show_feast_views


# Submit train experiment
split_data:
	python ./src/training/split_data.py --config_yaml_path ./src/config/training-config.yml --logger_path ./src/config/logging.conf

train:
	python ./src/training/train.py --config_yaml_path ./src/config/training-config.yml

evaluate:
	python ./src/training/evaluate.py --config_yaml_path ./src/config/training-config.yml

submit_train: prep_data split_data train evaluate

# Test model locally
test_model:
	python ./src/inference/predict.py --config_yaml_path ./src/config/training-config.yml --logger_path ./src/config/logging.conf

# Test model via API (go to http://localhost:8000/docs page to test sample)
test_packaged_model:
	cd ./src/inference && uvicorn --host 0.0.0.0 main:app
