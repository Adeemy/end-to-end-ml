# Run this file in project root directory

# Update requirements.txt
#https://stackoverflow.com/questions/62496083/pipreqs-has-the-unicode-error-under-a-virtualenv
update_reqs:
	pip install pipreqs
	pipreqs --force --ignore bin,etc,include,lib,lib64 "./"

# Install packages, format code, sort imports, and run unit tests
install:
	pip install --upgrade pip &&\
		pip install black[jupyter] pytest pylint isort pytest-cov pytest-mock &&\
		pip install -r requirements.txt

isort:
	isort --profile black ./notebooks
	isort --profile black ./src
	isort --profile black ./tests

format:
	black ./notebooks
	black ./src
	black ./tests

test:	
	coverage run -m pytest -vvv
	coverage report -m

debug:
	pytest -vvv --pdb

# Exclude W0511 (fixme) and E1120 (no value for argument in function call) in .pylintrc file
lint:
	pylint ./src/feature_store ./src/training ./src/inference ./tests

all: install isort format test lint


# Import raw dataset from source
get_init_data:
	python ./src/feature_store/generate_initial_data.py --config_yaml_path ./src/config/feature_store_config.yml --logger_path ./src/config/logging.conf

# Preprocess and transform data before ingestion by feature store
prep_data:
	python ./src/feature_store/prep_data.py --config_yaml_path ./src/config/feature_store_config.yml --logger_path ./src/config/logging.conf

# Setup feature store, view entities and feature views
teardown_feast:
	cd ./src/feature_store/feature_repo &&\
	feast teardown

init_feast:
	cd ./src/feature_store/feature_repo &&\
	feast apply

show_feast_entities:
	cd ./src/feature_store/feature_repo &&\
	feast entities list

show_feast_views:
	cd ./src/feature_store/feature_repo &&\
	feast feature-views list

show_feast_ui:
	cd ./src/feature_store/feature_repo &&\
	feast ui

setup_feast: teardown_feast init_feast show_feast_entities show_feast_views


# Submit train experiment
split_data:
	python ./src/training/split_data.py --config_yaml_path ./src/config/training_config.yml --logger_path ./src/config/logging.conf

train:
	python ./src/training/train.py --config_yaml_path ./src/config/training_config.yml --logger_path ./src/config/logging.conf

evaluate:
	python ./src/training/evaluate.py --config_yaml_path ./src/config/training_config.yml --logger_path ./src/config/logging.conf

submit_train: prep_data split_data train evaluate

# Test model locally
test_model:
	python ./src/inference/predict.py --config_yaml_path ./src/config/training_config.yml --logger_path ./src/config/logging.conf

# Test model via API (go to http://localhost:8000/docs page to test sample)
test_packaged_model:
	cd ./src/inference &&\
	uvicorn --host 0.0.0.0 main:app

