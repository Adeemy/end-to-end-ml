# Run this file in project root directory

# Install packages, format code, sort imports, and run unit tests
install:
	pip install --upgrade pip &&\
		pip install black[jupyter] pytest pylint isort pytest-cov &&\
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
	pytest -vvv
# coverage run -m pytest -vvv
# coverage report -m

debug:
	pytest -vvv --pdb

lint:
	pylint --disable=R,C,E1120 ./src/feature_store ./src/training ./src/inference 

check_code: install isort format test lint


# Import raw dataset from source
get_init_data:
	python ./src/feature_store/initial_data_setup/generate_initial_data.py ./config/feature_store/config.yml

# Preprocess and transform data before ingestion by feature store
prep_data:
	python ./src/feature_store/prep_data.py  ./config/feature_store/config.yml

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
	python ./src/training/split_data.py ./config/training/config.yml

train:
	python ./src/training/train.py ./config/training/config.yml

submit_train: prep_data split_data train


# Test model locally (go to http://localhost:8000/docs page to test sample)
test_container:
	cd ./src/inference &&\
	uvicorn --host 0.0.0.0 main:app

