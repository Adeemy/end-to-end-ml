# Run this file in root path path

# Install packages, format code, sort imports, and run unit tests
install:
	pip install --upgrade pip &&\
		pip install black[jupyter] pytest pylint isort pytest-cov &&\
		pip install -r requirements.txt

isort:
	isort ./notebooks
	isort ./src
	isort ./tests

format:
	black ./notebooks
	black ./src
	black ./tests

test:	
	coverage run -m pytest -vvv
	coverage report -m

debug:
	pytest -vvv --pdb

lint:
	pylint --disable=R,C,E1120,import-error ./src/feature_store ./src/training ./src/inference 

all: install isort format test lint


# Generate and prepare initial dataset
gen_init_data:
	python ./src/feature_store/initial_data_setup/generate_initial_data.py ./config/feature_store/config.yml

# Preprocess and transform data before ingestion by feature store
prep_data:
	python ./src/feature_store/prep_data.py  ./config/feature_store/config.yml

# Setup feature store and view entities and feature views
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

setup_feast: teardown_feast init_feast show_feast_entities show_feast_views


# Submit train experiment
split_data:
	python ./src/training/split_data.py ./src/feature_store/feature_repo/  ./config/training/config.yml

train:
	python ./src/training/train.py ./config/training/config.yml

submit_train: prep_data split_data train


# Test containerized model
test_container:
	cd /workspaces/end-to-end-ml/src/inference &&\
	uvicorn --host 0.0.0.0 main:app

