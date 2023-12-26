[![CI/CD](https://github.com/Adeemy/end-to-end-tabular-ml/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/Adeemy/end-to-end-tabular-ml/actions/workflows/main.yml)[![python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.99.1-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)

# End-to-end ML Project Template

A template for end-to-end ML project for tabular data. The repo is still under development.

### Project structure
        end-to-end-tabular-ml
        ├── LICENSE
        ├── Makefile
        ├── README.md
        ├── config
        │   ├── __init__.py
        │   ├── feature_store
        │   │   └── config.yml
        │   └── training
        │       └── config.yml
        ├── notebooks
        │   ├── eda.ipynb
        │   ├── eda_requirements.txt
        │   └── utils.py
        ├── requirements.txt
        ├── src
        │   ├── __init__.py
        │   ├── feature_store
        │   │   ├── README.md
        │   │   ├── feature_repo
        │   │   │   ├── data
        │   │   │   │   ├── inference.parquet
        │   │   │   │   ├── online_store.db
        │   │   │   │   ├── preprocessed_dataset.parquet
        │   │   │   │   ├── raw_dataset.parquet
        │   │   │   │   ├── raw_dataset_features.parquet
        │   │   │   │   ├── raw_dataset_target.parquet
        │   │   │   │   ├── registry.db
        │   │   │   │   ├── test.parquet
        │   │   │   │   └── train.parquet
        │   │   │   ├── define_feature.py
        │   │   │   └── feature_store.yaml
        │   │   ├── initial_data_setup
        │   │   │   ├── generate_initial_data.py
        │   │   │   └── prep_initial_data.py
        │   │   ├── prep_data.py
        │   │   └── utils
        │   │       ├── __init__.py
        │   │       ├── config.py
        │   │       └── prep.py
        │   ├── inference
        │   │   ├── Dockerfile
        │   │   ├── __init__.py
        │   │   ├── main.py
        │   │   └── utils.py
        │   └── training
        │       ├── artifacts
        │       │   ├── logistic-regression.pkl
        │       │   └── study_LogisticRegression.csv
        │       ├── check_drift.py
        │       ├── split_data.py
        │       ├── train.py
        │       └── utils
        │           ├── __init__.py
        │           ├── config.py
        │           ├── data.py
        │           ├── job.py
        │           ├── model.py
        │           └── path.py
        └── tests
        └── test_feature_utils.py


### Setup environment

    python3.10 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

### Usage

- Generate raw dataset

        make get_init_data

- Setup feature store

        make setup_feast

- Preprocess data from feature store

        make prep_data

- Split dataset

        make split_data

- Submit training job

        make train

### Pull packaged model

    docker pull ghcr.io/adeemy/end-to-end-tabular-ml:9c670633e181da234e0c57639d72a4b2834c7809
