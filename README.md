[![CI/CD](https://github.com/Adeemy/end-to-end-tabular-ml/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/Adeemy/end-to-end-tabular-ml/actions/workflows/main.yml)[![python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.99.1-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)

# End-to-end ML Project

This project is an end-to-end ML project on tabular data that applies software engineering practices in machine learning. It covers the entire lifecycle of a ML model from data exploration, preprocessing, feature engineering, model selection, training, evaluation, to deployment.

The project uses the Diabetes Health Indicators public dataset from [USC](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators), which was originally published by CDC. The dataset contains various information about patients such as demographics, lab results, and self-reported health history. The objective is to build a classifier that can predict whether a patient has diabetes, is pre-diabetic, or healthy.

The project follows best software engineering practices in machine learning, such as modular code, documentation, testing, logging, configuration, and version control. The project also demonstrates how to use various tools and frameworks, such as pandas, scikit-learn, [feast](https://feast.dev), [optuna](https://optuna.org), experiment tracking using [Comet](https://www.comet.com/site/), and Docker, to streamline the ML workflow and improve the model performance.

### Project structure

The project consists of an EDA notebook in notebooks folder and scripts in src folder, which are organized as follows:

- eda.ipynb: This notebook performs exploratory data analysis (EDA) on the dataset, such as descriptive statistics, data visualization, and correlation analysis. It also builds a baseline model (logistic regression) using scikit-learn, which achieves high precision and recall scores (above 0.80) on the test set.

- src/feature_store: This folder contains the script (generate_initial_data.py) that imports the original dataset from the source [USC](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators), and creates the inference set (5\% of the original dataset). The inference set is used to simulate production data, which is scored using the deployed model via a REST API call. The raw dataset is then preprocessed and transformed (prep_data.py) before ingesting it by feature store. See README.md in the feature_store folder for more details about feature store setup.

- src/training: This folder contains the scripts for data splitting, model training, evaluation, and selection. The train set is split into a training set, to train models, and a validation set for model selection. The test set is used to assess the generalization capability of the best model (used only once). It applies data preprocessing on the training set using a sklearn pipeline, such as handling missing values, feature scaling, feature engineering, feature selection, and categorical features encoding. It also implements hyperparameter optimization (train.py) using the optuna framework, and compares the results with the baseline model. The training pipeline is tracked and managed by [Comet](https://www.comet.com/site/), which records the model parameters, metrics, and artifacts. Once the best challenger model is selected, it is registered in the Comet workspace.

- src/inference: This folder contains the script for scoring new data via REST API using containerized model, which is deployed using GitHub Actions CI/CD pipeline.

Below is the project structure.

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
        │   │   │   │   ├── preprocessed_dataset_features.parquet
        │   │   │   │   ├── preprocessed_dataset_target.parquet
        │   │   │   │   ├── raw_dataset.parquet
        │   │   │   │   ├── registry.db
        │   │   │   │   ├── test.parquet
        │   │   │   │   └── train.parquet
        │   │   │   ├── define_feature.py
        │   │   │   └── feature_store.yaml
        │   │   ├── initial_data_setup
        │   │   │   ├── generate_initial_data.py
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

This project uses a devcontainer to set up a full-featured development environment and install required dependencies in addition to some useful VS Code extensions. It allows isolating the tools, libraries, and runtimes needed for working with this project codebase, and to use VS Code’s full feature set inside the container. A devcontainer requires [Docker](https://docs.docker.com/engine/install/) to be up and running. It's recommended to use the devcontainer for this project to ensure consistency and reproducibility across different machines and platforms, but if you prefer not to, for whatever reason, you can create a virtual environment and install the python dependencies by running the following commands from the project root:

    python3.10 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

### Usage

The training and deployment pipelines can be run in GitHub Actions. You can also run the following commands in CLI to implement all steps from generating raw dataset to pulling packaged model:

- Generate raw dataset

        make gen_init_data

- Preprocess data before ingesting it by feature store

        make prep_data

- Setup feature store

        make setup_feast

- Split dataset extracted from feature store

        make split_data

- Submit training job

        make train

- Pull packaged model

        docker pull ghcr.io/adeemy/end-to-end-tabular-ml:9c670633e181da234e0c57639d72a4b2834c7809
