[![CI/CD](https://github.com/Adeemy/end-to-end-tabular-ml/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/Adeemy/end-to-end-tabular-ml/actions/workflows/main.yml)[![python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.99.1-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)

# End-to-end ML Project

This project is an end-to-end ML project on tabular data that incorporates software engineering principles in machine learning. It spans the whole lifecycle of a ML model, from data exploration, preprocessing, feature engineering, model selection, training, evaluation, to deployment.

The project leverages the Diabetes Health Indicators public dataset from [UCI](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators), which was originally sourced from CDC. The dataset comprises various information about patients, such as demographics, lab results, and self-reported health history. The goal is to develop a classifier that can discern whether a patient has diabetes, is pre-diabetic, or healthy. This project, however, does not focus on achieving higher accuracy, as that is beyond the scope of this project.

The project adheres to best software engineering practices in machine learning, such as modular code, documentation, testing, logging, configuration, and version control. The project also demonstrates how to utilize various tools and frameworks, such as pandas, scikit-learn, [feast](https://feast.dev), [optuna](https://optuna.org), experiment tracking using [Comet](https://www.comet.com/site/), and Docker, to facilitate the ML workflow and enhance the model performance.

Some of the notable features of the project are:

1. The repo is **configurable** using config files, which allow the user to easily change the dataset, model, hyperparameters, and other settings without modifying the code.

2. The project uses **hyperparameters optimization** using optuna, which is a hyperparameters optimization framework that offers several advantages, such as efficient search algorithms, parallel and distributed optimization, and visualization of the optimization process.

3. The project uses the **f_beta score** as the optimization metric, which is a generalization of the f1 score (i.e., beta = 1) that can be adjusted to give more weights to precision or recall. The use of f_beta score is appropriate in many practical use cases, as in reality precision and recall are rarely equally important.

4. The project ensures **reproducibility** using Docker, which is a tool that creates isolated environments for running applications. The project containerizes the model with its dependencies, and provides a devcontainer configuration that allows the user to recreate the dev environment in VS code.

5. The project uses **experiment tracking and logging** using Comet, which is a platform for managing and comparing ML experiments. The project logs the model performance, hyperparameters, and artifacts to Comet, which can be accessed through a web dashboard. The user can also visualize and compare different experiments using Comet.

6. This project uses a makefile to provide convenient CLI commands to improve the efficiency, reliability, and quality of development, testing, and deployment. For instance, running the command `make prep_data` will transform the raw data into features and stores them in local file to be ingested by feature store, whereas running `make setup_feast` will apply the feature definitions to the feature store.

### Project structure

The project consists of the following folders and files:

- config: contains configuration files that includes parameters for data preprocessing and transformation, and model training.

- notebooks: contains a notebook (eda.ipynb) that conducts exploratory data analysis (EDA) on the dataset, such as descriptive statistics, data visualization, and correlation analysis. It also establishes a baseline model (logistic regression) using scikit-learn, which achieves high precision and recall scores (above 0.80) on the test set. Other notebooks can be added to this folder if needed.

- src/feature_store: contains the scripts for data ingestion and transformation. The script (generate_initial_data.py) imports the original dataset from the source [UCI](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators), and creates the inference set (5% of the original dataset). The inference set is used to simulate production data, which is scored using the deployed model via a REST API call. The script (prep_data.py) preprocesses and transforms the raw dataset before ingesting it by feature store. For more details about feature store setup, see README.md in the feature_store folder.

- src/training: contains the scripts for data splitting, model training, evaluation, and selection. The script (split_data.py) splits the train set into a training set, to train models, and a validation set for model selection. The test set is used to assess the generalization capability of the best model (used only once). The script (train.py) applies data preprocessing on the training set using a sklearn pipeline, such as handling missing values, feature scaling, feature engineering, feature selection, and categorical features encoding. It also implements hyperparameter optimization using the optuna framework. The training pipeline is tracked and managed by [Comet](https://www.comet.com/site/), which records the model parameters, metrics, and artifacts. Once the best model is selected, it is registered in the Comet workspace as champion model if it's score on test set is better than a required threshold value.

- src/inference: contains the script for scoring new data via REST API using containerized model, which is deployed using GitHub Actions CI/CD pipeline.

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

### Setup environment & Usage

This project uses a devcontainer to set up a full-featured development environment and install required dependencies in addition to some useful VS Code extensions. It allows isolating the tools, libraries, and runtimes needed for working with this project codebase, and to use VS Code’s full feature set inside the container. A devcontainer requires [Docker](https://docs.docker.com/engine/install/) to be up and running. It's recommended to use the devcontainer for this project to ensure consistency and reproducibility across different machines and platforms, but if you prefer not to, for whatever reason, you can create a virtual environment and install the python dependencies by running the following commands from the project root:

    python3.10 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

The training and deployment pipelines can be run in GitHub Actions. You can also run the following commands in CLI to implement all steps from generating raw dataset to pulling packaged model:

- Import raw dataset from [UCI](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)

        make get_init_data

- Preprocess data before ingesting it by feature store

        make prep_data

- Setup feature store

        make setup_feast

- Split dataset extracted from feature store

        make split_data

- Submit training job

        make train

- Pull containerized model

        docker pull ghcr.io/adeemy/end-to-end-tabular-ml:9c670633e181da234e0c57639d72a4b2834c7809
