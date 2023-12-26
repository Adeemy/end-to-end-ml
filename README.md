[![CI/CD](https://github.com/Adeemy/end-to-end-tabular-ml/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/Adeemy/end-to-end-tabular-ml/actions/workflows/main.yml)[![python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.99.1-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)

# End-to-end ML Project Template

A template for end-to-end ML project for tabular data. The repo is still under development.

### Setup environment

    python3.10 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

### Usage

- Generate raw dataset

        make get_init_data

- Setup feature store

        make setup_feast

- Import data from feature store

        make prep_data

- Split dataset

        make split_data

- Submit training job

        make train

### Pull packaged model

    docker pull ghcr.io/adeemy/end-to-end-tabular-ml:9c670633e181da234e0c57639d72a4b2834c7809
