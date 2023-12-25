[![CI/CD](https://github.com/Adeemy/end-to-end-tabular-ml/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/Adeemy/end-to-end-tabular-ml/actions/workflows/main.yml)[![python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.99.1-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)

# End-to-end ML Project Template

A template for end-to-end ML project for tabular data. The repo is still under development.

# Usage

- Generate raw dataset

  get_init_data

- Import data from feature store

  prep_data

- Split original dataset

  split_data

- Submit training job

  train

### Build container

`docker build .`
`docker image ls`

### Pull packaged model

docker pull ghcr.io/adeemy/end-to-end-ml:36ebd68919acff1dda2454c38d0c59b7df1c2daf@sha256:6343e8f8e946bc3458b6a3a064e46c58edf7fc0e0b68bbb6e84ab536ade0f80f
