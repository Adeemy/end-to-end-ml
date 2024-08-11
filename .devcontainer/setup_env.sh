#!/bin/bash

# Setup default virtual environment and install requirements
# echo -e 'export PYTHONPATH=$PYTHONPATH:~/' >> ~/.bashrc
# python3 -m venv .venv
# echo 'source .venv/bin/activate' >> ~/.bashrc
# pip install --upgrade pip
# pip install -r requirements.txt

# # Install conda and create the environment
# conda env create -f train-conda.yml
# conda activate train-env

# To remove the deprecation warning from jupyter_client when running tests
touch pytest.ini
echo -e '[pytest]\nfilterwarnings ='\
'\n    ignore::DeprecationWarning:jupyter_client.*'\
'\n    ignore::optuna.exceptions.ExperimentalWarning'\
'\n    ignore::DeprecationWarning:comet_ml.*' > pytest.ini

# Install auxiliar tools
sudo apt-get update
sudo apt-get install tree # To print project structure
