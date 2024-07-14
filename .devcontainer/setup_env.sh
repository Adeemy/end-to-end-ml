#!/bin/bash

# Setup default virtual environment and install requirements
echo -e 'export PYTHONPATH=$PYTHONPATH:~/' >> ~/.bashrc
conda env create -f train-conda.yml
echo 'conda activate train-env' >> ~/.bashrc

# To remove the deprecation warning from jupyter_client when running tests
touch pytest.ini
echo -e '[pytest]\nfilterwarnings ='\
'\n    ignore::DeprecationWarning:jupyter_client.*'\
'\n    ignore::optuna.exceptions.ExperimentalWarning'\
'\n    ignore::DeprecationWarning:comet_ml.*' > pytest.ini

# Install auxiliar tools
sudo apt-get update
sudo apt-get install tree # To print project structure

# Install Azure CLI for Azure development (optional)
# curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
