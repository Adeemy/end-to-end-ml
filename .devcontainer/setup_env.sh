#!/bin/bash

# Set up the default conda environment and install dependencies
conda env create -f src/training/train-env.yml
echo -e 'conda activate train-env' >> ~/.bashrc

# Remove deprecation warning from jupyter_client when running tests
touch pytest.ini
echo -e '[pytest]\nfilterwarnings ='\
'\n    ignore::DeprecationWarning:jupyter_client.*'\
'\n    ignore::optuna.exceptions.ExperimentalWarning'\

# Install auxiliary tools
sudo apt-get update && apt-get install -y \
    tree # To print directory tree
