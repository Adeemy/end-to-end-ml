#!/bin/bash
set -euo pipefail # Exit on error, undefined variable, or error in a pipeline

# Setup default virtual environment and ensure PYTHONPATH is available in future shell sessions if not already present
if ! [[ $(grep -c 'export PYTHONPATH=$PYTHONPATH:~/' ~/.bashrc) -gt 0 ]]; then
    echo 'export PYTHONPATH=$PYTHONPATH:~/' >> ~/.bashrc
fi


# Create a virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Activate the virtual environment
source .venv/bin/activate

# Install/Upgrade uv using pip

# Install all dependencies from pyproject.toml using uv
echo "Installing project dependencies with uv..."
uv pip install --link-mode=copy -e '.[dev]'

# Create pytest.ini to filter common warnings from jupyter_client when running tests
echo "Configuring pytest..."
touch pytest.ini
echo -e '[pytest]\nfilterwarnings ='\
'\n    ignore::DeprecationWarning:jupyter_client.*'\
'\n    ignore::optuna.exceptions.ExperimentalWarning'\
'\n    ignore::DeprecationWarning:comet_ml.*' > pytest.ini

# Install auxiliary tools
echo "Installing auxiliary tools..."
sudo apt-get update
sudo apt-get install -y tree # To print project structure

echo "Environment setup complete."pip install -U uv
