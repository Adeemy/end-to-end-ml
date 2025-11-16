#!/bin/bash
set -euo pipefail # Exit on error, undefined variable, or error in a pipeline

# Add the project root to PYTHONPATH in .bashrc if it's not already there.
# This makes project modules importable in interactive shells.
PROJECT_ROOT_PATH="export PYTHONPATH=\$PYTHONPATH:$(pwd)"
if ! grep -qF "$PROJECT_ROOT_PATH" ~/.bashrc; then
    echo "$PROJECT_ROOT_PATH" >> ~/.bashrc
fi
# Create a virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Activate the virtual environment
source .venv/bin/activate

# Install/Upgrade uv using pip
echo "Installing/upgrading uv..."
pip install -U uv

# Install all dependencies from pyproject.toml using uv
echo "Installing project dependencies with uv..."
uv pip install --link-mode=copy -e '.[dev]'

# Create pytest.ini to filter common warnings from jupyter_client when running tests
echo "Configuring pytest..."
cat > pytest.ini <<'PYTEST'
[pytest]
filterwarnings =
    ignore::DeprecationWarning:jupyter_client.*
    ignore::optuna.exceptions.ExperimentalWarning
    ignore::DeprecationWarning:comet_ml.*
PYTEST

# Install auxiliary tools
echo "Installing auxiliary tools..."
sudo apt-get update
sudo apt-get install -y tree # To print project structure

echo "Environment setup complete."
