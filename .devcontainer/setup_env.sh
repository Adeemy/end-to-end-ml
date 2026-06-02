#!/bin/bash
set -euo pipefail # Exit on error, undefined variable, or error in a pipeline

# Add the project root to PYTHONPATH in .bashrc if it's not already there.
# This makes project modules importable in interactive shells.
echo "Configuring PYTHONPATH..."
PROJECT_ROOT_PATH="export PYTHONPATH=\$PYTHONPATH:$(pwd)"
if ! grep -qF "$PROJECT_ROOT_PATH" ~/.bashrc; then
    echo "$PROJECT_ROOT_PATH" >> ~/.bashrc
fi

# Create a virtual environment if it doesn't exist
echo "Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Activate the virtual environment for the current script
echo "Activating virtual environment..."
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

# Add the venv activation command to .bashrc if it's not already there.
echo "Configuring virtual environment activation in .bashrc..."
VENV_ACTIVATE_COMMAND="source $(pwd)/.venv/bin/activate"
if ! grep -qF "$VENV_ACTIVATE_COMMAND" ~/.bashrc; then
    echo "$VENV_ACTIVATE_COMMAND" >> ~/.bashrc
fi

# Install Claude Code so it is available on every container build/rebuild,
# and make sure its install location is on the PATH.
echo "Installing Claude Code..."
curl -fsSL https://claude.ai/install.sh | bash
# shellcheck disable=SC2016  # keep $HOME literal so it expands at login, not now
CLAUDE_PATH_COMMAND='export PATH="$HOME/.local/bin:$PATH"'
if ! grep -qF "$CLAUDE_PATH_COMMAND" ~/.bashrc; then
    # shellcheck disable=SC1090  # sourcing the user's .bashrc is intentional
    echo "$CLAUDE_PATH_COMMAND" >> ~/.bashrc && source ~/.bashrc
fi

echo "Environment setup complete."
