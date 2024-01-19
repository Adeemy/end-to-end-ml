import os
from pathlib import Path

# Specify paths to data and training artifacts directories
PARENT_DIR = Path(__file__).parent.resolve().parent.parent

# Path to data directory
FEATURE_REPO_DIR = PARENT_DIR / "feature_store/feature_repo"
DATA_DIR = PARENT_DIR / "feature_store/feature_repo/data"

# Path to directory to save training artifacts
ARTIFACTS_DIR = PARENT_DIR / "training/artifacts"

if not Path(FEATURE_REPO_DIR).exists():
    os.mkdir(FEATURE_REPO_DIR)

if not Path(DATA_DIR).exists():
    os.mkdir(DATA_DIR)

if not Path(ARTIFACTS_DIR).exists():
    os.mkdir(ARTIFACTS_DIR)
