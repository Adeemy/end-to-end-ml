"""
Define paths to data and training artifacts directories and creates
them if they do not exist.
"""

import os
from pathlib import Path, PurePath
from typing import Union

# Specify paths to data and training artifacts directories
PARENT_DIR = Path(__file__).parent.resolve().parent
LOG_CONF_PATH = PARENT_DIR / "config/logging.conf"

# File (under ARTIFACTS_DIR) where train.py records the [model_name, run_id] pairs
# of the models it just trained, so a standalone `make evaluate` can evaluate that
# exact (most recent) training run without querying a remote tracker.
TRAINING_EXPERIMENTS_FILE = "training_experiments.json"


def encoded_split_path(data_dir: Union[str, PurePath], file_name: str) -> Path:
    """Returns the path of the label-encoded variant of a data split file.

    train.py writes the feature-selected, label-encoded splits used by
    evaluate.py to separate ``*_encoded.parquet`` files rather than overwriting
    the canonical splits produced by split_data.py (which keeps the pipeline
    idempotent). Both modules resolve the encoded path through this helper.

    Args:
        data_dir: Directory holding the split files.
        file_name: Canonical split file name (e.g. ``train.parquet``).

    Returns:
        Path: ``<data_dir>/<stem>_encoded<suffix>``.
    """
    stem = PurePath(file_name)
    return Path(data_dir) / f"{stem.stem}_encoded{stem.suffix}"


# Path to data directory
FEATURE_REPO_DIR = PARENT_DIR / "feature/feature_repo"
DATA_DIR = PARENT_DIR / "feature/feature_repo/data"

# Path to directory to save training artifacts
ARTIFACTS_DIR = PARENT_DIR / "training/artifacts"

if not Path(FEATURE_REPO_DIR).exists():
    os.mkdir(FEATURE_REPO_DIR)

if not Path(DATA_DIR).exists():
    os.mkdir(DATA_DIR)

if not Path(ARTIFACTS_DIR).exists():
    os.mkdir(ARTIFACTS_DIR)
