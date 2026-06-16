"""
Test functions for the utils module in src/utils.
"""

import logging
from pathlib import Path

from src.utils import path
from src.utils.logger import get_logger


def test_paths_exist():
    """Tests if the paths exist."""

    assert Path(path.FEATURE_REPO_DIR).exists()
    assert Path(path.DATA_DIR).exists()
    assert Path(path.ARTIFACTS_DIR).exists()


def test_paths_are_directories():
    """Tests if the paths are directories."""

    assert Path(path.FEATURE_REPO_DIR).is_dir()
    assert Path(path.DATA_DIR).is_dir()
    assert Path(path.ARTIFACTS_DIR).is_dir()


def test_encoded_split_path():
    """encoded_split_path inserts an '_encoded' suffix before the extension."""
    result = path.encoded_split_path("/data", "train.parquet")
    assert Path(result) == Path("/data/train_encoded.parquet")
    # Works for the other split names too.
    assert Path(path.encoded_split_path("/d", "validation.parquet")) == Path(
        "/d/validation_encoded.parquet"
    )


def test_get_logger(mocker):
    """Tests if the get_logger function returns a logger with the
    correct properties."""

    # Mock the logging.getLogger, logging.StreamHandler, and logging.Formatter functions
    mock_get_logger = mocker.patch(
        "logging.getLogger", return_value=logging.Logger("test")
    )
    mock_stream_handler = mocker.patch(
        "logging.StreamHandler",
        return_value=mocker.MagicMock(spec=logging.StreamHandler),
    )
    mock_formatter = mocker.patch("logging.Formatter", return_value=logging.Formatter())

    # Call the get_logger function
    logger = get_logger("test")

    # Check that the logger has the correct name
    assert logger.name == "test"

    # Check that the logger has one handler
    assert len(logger.handlers) == 1

    # Check that the handler is a MagicMock
    handler = logger.handlers[0]
    assert isinstance(handler, mocker.MagicMock)

    # Check that the setLevel and setFormatter methods of the StreamHandler mock object were called
    handler.setLevel.assert_called_once_with(logging.DEBUG)
    handler.setFormatter.assert_called_once()

    # Check that the mocked functions were called
    mock_get_logger.assert_called_once_with("test")
    mock_stream_handler.assert_called_once()
    mock_formatter.assert_called_once_with(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
