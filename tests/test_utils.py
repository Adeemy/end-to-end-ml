from pathlib import Path
from unittest.mock import MagicMock

from src.utils import path
from src.utils.logger import LoggerWriter


def test_paths_exist():
    """Test that the paths exist."""

    assert Path(path.FEATURE_REPO_DIR).exists()
    assert Path(path.DATA_DIR).exists()
    assert Path(path.ARTIFACTS_DIR).exists()


def test_paths_are_directories():
    """Test that the paths are directories."""

    assert Path(path.FEATURE_REPO_DIR).is_dir()
    assert Path(path.DATA_DIR).is_dir()
    assert Path(path.ARTIFACTS_DIR).is_dir()


def test_logger_writer_write():
    """Test that the write method writes messages to the appropriate logger object."""

    # Create mock logger objects
    console_logger = MagicMock()
    print_logger = MagicMock()

    # Create a LoggerWriter instance
    logger_writer = LoggerWriter(console_logger, print_logger)

    # Call the write method with a message
    logger_writer.write("DEBUG: This is a debug message")

    # Assert that the message is written to the console logger
    console_logger.info.assert_called_once_with("DEBUG: This is a debug message")

    # Call the write method with another message
    logger_writer.write("INFO: This is an info message")

    # Assert that the message is written to the console logger
    console_logger.info.assert_called_with("INFO: This is an info message")

    # Call the write method with a different message
    logger_writer.write("This is a print message")

    # Assert that the message is written to the print logger
    print_logger.info.assert_called_once_with("This is a print message")


def test_logger_writer_flush():
    """Tests that the flush method does nothing."""

    # Create mock logger objects
    console_logger = MagicMock()
    print_logger = MagicMock()

    # Create a LoggerWriter instance
    logger_writer = LoggerWriter(console_logger, print_logger)

    # Call the flush method
    logger_writer.flush()

    # Assert that the flush method does nothing
    console_logger.assert_not_called()
    print_logger.assert_not_called()
