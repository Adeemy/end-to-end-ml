import logging
from pathlib import Path

from src.utils import path
from src.utils.logger import LoggerWriter, get_console_logger


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


def test_logger_writer_write(mocker):
    """Tests if the write method writes messages to the appropriate logger object."""

    # Create mock logger objects
    console_logger = mocker.MagicMock()
    print_logger = mocker.MagicMock()

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


def test_logger_writer_flush(mocker):
    """Tests if the flush method does nothing (it's just required by
    io.TextIOBase class)."""

    # Create mock logger objects
    console_logger = mocker.MagicMock()
    print_logger = mocker.MagicMock()

    # Create a LoggerWriter instance
    logger_writer = LoggerWriter(console_logger, print_logger)

    # Call the flush method
    logger_writer.flush()

    # Assert that the flush method does nothing
    console_logger.assert_not_called()
    print_logger.assert_not_called()


def test_get_console_logger(mocker):
    """Tests if the get_console_logger function returns a logger with the
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

    # Call the get_console_logger function
    logger = get_console_logger("test")

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
