"""
This module defines a function to create a console logger.
"""

import logging


def create_console_logger(logger_name: str) -> logging.Logger:
    """Creates a console logger with a specified name.

    Args:
        logger_name (str): The name of the logger.

    Returns:
        logger (logging.Logger): The console logger object.
    """

    # Create logger object
    logger = logging.getLogger(logger_name)

    # Configure logger if it doesn't have any handlers
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger
