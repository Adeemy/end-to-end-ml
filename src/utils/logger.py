"""
Defines a class that redirects printed messages in addition to some select events that
need to be logged to logger objects.
"""

import logging


def get_console_logger(name: str) -> logging.Logger:
    """Creates a console logger. It can be used when only select
    events only needs to be logged but not print messages.

    Args:
        name (str): name of logger.

    Returns:
        logger (logging.logger): console logger object.
    """

    # Create logger object
    logger = logging.getLogger(name)

    # Ensure the logger has only one handler
    if not logger.handlers:
        # Set logger level
        logger.setLevel(logging.DEBUG)

        # Create console handler with formatting
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

        # Add console handler to the logger
        logger.addHandler(console_handler)

    return logger
