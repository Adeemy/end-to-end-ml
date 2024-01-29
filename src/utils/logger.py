import io
import logging


class LoggerWriter(io.TextIOBase):
    """Provider console logger and print logger that redirects printed messages
    to logger objects. It can be used when all print messages are logged (like
    message output by function or class methods) in addition to some select
    events that need to be logged.

    Attributes:
        console_logger (logging.Logger): The logger object for console output.
        print_logger (logging.Logger): The logger object for print output.
    """

    def __init__(self, console_logger, print_logger):
        """Initializes the LoggerWriter object with two logger objects.

        Args:
            console_logger (logging.Logger): The logger object for console output.
            print_logger (logging.Logger): The logger object for print output.
        """

        super().__init__()

        # Store the console logger and the print logger
        self.console_logger = console_logger
        self.print_logger = print_logger

    def write(self, message) -> None:
        """Writes a message to the appropriate logger object.

        Args:
            message (str): The message to be written.

        Returns:
            None.
        """

        # Ignore empty messages
        if message != "\n":
            # Check the level of the message
            if message.startswith("DEBUG") or message.startswith("INFO"):
                # Write the message to the console logger
                self.console_logger.info(message)

            else:
                # Write the message to the print logger
                self.print_logger.info(message)

    def flush(self):
        """Does nothing, as the logger objects handle flushing but it's
        required by io.TextIOBase class.
        """

        pass


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
