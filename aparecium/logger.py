"""
Aparecium Logging Module

This module provides a centralized logger for the Aparecium system using a singleton
approach. It logs messages to both a file (with DEBUG level) and the console (with INFO level).
A 'logs' directory is automatically created if it does not exist. You can optionally
customize behavior using environment variables or direct assignment.

Typical usage example:
    from logger import logger

    logger.info("This is an info message.")
    logger.debug("This is a debug message for detailed troubleshooting.")
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler


class ApareciumLogger:
    """
    A singleton class that configures and provides a logger for Aparecium.

    The logger writes DEBUG-level messages to a dated log file and INFO-level (or above)
    messages to the console. It can be customized via environment variables or by
    modifying the code directly.

    Usage:
        logger = ApareciumLogger()
        logger.info("Your log message.")

    Attributes:
        logger (logging.Logger): The underlying Python logger instance.
        _instance (Optional[ApareciumLogger]): The singleton instance reference.
        _initialized (bool): Tracks whether the logger has already been configured.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        """
        Ensures that only one instance of ApareciumLogger exists.

        Returns:
            ApareciumLogger: The existing or newly created singleton instance.
        """
        if cls._instance is None:
            cls._instance = super(ApareciumLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Initializes the logger if it hasn't been set up yet. By default, logs
        go to 'logs/aparecium_<YYYYMMDD>.log' for DEBUG level and above,
        and also to console for INFO level and above.

        Environment Variables:
            APARECIUM_LOG_DIR (Optional[str]): Path to the logs directory. Defaults to 'logs'.
            APARECIUM_LOG_LEVEL (Optional[str]): Log level for file and console.
                Defaults to 'DEBUG' for file and 'INFO' for console if not set.

        Raises:
            OSError: If there is an error creating or writing to the log directory or file.
        """
        if not self._initialized:
            log_dir_path = os.getenv("APARECIUM_LOG_DIR", "logs")
            log_dir = Path(log_dir_path)
            log_dir.mkdir(exist_ok=True)

            file_log_level_str = os.getenv("APARECIUM_LOG_LEVEL", "DEBUG").upper()
            console_log_level_str = os.getenv("APARECIUM_CONSOLE_LEVEL", "INFO").upper()

            file_log_level = getattr(logging, file_log_level_str, logging.DEBUG)
            console_log_level = getattr(logging, console_log_level_str, logging.INFO)

            self.logger = logging.getLogger("aparecium")
            self.logger.setLevel(logging.DEBUG)

            log_file = log_dir / f'aparecium_{datetime.now().strftime("%Y%m%d")}.log'

            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB per file
                backupCount=5,  # Keep 5 backup files
                encoding="utf-8",
            )
            file_handler.setLevel(file_log_level)

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(console_log_level)

            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_formatter = logging.Formatter("%(levelname)s: %(message)s")
            file_handler.setFormatter(file_formatter)
            console_handler.setFormatter(console_formatter)

            if not self.logger.handlers:
                self.logger.addHandler(file_handler)
                self.logger.addHandler(console_handler)

            self._initialized = True

    def debug(self, message: str) -> None:
        """
        Logs a message with level DEBUG.

        Args:
            message (str): The log message.
        """
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """
        Logs a message with level INFO.

        Args:
            message (str): The log message.
        """
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """
        Logs a message with level WARNING.

        Args:
            message (str): The log message.
        """
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """
        Logs a message with level ERROR.

        Args:
            message (str): The log message.
        """
        self.logger.error(message)

    def critical(self, message: str) -> None:
        """
        Logs a message with level CRITICAL.

        Args:
            message (str): The log message.
        """
        self.logger.critical(message)

    def exception(self, message: str) -> None:
        """
        Logs an exception with level ERROR.

        Args:
            message (str): The log message describing the exception context.
        """
        self.logger.exception(message)


logger = ApareciumLogger()
