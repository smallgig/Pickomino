"""Logging configuration for debugging."""

__all__ = ["log"]

import tempfile
from pathlib import Path

try:  # pylint: disable=too-many-try-statements
    from loguru import logger

    logger.remove()  # Remove default stderr handler.
    # On Windows: C:\Users\<username>\AppData\Local\Temp\pickomino.log.
    logger.add(Path(tempfile.gettempdir()) / "pickomino.log", mode="w")  # See the path for the log file above.

    def log(message: str) -> None:
        """Log a debug message if loguru is available."""
        logger.debug(message)

except ImportError:
    # noinspection PyUnusedLocal
    def log(message: str) -> None:  # pylint: disable=unused-argument
        """No-op when logging is disabled."""
