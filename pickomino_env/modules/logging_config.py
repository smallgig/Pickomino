"""Logging configuration for debugging."""

from pathlib import Path

__all__ = ["log"]

try:  # pylint: disable=too-many-try-statements
    from loguru import logger

    logger.remove()  # Remove default stderr handler.
    logger.add(Path(__file__).parent.parent / "pickomino.log", mode="w")

    def log(message: str) -> None:
        """Log a debug message if loguru is available."""
        logger.debug(message)

except ImportError:
    # noinspection PyUnusedLocal
    def log(message: str) -> None:  # pylint: disable=unused-argument
        """No-op when logging is disabled."""
