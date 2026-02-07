"""Logging configuration for debugging."""

__all__ = ["log"]

try:  # pylint: disable=too-many-try-statements
    from loguru import logger

    logger.remove()  # Remove default stderr handler.
    logger.add("loguru.log")

    def log(message: str) -> None:
        """Log a debug message if loguru is available."""
        logger.debug(message)

except ImportError:
    # noinspection PyUnusedLocal
    def log(message: str) -> None:  # pylint: disable=unused-argument
        """No-op when logging is disabled."""
