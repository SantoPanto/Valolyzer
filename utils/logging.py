"""
Logging utilities for the Valolyzer pipeline.
Uses loguru for structured logging with rotation and formatting.
"""

import sys
from pathlib import Path
from loguru import logger as _logger
from typing import Optional


class Logger:
    """Centralized logging configuration."""

    _configured = False

    @classmethod
    def configure(cls, 
                  log_dir: str = "logs",
                  level: str = "INFO",
                  rotation: str = "500 MB",
                  retention: str = "7 days"):
        """
        Configure loguru logger with file and console output.

        Args:
            log_dir: Directory to store log files
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            rotation: Log rotation size
            retention: How long to keep old logs
        """
        if cls._configured:
            return

        # Create logs directory
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        # Remove default handler
        _logger.remove()

        # Console handler with formatting
        _logger.add(
            sys.stderr,
            level=level,
            format="<level>{time:YYYY-MM-DD HH:mm:ss}</level> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True
        )

        # File handler with rotation
        _logger.add(
            log_path / "valolyzer_{time:YYYY-MM-DD}.log",
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation=rotation,
            retention=retention,
        )

        # Scraper-specific log file
        _logger.add(
            log_path / "scraper_{time:YYYY-MM-DD}.log",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            filter=lambda record: "scraper" in record["name"].lower(),
        )

        cls._configured = True

    @classmethod
    def get_logger(cls, name: str = __name__):
        """
        Get a configured logger instance.

        Args:
            name: Logger name (typically __name__)

        Returns:
            Configured logger instance
        """
        return _logger.bind(name=name)


# Convenience function
def get_logger(name: Optional[str] = None):
    """Get a logger instance."""
    if name is None:
        name = __name__
    return _logger.bind(name=name)


# Configure on import
Logger.configure()

if __name__ == "__main__":
    log = get_logger(__name__)
    log.info("Logging configured successfully")
    log.debug("Debug message")
    log.warning("Warning message")
    log.error("Error message")
