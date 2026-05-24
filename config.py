"""
Configuration module for Valolyzer.
Centralized settings for scrapers, parsers, and pipelines.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class ScraperConfig:
    """Scraper configuration."""

    name: str
    rate_limit: float = 2.0  # Requests per second
    timeout: int = 30  # Request timeout
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0


@dataclass
class PipelineConfig:
    """Pipeline configuration."""

    output_format: str = "csv"  # csv or parquet
    data_directory: str = "data"
    parallel_scrapers: bool = True
    max_concurrent_scrapers: int = 3
    deduplicate_on_export: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""

    log_directory: str = "logs"
    log_level: str = "INFO"
    log_rotation: str = "500 MB"
    log_retention: str = "7 days"
    include_debug_scraper_logs: bool = True


# Default configurations

VLR_SCRAPER_CONFIG = ScraperConfig(
    name="vlr_scraper",
    rate_limit=1.0,  # Conservative rate limit for VLR.gg
    timeout=30,
    max_retries=3,
)

RIB_SCRAPER_CONFIG = ScraperConfig(
    name="rib_scraper",
    rate_limit=1.5,
    timeout=30,
    max_retries=3,
)

TRACKER_SCRAPER_CONFIG = ScraperConfig(
    name="tracker_scraper",
    rate_limit=2.0,
    timeout=30,
    max_retries=3,
)

DEFAULT_PIPELINE_CONFIG = PipelineConfig(
    output_format="csv",
    data_directory="data",
    parallel_scrapers=True,
    max_concurrent_scrapers=3,
    deduplicate_on_export=True,
)

DEFAULT_LOGGING_CONFIG = LoggingConfig(
    log_directory="logs",
    log_level="INFO",
    log_rotation="500 MB",
    log_retention="7 days",
    include_debug_scraper_logs=True,
)


class Config:
    """Global configuration manager."""

    scrapers: Dict[str, ScraperConfig] = {
        "vlr": VLR_SCRAPER_CONFIG,
        "rib": RIB_SCRAPER_CONFIG,
        "tracker": TRACKER_SCRAPER_CONFIG,
    }

    pipeline: PipelineConfig = DEFAULT_PIPELINE_CONFIG
    logging: LoggingConfig = DEFAULT_LOGGING_CONFIG

    @classmethod
    def get_scraper_config(cls, scraper_name: str) -> Optional[ScraperConfig]:
        """Get configuration for scraper."""
        return cls.scrapers.get(scraper_name)

    @classmethod
    def set_scraper_config(cls, scraper_name: str, config: ScraperConfig):
        """Set configuration for scraper."""
        cls.scrapers[scraper_name] = config

    @classmethod
    def set_pipeline_config(cls, config: PipelineConfig):
        """Set pipeline configuration."""
        cls.pipeline = config

    @classmethod
    def set_logging_config(cls, config: LoggingConfig):
        """Set logging configuration."""
        cls.logging = config


if __name__ == "__main__":
    # Show current configuration
    print("Current Configuration:")
    print(f"Pipeline: {Config.pipeline}")
    print(f"Logging: {Config.logging}")
    print(f"Scrapers: {Config.scrapers}")
