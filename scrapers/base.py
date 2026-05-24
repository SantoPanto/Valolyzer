"""
Base scraper class for Valolyzer.
Provides abstract methods and common functionality for all scrapers.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from utils.logging import get_logger
from utils.http import AsyncHTTPClient, RetryStrategy
from utils.csv_handler import CSVManager
from utils.models import ScraperState

logger = get_logger(__name__)


class BaseScraper(ABC):
    """
    Abstract base class for all Valolyzer scrapers.
    Provides common functionality for scraping, rate limiting, and data export.
    """

    def __init__(self,
                 name: str,
                 rate_limit: float = 2.0,
                 timeout: int = 30,
                 data_dir: str = "data/raw"):
        """
        Initialize base scraper.

        Args:
            name: Scraper identifier
            rate_limit: Requests per second
            timeout: Request timeout in seconds
            data_dir: Directory for raw data storage
        """
        self.name = name
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # HTTP client (lazy initialized)
        self._http_client: Optional[AsyncHTTPClient] = None

        # State tracking
        self.state = ScraperState(
            scraper_name=name,
            last_scraped_at=datetime.now()
        )

        # Data containers
        self.matches: List[Dict[str, Any]] = []
        self.maps: List[Dict[str, Any]] = []
        self.compositions: List[Dict[str, Any]] = []
        self.player_stats: List[Dict[str, Any]] = []
        self.rounds: List[Dict[str, Any]] = []

        logger.info(f"Initialized scraper: {name}")

    @property
    async def http_client(self) -> AsyncHTTPClient:
        """Lazy initialize HTTP client."""
        if self._http_client is None:
            self._http_client = AsyncHTTPClient(
                rate_limit=self.rate_limit,
                timeout=self.timeout,
                retry_strategy=RetryStrategy()
            )
            await self._http_client.connect()
        return self._http_client

    @abstractmethod
    async def scrape(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Main scraping method to be implemented by subclasses.

        Returns:
            Dictionary with keys: matches, maps, compositions, player_stats, rounds
        """
        pass

    @abstractmethod
    async def parse_match(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse raw match data into normalized format."""
        pass

    @abstractmethod
    async def parse_map(self, map_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse raw map data into normalized format."""
        pass

    async def run(self) -> Dict[str, int]:
        """
        Execute scraper with error handling and state tracking.

        Returns:
            Statistics about scraping run
        """
        try:
            logger.info(f"Starting {self.name}")
            results = await self.scrape()

            stats = {
                "matches": len(results.get("matches", [])),
                "maps": len(results.get("maps", [])),
                "compositions": len(results.get("compositions", [])),
                "player_stats": len(results.get("player_stats", [])),
                "rounds": len(results.get("rounds", [])),
            }

            # Update state
            self.state.last_scraped_at = datetime.now()
            self.state.total_matches += stats["matches"]
            self.state.status = "active"

            logger.info(f"Completed {self.name}: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error in {self.name}: {e}", exc_info=True)
            self.state.total_errors += 1
            self.state.status = "error"
            raise

        finally:
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources."""
        if self._http_client:
            await self._http_client.close()

    def export_to_csv(self, 
                      format_type: str = "append",
                      deduplicate: bool = True) -> Dict[str, int]:
        """
        Export scraped data to CSV files.

        Args:
            format_type: "append" to existing or "overwrite"
            deduplicate: Whether to remove duplicates

        Returns:
            Number of rows written per data type
        """
        stats = {}

        # Define export mappings
        exports = {
            "matches.csv": (self.matches, ["match_id"]),
            "maps.csv": (self.maps, ["map_id"]),
            "compositions.csv": (self.compositions, ["map_id", "team"]),
            "player_stats.csv": (self.player_stats, ["map_id", "player", "team"]),
            "rounds.csv": (self.rounds, ["round_id"]),
        }

        for filename, (data, dedup_cols) in exports.items():
            if not data:
                continue

            output_path = self.data_dir.parent / "processed" / filename
            dedup = dedup_cols if deduplicate else None

            if format_type == "append":
                rows = CSVManager.append_csv(
                    data,
                    output_path,
                    deduplicate_on=dedup
                )
            else:
                CSVManager.list_dicts_to_csv(
                    data,
                    output_path,
                    deduplicate_on=dedup,
                    append=False
                )
                rows = len(data)

            stats[filename] = rows

        logger.info(f"Exported data: {stats}")
        return stats

    def export_to_parquet(self) -> Dict[str, int]:
        """
        Export scraped data to Parquet format (compressed, efficient).

        Returns:
            Number of rows per data type
        """
        import pandas as pd
        stats = {}

        exports = {
            "matches.parquet": self.matches,
            "maps.parquet": self.maps,
            "compositions.parquet": self.compositions,
            "player_stats.parquet": self.player_stats,
            "rounds.parquet": self.rounds,
        }

        for filename, data in exports.items():
            if not data:
                continue

            output_path = self.data_dir.parent / "parquet" / filename
            df = pd.DataFrame(data)
            CSVManager.to_parquet(df, output_path)
            stats[filename] = len(df)

        logger.info(f"Exported to parquet: {stats}")
        return stats

    def get_statistics(self) -> Dict[str, Any]:
        """Get scraper statistics."""
        return {
            "scraper_name": self.name,
            "matches_scraped": len(self.matches),
            "maps_scraped": len(self.maps),
            "compositions_scraped": len(self.compositions),
            "player_stats_scraped": len(self.player_stats),
            "rounds_scraped": len(self.rounds),
            "last_scraped_at": self.state.last_scraped_at,
            "total_errors": self.state.total_errors,
            "status": self.state.status,
        }


class ScraperPipeline:
    """Run multiple scrapers in sequence or parallel."""

    def __init__(self, scrapers: List[BaseScraper]):
        """
        Initialize pipeline with scrapers.

        Args:
            scrapers: List of scraper instances
        """
        self.scrapers = scrapers
        self.results = {}

    async def run_sequential(self) -> Dict[str, Dict[str, int]]:
        """Run scrapers sequentially."""
        logger.info(f"Running {len(self.scrapers)} scrapers sequentially")

        for scraper in self.scrapers:
            try:
                stats = await scraper.run()
                self.results[scraper.name] = stats
            except Exception as e:
                logger.error(f"Scraper {scraper.name} failed: {e}")

        return self.results

    async def run_parallel(self, max_concurrent: int = 3) -> Dict[str, Dict[str, int]]:
        """
        Run scrapers in parallel with concurrency limit.

        Args:
            max_concurrent: Maximum number of concurrent scrapers

        Returns:
            Results from all scrapers
        """
        logger.info(f"Running {len(self.scrapers)} scrapers in parallel "
                    f"(max {max_concurrent} concurrent)")

        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_with_semaphore(scraper):
            async with semaphore:
                try:
                    stats = await scraper.run()
                    return scraper.name, stats
                except Exception as e:
                    logger.error(f"Scraper {scraper.name} failed: {e}")
                    return scraper.name, {"error": str(e)}

        tasks = [run_with_semaphore(s) for s in self.scrapers]
        results = await asyncio.gather(*tasks)

        self.results = {name: stats for name, stats in results}
        return self.results

    def export_all(self, format_type: str = "append") -> Dict[str, Dict[str, int]]:
        """Export data from all scrapers."""
        export_results = {}

        for scraper in self.scrapers:
            export_results[scraper.name] = scraper.export_to_csv(format_type=format_type)

        logger.info(f"Exported data from {len(self.scrapers)} scrapers")
        return export_results

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline execution."""
        return {
            "total_scrapers": len(self.scrapers),
            "results": self.results,
            "timestamp": datetime.now(),
        }


if __name__ == "__main__":
    print("Base scraper module loaded")
