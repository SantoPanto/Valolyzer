"""
Main data pipeline orchestrator.
Coordinates scraping, parsing, and exporting workflow.
"""

import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import json

from scrapers.base import ScraperPipeline, BaseScraper
from parsers.data_parser import DataPipeline
from utils.logging import get_logger
from utils.csv_handler import CSVManager

logger = get_logger(__name__)


class ValolyzerPipeline:
    """Main pipeline for end-to-end data processing."""

    def __init__(self,
                 scrapers: List[BaseScraper],
                 data_dir: str = "data",
                 output_format: str = "csv"):
        """
        Initialize pipeline.

        Args:
            scrapers: List of scraper instances
            data_dir: Base data directory
            output_format: "csv" or "parquet"
        """
        self.scrapers = scrapers
        self.data_dir = Path(data_dir)
        self.output_format = output_format

        self.raw_data: Dict[str, List[Dict[str, Any]]] = {}
        self.processed_data: Dict[str, List[Dict[str, Any]]] = {}
        self.export_stats: Dict[str, int] = {}

        # Ensure directories exist
        (self.data_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "processed").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "parquet").mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized pipeline with {len(scrapers)} scrapers")

    async def run(self, sequential: bool = False) -> Dict[str, Any]:
        """
        Run complete pipeline.

        Args:
            sequential: Run scrapers sequentially (safer) or parallel

        Returns:
            Pipeline execution statistics
        """
        logger.info("Starting Valolyzer pipeline")
        start_time = datetime.now()

        try:
            # Step 1: Scrape
            logger.info("Step 1: Scraping data...")
            scrape_stats = await self._scrape(sequential=sequential)

            # Step 2: Aggregate
            logger.info("Step 2: Aggregating raw data...")
            self._aggregate_data()

            # Step 3: Parse
            logger.info("Step 3: Parsing and normalizing data...")
            self._parse_data()

            # Step 4: Export
            logger.info("Step 4: Exporting processed data...")
            self._export_data()

            duration = (datetime.now() - start_time).total_seconds()

            return {
                "status": "success",
                "duration_seconds": duration,
                "scrape_stats": scrape_stats,
                "export_stats": self.export_stats,
                "processed_data_stats": self._get_data_stats(),
            }

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
            }

    async def _scrape(self, sequential: bool = False) -> Dict[str, Dict[str, int]]:
        """Execute scraping phase."""
        pipeline = ScraperPipeline(self.scrapers)

        if sequential:
            return await pipeline.run_sequential()
        else:
            return await pipeline.run_parallel(max_concurrent=3)

    def _aggregate_data(self):
        """Aggregate data from all scrapers."""
        for scraper in self.scrapers:
            if not self.raw_data:
                self.raw_data = {
                    "matches": [],
                    "maps": [],
                    "compositions": [],
                    "player_stats": [],
                    "rounds": [],
                }

            self.raw_data["matches"].extend(scraper.matches)
            self.raw_data["maps"].extend(scraper.maps)
            self.raw_data["compositions"].extend(scraper.compositions)
            self.raw_data["player_stats"].extend(scraper.player_stats)
            self.raw_data["rounds"].extend(scraper.rounds)

        logger.info(f"Aggregated raw data: {self._get_data_stats(self.raw_data)}")

    def _parse_data(self):
        """Parse and normalize raw data."""
        self.processed_data = DataPipeline.process_all(self.raw_data)

        stats = {
            "matches": len(self.processed_data.get("matches", [])),
            "maps": len(self.processed_data.get("maps", [])),
            "compositions": len(self.processed_data.get("compositions", [])),
            "player_stats": len(self.processed_data.get("player_stats", [])),
            "rounds": len(self.processed_data.get("rounds", [])),
        }
        logger.info(f"Parsed data: {stats}")

    def _export_data(self):
        """Export processed data to files."""
        if self.output_format == "parquet":
            self.export_stats = self._export_parquet()
        else:
            self.export_stats = self._export_csv()

        logger.info(f"Export complete: {self.export_stats}")

    def _export_csv(self) -> Dict[str, int]:
        """Export to CSV files."""
        stats = {}

        exports = {
            "matches.csv": self.processed_data.get("matches", []),
            "maps.csv": self.processed_data.get("maps", []),
            "compositions.csv": self.processed_data.get("compositions", []),
            "player_stats.csv": self.processed_data.get("player_stats", []),
            "rounds.csv": self.processed_data.get("rounds", []),
        }

        for filename, data in exports.items():
            if data:
                output_path = self.data_dir / "processed" / filename
                rows = CSVManager.list_dicts_to_csv(data, output_path, append=True)
                stats[filename] = rows

        return stats

    def _export_parquet(self) -> Dict[str, int]:
        """Export to Parquet files."""
        import pandas as pd
        stats = {}

        exports = {
            "matches.parquet": self.processed_data.get("matches", []),
            "maps.parquet": self.processed_data.get("maps", []),
            "compositions.parquet": self.processed_data.get("compositions", []),
            "player_stats.parquet": self.processed_data.get("player_stats", []),
            "rounds.parquet": self.processed_data.get("rounds", []),
        }

        for filename, data in exports.items():
            if data:
                df = pd.DataFrame(data)
                output_path = self.data_dir / "parquet" / filename
                CSVManager.to_parquet(df, output_path)
                stats[filename] = len(df)

        return stats

    def _get_data_stats(self, data: Optional[Dict[str, List[Dict[str, Any]]]] = None
                        ) -> Dict[str, int]:
        """Get statistics about data."""
        if data is None:
            data = self.processed_data

        return {
            "matches": len(data.get("matches", [])),
            "maps": len(data.get("maps", [])),
            "compositions": len(data.get("compositions", [])),
            "player_stats": len(data.get("player_stats", [])),
            "rounds": len(data.get("rounds", [])),
        }

    def save_config(self, config_path: str = "pipeline_config.json"):
        """Save pipeline configuration."""
        config = {
            "timestamp": datetime.now().isoformat(),
            "scrapers": [s.name for s in self.scrapers],
            "output_format": self.output_format,
            "data_directory": str(self.data_dir),
        }

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved config to {config_path}")

    def generate_report(self) -> Dict[str, Any]:
        """Generate execution report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "scrapers": [s.name for s in self.scrapers],
            "raw_data_stats": self._get_data_stats(self.raw_data),
            "processed_data_stats": self._get_data_stats(self.processed_data),
            "export_stats": self.export_stats,
            "data_directory": str(self.data_dir),
        }


class ScheduledPipeline:
    """Run pipeline on a schedule."""

    def __init__(self, pipeline: ValolyzerPipeline, interval_seconds: int = 3600):
        """
        Initialize scheduled pipeline.

        Args:
            pipeline: ValolyzerPipeline instance
            interval_seconds: Interval between runs
        """
        self.pipeline = pipeline
        self.interval_seconds = interval_seconds
        self.running = False

    async def run_loop(self):
        """Run pipeline in a loop."""
        self.running = True

        while self.running:
            try:
                logger.info("Running scheduled pipeline")
                result = await self.pipeline.run()

                if result["status"] == "success":
                    logger.info(f"Pipeline completed successfully")
                else:
                    logger.error(f"Pipeline failed: {result['error']}")

            except Exception as e:
                logger.error(f"Error in pipeline loop: {e}")

            # Wait for next interval
            logger.info(f"Next run in {self.interval_seconds} seconds")
            await asyncio.sleep(self.interval_seconds)

    def stop(self):
        """Stop scheduled pipeline."""
        self.running = False
        logger.info("Stopping scheduled pipeline")


if __name__ == "__main__":
    print("Pipeline module loaded")
