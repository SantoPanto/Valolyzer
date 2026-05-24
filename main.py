"""
Valolyzer - Valorant Analytics and ML Pipeline
Main entry point for scraping, parsing, and exporting Valorant esports data.

Usage:
    python main.py scrape     # Run scrapers
    python main.py parse      # Parse data
    python main.py full       # Run complete pipeline
    python main.py schedule   # Run on schedule
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import Optional

from scrapers.vlr import VLRScraper
from pipelines import ValolyzerPipeline, ScheduledPipeline
from utils.logging import Logger, get_logger

# Initialize logging
Logger.configure()
logger = get_logger(__name__)


class ValolyzerCLI:
    """Command-line interface for Valolyzer."""

    def __init__(self):
        """Initialize CLI."""
        self.pipeline: Optional[ValolyzerPipeline] = None

    async def command_scrape(self, args):
        """Run scrapers only."""
        logger.info("Running scrapers...")

        scrapers = [
            VLRScraper(rate_limit=1.0),
        ]

        pipeline = ValolyzerPipeline(scrapers)
        result = await pipeline.run(sequential=args.sequential)

        self._print_result(result)
        return result

    async def command_parse(self, args):
        """Parse existing raw data."""
        logger.info("Parsing data...")

        from parsers import DataPipeline
        from utils.csv_handler import CSVManager

        # Read raw data
        raw_data = {
            "matches": [],
            "maps": [],
            "compositions": [],
            "player_stats": [],
            "rounds": [],
        }

        # Try to read from raw directory
        raw_dir = Path("data/raw")
        if raw_dir.exists():
            for data_type in raw_data.keys():
                csv_file = raw_dir / f"{data_type}.csv"
                if csv_file.exists():
                    df = CSVManager.read_csv(csv_file)
                    if df is not None:
                        raw_data[data_type] = CSVManager.DataFrameConverter.dataframe_to_records(df)

        # Parse data
        processed = DataPipeline.process_all(raw_data)
        logger.info(f"Parsed {sum(len(v) for v in processed.values())} total records")

        return {"status": "success", "processed_data": processed}

    async def command_full(self, args):
        """Run complete pipeline."""
        logger.info("Running complete pipeline...")

        scrapers = [
            VLRScraper(rate_limit=1.0),
        ]

        pipeline = ValolyzerPipeline(
            scrapers,
            output_format=args.format
        )

        result = await pipeline.run(sequential=args.sequential)
        self._print_result(result)

        if result["status"] == "success":
            # Generate report
            report = pipeline.generate_report()
            pipeline.save_config()
            logger.info("Report generated successfully")

        return result

    async def command_schedule(self, args):
        """Run pipeline on a schedule."""
        logger.info(f"Starting scheduled pipeline (interval: {args.interval} seconds)")

        scrapers = [
            VLRScraper(rate_limit=1.0),
        ]

        pipeline = ValolyzerPipeline(scrapers)
        scheduled = ScheduledPipeline(pipeline, interval_seconds=args.interval)

        try:
            await scheduled.run_loop()
        except KeyboardInterrupt:
            logger.info("Stopping scheduled pipeline")
            scheduled.stop()

    async def command_status(self, args):
        """Show data status."""
        from utils.csv_handler import CSVManager

        logger.info("Data Status Report")
        logger.info("=" * 50)

        processed_dir = Path("data/processed")
        if processed_dir.exists():
            for csv_file in processed_dir.glob("*.csv"):
                stats = CSVManager.get_stats(csv_file)
                logger.info(f"\n{csv_file.name}:")
                logger.info(f"  Rows: {stats.get('rows', 0)}")
                logger.info(f"  Columns: {stats.get('columns', 0)}")
                logger.info(f"  Memory: {stats.get('memory_usage_mb', 0):.2f} MB")

    def _print_result(self, result: dict):
        """Pretty print result."""
        logger.info("Pipeline Result:")
        logger.info(f"  Status: {result.get('status')}")

        if result.get("status") == "success":
            logger.info(f"  Duration: {result.get('duration_seconds'):.2f} seconds")

            if result.get("export_stats"):
                logger.info("  Export Stats:")
                for key, value in result.get("export_stats", {}).items():
                    logger.info(f"    {key}: {value} rows")

        elif result.get("error"):
            logger.error(f"  Error: {result['error']}")

    async def main(self):
        """Main entry point."""
        parser = argparse.ArgumentParser(
            description="Valolyzer - Valorant Analytics Pipeline"
        )

        subparsers = parser.add_subparsers(dest="command", help="Command to run")

        # Scrape command
        scrape_parser = subparsers.add_parser("scrape", help="Run scrapers")
        scrape_parser.add_argument(
            "--sequential",
            action="store_true",
            help="Run scrapers sequentially instead of parallel"
        )

        # Parse command
        parse_parser = subparsers.add_parser("parse", help="Parse raw data")

        # Full pipeline command
        full_parser = subparsers.add_parser("full", help="Run complete pipeline")
        full_parser.add_argument(
            "--format",
            choices=["csv", "parquet"],
            default="csv",
            help="Output format"
        )
        full_parser.add_argument(
            "--sequential",
            action="store_true",
            help="Run scrapers sequentially"
        )

        # Schedule command
        schedule_parser = subparsers.add_parser("schedule", help="Run on schedule")
        schedule_parser.add_argument(
            "--interval",
            type=int,
            default=3600,
            help="Interval between runs in seconds (default: 3600)"
        )

        # Status command
        status_parser = subparsers.add_parser("status", help="Show data status")

        args = parser.parse_args()

        if not args.command:
            parser.print_help()
            return

        # Route to command handler
        command_method = getattr(self, f"command_{args.command}", None)
        if command_method:
            return await command_method(args)
        else:
            logger.error(f"Unknown command: {args.command}")


async def main():
    """Entry point."""
    cli = ValolyzerCLI()
    await cli.main()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
