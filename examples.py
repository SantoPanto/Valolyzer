"""
Example usage scenarios for Valolyzer pipeline.
Demonstrates common workflows and customization options.
"""

import asyncio
from pathlib import Path
from scrapers.vlr import VLRScraper
from pipelines import ValolyzerPipeline
from utils.csv_handler import CSVManager
from utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Example 1: Basic Pipeline Run
# ============================================================================

async def example_basic_pipeline():
    """
    Simple usage: Run scraper, parse, and export.
    """
    logger.info("=== Example 1: Basic Pipeline ===")

    # Create scraper
    scraper = VLRScraper(rate_limit=1.0)

    # Create pipeline
    pipeline = ValolyzerPipeline([scraper], output_format="csv")

    # Run
    result = await pipeline.run()

    if result["status"] == "success":
        logger.info(f"Success! Exported: {result['export_stats']}")
    else:
        logger.error(f"Failed: {result['error']}")


# ============================================================================
# Example 2: Parquet Export for Analytics
# ============================================================================

async def example_parquet_export():
    """
    Export to Parquet format for efficient analytics.
    Better compression, faster reads for large datasets.
    """
    logger.info("=== Example 2: Parquet Export ===")

    scraper = VLRScraper(rate_limit=1.0)
    pipeline = ValolyzerPipeline([scraper], output_format="parquet")

    result = await pipeline.run()

    if result["status"] == "success":
        # Read parquet files
        matches_df = CSVManager.read_parquet(
            "data/parquet/matches.parquet",
            engine="polars"
        )
        logger.info(f"Loaded {len(matches_df)} matches from Parquet")


# ============================================================================
# Example 3: Sequential Scraping (Safe Mode)
# ============================================================================

async def example_sequential_scraping():
    """
    Run scrapers sequentially instead of parallel.
    Useful for debugging or avoiding rate limit issues.
    """
    logger.info("=== Example 3: Sequential Scraping ===")

    scrapers = [
        VLRScraper(rate_limit=0.5),  # Conservative rate limit
    ]

    pipeline = ValolyzerPipeline(scrapers)

    # Run sequentially
    result = await pipeline.run(sequential=True)

    logger.info(f"Result: {result['status']}")


# ============================================================================
# Example 4: Incremental Updates with Deduplication
# ============================================================================

async def example_incremental_update():
    """
    Run pipeline in append mode to add new data without duplicates.
    Useful for regular scheduled updates.
    """
    logger.info("=== Example 4: Incremental Updates ===")

    scraper = VLRScraper(rate_limit=1.0)
    pipeline = ValolyzerPipeline([scraper])

    # Run scraper
    scrape_stats = await pipeline.run()

    if scrape_stats["status"] == "success":
        # Data is automatically deduplicated on key fields
        logger.info("Data appended with automatic deduplication")

        # Check statistics
        stats = CSVManager.get_stats("data/processed/matches.csv")
        logger.info(f"Total matches: {stats['rows']}")


# ============================================================================
# Example 5: Custom Rate Limiting
# ============================================================================

async def example_custom_rate_limit():
    """
    Configure custom rate limiting for different sources.
    """
    logger.info("=== Example 5: Custom Rate Limiting ===")

    # Conservative rate limit to avoid blocking
    scraper = VLRScraper(
        rate_limit=0.5,  # 1 request every 2 seconds
        timeout=60,       # 60 second timeout
    )

    pipeline = ValolyzerPipeline([scraper])
    result = await pipeline.run()

    logger.info(f"Completed with custom rate limiting")


# ============================================================================
# Example 6: Data Analysis
# ============================================================================

def example_data_analysis():
    """
    Analyze exported data using pandas/polars.
    """
    logger.info("=== Example 6: Data Analysis ===")

    # Read processed data
    matches_df = CSVManager.read_csv(
        "data/processed/matches.csv",
        engine="pandas"
    )

    if matches_df is None:
        logger.warning("No match data found")
        return

    # Example analyses
    logger.info(f"Total matches: {len(matches_df)}")

    # Win rate by team
    if "winner" in matches_df.columns:
        wins = matches_df["winner"].value_counts()
        logger.info(f"Top winning teams:\n{wins.head()}")

    # Patch distribution
    if "patch" in matches_df.columns:
        patches = matches_df["patch"].value_counts()
        logger.info(f"Patches represented: {len(patches)}")


# ============================================================================
# Example 7: Generate Report
# ============================================================================

def example_generate_report():
    """
    Generate execution report and statistics.
    """
    logger.info("=== Example 7: Report Generation ===")

    # Check all data directories
    stats = {}

    for data_type in ["matches", "maps", "compositions", "player_stats", "rounds"]:
        csv_path = Path(f"data/processed/{data_type}.csv")
        if csv_path.exists():
            file_stats = CSVManager.get_stats(csv_path)
            stats[data_type] = file_stats

    logger.info("Data Statistics Report:")
    for data_type, file_stats in stats.items():
        logger.info(f"\n{data_type}:")
        logger.info(f"  Rows: {file_stats.get('rows')}")
        logger.info(f"  Columns: {file_stats.get('columns')}")
        logger.info(f"  Memory: {file_stats.get('memory_usage_mb'):.2f} MB")


# ============================================================================
# Example 8: Merge Multiple Sources
# ============================================================================

def example_merge_sources():
    """
    Merge data from multiple scrapers into unified dataset.
    """
    logger.info("=== Example 8: Merge Multiple Sources ===")

    # Example: Merge match data from different scrapes
    csv_files = list(Path("data/raw").glob("matches_*.csv"))

    if csv_files:
        total_rows = CSVManager.merge_csvs(
            csv_files,
            "data/processed/matches_merged.csv",
            deduplicate_on=["match_id"]
        )
        logger.info(f"Merged {len(csv_files)} files into {total_rows} rows")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run examples."""
    examples = [
        ("Basic Pipeline", example_basic_pipeline),
        # ("Parquet Export", example_parquet_export),
        # ("Sequential Scraping", example_sequential_scraping),
        # ("Incremental Updates", example_incremental_update),
        # ("Custom Rate Limit", example_custom_rate_limit),
    ]

    for name, example_func in examples:
        try:
            if asyncio.iscoroutinefunction(example_func):
                await example_func()
            else:
                example_func()
        except Exception as e:
            logger.error(f"Example '{name}' failed: {e}", exc_info=True)

        logger.info("-" * 60)

    # Non-async examples
    logger.info("Running non-async examples...")
    example_data_analysis()
    logger.info("-" * 60)
    example_generate_report()
    logger.info("-" * 60)
    example_merge_sources()


if __name__ == "__main__":
    asyncio.run(main())
