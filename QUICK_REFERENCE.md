"""
Valolyzer Quick Reference - Common Commands & Code Snippets
"""

# ============================================================================
# CLI COMMANDS
# ============================================================================

"""
# Run complete pipeline (scrape + parse + export)
python main.py full

# Export as Parquet (compressed format)
python main.py full --format parquet

# Run scrapers sequentially (safer, slower)
python main.py full --sequential

# Run scrapers only
python main.py scrape

# Parse existing raw data
python main.py parse

# Show data statistics
python main.py status

# Schedule recurring runs (every hour)
python main.py schedule --interval 3600
"""

# ============================================================================
# PYTHON API - BASIC USAGE
# ============================================================================

import asyncio
from scrapers.vlr import VLRScraper
from pipelines import ValolyzerPipeline
from utils.logging import get_logger

logger = get_logger(__name__)


async def example_basic_usage():
    """Minimal pipeline example."""
    # Create scraper
    scraper = VLRScraper(rate_limit=1.0)

    # Create pipeline
    pipeline = ValolyzerPipeline([scraper])

    # Run
    result = await pipeline.run()

    # Check result
    if result["status"] == "success":
        print(f"✓ Scraped {result['export_stats']}")
    else:
        print(f"✗ Failed: {result['error']}")


# ============================================================================
# PYTHON API - ADVANCED USAGE
# ============================================================================


async def example_advanced():
    """Advanced usage with configuration."""
    from config import Config, PipelineConfig

    # Configure pipeline
    Config.pipeline = PipelineConfig(
        output_format="parquet",
        parallel_scrapers=True,
        max_concurrent_scrapers=3,
        deduplicate_on_export=True,
    )

    # Create scraper with custom rate limit
    scraper = VLRScraper(rate_limit=0.5, timeout=60)

    # Create pipeline
    pipeline = ValolyzerPipeline(
        [scraper],
        output_format="parquet",
    )

    # Run sequentially (safer)
    result = await pipeline.run(sequential=False)

    # Generate report
    if result["status"] == "success":
        report = pipeline.generate_report()
        pipeline.save_config("config.json")
        print(report)


# ============================================================================
# DATA NORMALIZATION
# ============================================================================

def example_normalizers():
    """Using data normalizers."""
    from utils.normalizers import Normalizers

    # Team names
    print(Normalizers.normalize_team_name("prx"))  # → "Paper Rex"
    print(Normalizers.normalize_team_name("geng"))  # → "Gen.G"

    # Agents
    print(Normalizers.normalize_agent_name("jett"))  # → "Jett"
    print(Normalizers.normalize_agent_name("kay/o"))  # → "KAY/O"

    # Maps
    print(Normalizers.normalize_map_name("bind"))  # → "Bind"

    # Patches
    print(Normalizers.normalize_patch("Episode 8, Act 1"))  # → "8.01"

    # Extract teams from match title
    team_a, team_b = Normalizers.extract_team_names("Paper Rex vs Gen.G")
    print(team_a, team_b)  # → "Paper Rex" "Gen.G"


# ============================================================================
# CSV/DATA MANAGEMENT
# ============================================================================

def example_csv_management():
    """Using CSV utilities."""
    from utils.csv_handler import CSVManager

    # Read CSV
    df = CSVManager.read_csv("data/processed/matches.csv", engine="pandas")

    # Read specific columns
    df = CSVManager.read_csv("data/processed/matches.csv")
    if df is not None:
        df = df[["match_id", "team_a", "team_b", "winner"]]

    # Get statistics
    stats = CSVManager.get_stats("data/processed/matches.csv")
    print(f"Rows: {stats['rows']}")
    print(f"Columns: {stats['columns']}")
    print(f"Memory: {stats['memory_usage_mb']:.2f} MB")

    # Write CSV
    CSVManager.write_csv(df, "data/output.csv")

    # Append to CSV (with deduplication)
    new_data = [
        {"match_id": "1", "team_a": "Paper Rex", "team_b": "Gen.G"},
        {"match_id": "2", "team_a": "OpTic", "team_b": "FaZe"},
    ]
    CSVManager.list_dicts_to_csv(new_data, "data/matches.csv")

    # Convert to Parquet
    CSVManager.to_parquet(df, "data/matches.parquet")

    # Read Parquet
    df_pq = CSVManager.read_parquet("data/matches.parquet", engine="polars")

    # Merge multiple CSVs
    total = CSVManager.merge_csvs(
        ["data/matches_1.csv", "data/matches_2.csv"],
        "data/matches_combined.csv",
        deduplicate_on=["match_id"]
    )


# ============================================================================
# CUSTOM PARSER EXAMPLE
# ============================================================================

def example_custom_parser():
    """Creating a custom parser."""
    from parsers.data_parser import MatchParser
    from utils.normalizers import Normalizers

    # Raw data from scraper
    raw_match = {
        "id": "542195",
        "event_name": "Valorant Champions 2025",
        "date": "2025-09-12",
        "patch": "11.05",
        "bo_format": "Bo3",
        "team_1": "prx",
        "team_2": "geng",
        "winner": "prx",
        "score1": 2,
        "score2": 0,
    }

    # Parse using MatchParser
    parsed = MatchParser.parse(raw_match)

    # Validate
    if MatchParser.validate(parsed):
        print(f"✓ Valid match: {parsed}")
    else:
        print(f"✗ Invalid match")


# ============================================================================
# DATA ANALYSIS
# ============================================================================

def example_data_analysis():
    """Analyzing scraped data."""
    import pandas as pd
    from utils.csv_handler import CSVManager

    # Load matches
    matches = CSVManager.read_csv("data/processed/matches.csv")
    if matches is None:
        return

    # Win rate by team
    print("\n=== Top Winning Teams ===")
    wins = matches["winner"].value_counts()
    print(wins.head(10))

    # Matches by event
    print("\n=== Matches by Event ===")
    events = matches["event"].value_counts()
    print(events.head(10))

    # Patch distribution
    print("\n=== Patches Represented ===")
    patches = matches["patch"].value_counts()
    print(patches)

    # Bo type distribution
    print("\n=== Best-of Formats ===")
    bo_types = matches["bo_type"].value_counts()
    print(bo_types)

    # Load player stats
    print("\n=== Top Players (by kills) ===")
    player_stats = CSVManager.read_csv("data/processed/player_stats.csv")
    if player_stats is not None:
        top_players = player_stats.nlargest(10, "kills")[
            ["player", "team", "agent", "kills", "deaths", "acs"]
        ]
        print(top_players)


# ============================================================================
# SCHEDULING & AUTOMATION
# ============================================================================

def example_schedule():
    """Scheduling pipeline runs."""
    import asyncio
    from scrapers.vlr import VLRScraper
    from pipelines import ValolyzerPipeline, ScheduledPipeline

    async def main():
        # Create pipeline
        scraper = VLRScraper()
        pipeline = ValolyzerPipeline([scraper])

        # Schedule for hourly runs
        scheduled = ScheduledPipeline(pipeline, interval_seconds=3600)

        # Run (will loop forever until stopped)
        try:
            await scheduled.run_loop()
        except KeyboardInterrupt:
            scheduled.stop()

    # asyncio.run(main())


# ============================================================================
# ERROR HANDLING
# ============================================================================

def example_error_handling():
    """Handling errors gracefully."""
    import asyncio
    from scrapers.vlr import VLRScraper
    from utils.logging import get_logger

    logger = get_logger(__name__)

    async def safe_run():
        scraper = VLRScraper()

        try:
            result = await scraper.run()
            if result:
                logger.info(f"Success: {result}")
        except Exception as e:
            logger.error(f"Scraper failed: {e}", exc_info=True)
        finally:
            await scraper.cleanup()

    # asyncio.run(safe_run())


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def example_logging():
    """Configure logging."""
    from utils.logging import Logger, get_logger

    # Configure with custom settings
    Logger.configure(
        log_dir="logs",
        level="DEBUG",
        rotation="500 MB",
        retention="7 days",
    )

    # Get logger for your module
    logger = get_logger(__name__)

    # Use logger
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")


# ============================================================================
# PERFORMANCE TIPS
# ============================================================================

"""
1. Use Parquet Format for Large Datasets
   - 10-100x compression vs CSV
   - Faster reads for analytics
   - Example: pipeline = ValolyzerPipeline([scraper], output_format="parquet")

2. Reduce Rate Limit for Stability
   - Start with 0.5 requests/second
   - Increase only if stable
   - Example: VLRScraper(rate_limit=0.5)

3. Process in Batches
   - Don't load all data in memory at once
   - Use chunking for large datasets
   - Recommended chunk size: 1000 rows

4. Enable Parallel Scraping
   - Run multiple scrapers concurrently
   - Max 3 concurrent by default
   - Example: pipeline.run(sequential=False)

5. Use Sequential Mode for Debugging
   - Slower but more stable
   - Better error messages
   - Example: python main.py full --sequential

6. Check Logs Regularly
   - Logs stored in logs/ directory
   - Rotate every 500MB
   - Keep for 7 days
"""

# ============================================================================
# QUICK COMMAND REFERENCE
# ============================================================================

"""
# Installation
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install

# Basic Operations
python main.py full              # Full pipeline
python main.py scrape            # Scrape only
python main.py parse             # Parse only
python main.py status            # Show statistics
python main.py schedule          # Schedule runs

# Advanced
python main.py full --format parquet     # Export as Parquet
python main.py full --sequential         # Sequential mode
python examples.py                       # Run examples
python -c "import valolyzer"            # Test imports

# Data Operations
cat data/processed/matches.csv | head    # Preview data
wc -l data/processed/*.csv               # Count rows
du -sh data/                             # Check size
"""

# ============================================================================
# FILE STRUCTURE QUICK REFERENCE
# ============================================================================

"""
valolyzer/
├── data/
│   ├── raw/              (Raw scraper output - optional)
│   ├── processed/        (Normalized CSVs - PRIMARY)
│   └── parquet/          (Compressed Parquet files)
│
├── scrapers/
│   ├── base.py           (Abstract base class)
│   ├── vlr/              (VLR.gg scraper - ACTIVE)
│   ├── rib/              (RIB.gg scraper - TODO)
│   └── tracker/          (Tracker.gg scraper - TODO)
│
├── parsers/
│   └── data_parser.py    (Normalization & parsing)
│
├── pipelines/
│   └── main_pipeline.py  (Orchestration)
│
├── utils/                (Shared utilities)
├── logs/                 (Execution logs)
│
├── main.py               (CLI entry point)
├── config.py             (Configuration)
├── requirements.txt      (Dependencies)
└── README.md             (Documentation)
"""

if __name__ == "__main__":
    print("Valolyzer Quick Reference Guide")
    print("Run individual examples as needed")
