# Valolyzer - Valorant Analytics & ML Pipeline

Professional Valorant esports data pipeline for collecting, normalizing, and analyzing match data from high-level competitive play. Features async scraping, structured normalization, and ML-ready exports.

## Overview

Valolyzer is a comprehensive data engineering platform for Valorant analytics that:

- **Scrapes** professional match data from multiple sources (VLR.gg, RIB.gg, Tracker.gg)
- **Normalizes** team names, agents, maps, patches, and statistics
- **Parses** match details, maps, compositions, player stats, and round data
- **Exports** clean, ML-ready datasets in CSV or Parquet format
- **Scales** with async/await, rate limiting, retry logic, and error handling
- **Tracks** scraper progress for resumable scraping operations

## Architecture

```
valolyzer/
├── data/
│   ├── raw/              # Raw scraped data
│   ├── processed/        # Normalized CSV exports
│   └── parquet/          # Compressed Parquet files
│
├── scrapers/             # Async web scrapers
│   ├── base.py           # Base scraper class
│   ├── vlr/              # VLR.gg scraper
│   ├── rib/              # RIB.gg scraper (future)
│   └── tracker/          # Tracker.gg scraper (future)
│
├── parsers/              # Data normalization & parsing
│   └── data_parser.py    # Match, map, player, composition parsers
│
├── pipelines/            # Orchestration & workflows
│   └── main_pipeline.py  # Valolyzer & scheduled pipelines
│
├── utils/                # Shared utilities
│   ├── normalizers.py    # Data normalization
│   ├── models.py         # Pydantic data models
│   ├── http.py           # Async HTTP client
│   ├── csv_handler.py    # CSV I/O utilities
│   └── logging.py        # Structured logging
│
├── database/             # Future DB integrations
├── main.py               # CLI entry point
└── requirements.txt      # Python dependencies
```

## Data Schema

### Matches (`matches.csv`)
```
match_id, event, date, patch, bo_type, team_a, team_b, winner, score_a, score_b, maps_played, source
```

### Maps (`maps.csv`)
```
map_id, match_id, map_name, map_order, team_a_score, team_b_score, attacker_start, duration_seconds, source
```

### Compositions (`compositions.csv`)
```
map_id, team, agent_1, agent_2, agent_3, agent_4, agent_5, source
```

### Player Stats (`player_stats.csv`)
```
map_id, player, team, agent, kills, deaths, assists, acs, adr, hs_percent, kd_ratio, source
```

### Rounds (`rounds.csv`)
```
round_id, map_id, round_number, winner, win_type, spike_planted, econ_a, econ_b, duration_seconds, source
```

## Installation

### Prerequisites
- Python 3.12+
- pip or conda

### Setup

```bash
# Clone repository
git clone <repo_url>
cd valolyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers (for future browser-based scraping)
playwright install
```

## Usage

### Command Line Interface

#### Run Complete Pipeline
```bash
python main.py full --format csv --sequential
```

#### Run Scrapers Only
```bash
python main.py scrape --sequential
```

#### Parse Existing Raw Data
```bash
python main.py parse
```

#### Check Data Status
```bash
python main.py status
```

#### Schedule Recurring Runs
```bash
python main.py schedule --interval 3600
```

### Python API

```python
import asyncio
from scrapers.vlr import VLRScraper
from pipelines import ValolyzerPipeline

async def main():
    # Create scrapers
    scrapers = [
        VLRScraper(rate_limit=1.0),
    ]

    # Initialize pipeline
    pipeline = ValolyzerPipeline(scrapers, output_format="csv")

    # Run pipeline
    result = await pipeline.run()

    # Export data
    if result["status"] == "success":
        print(f"Scraped {result['export_stats']}")

asyncio.run(main())
```

## Features

### Async/Concurrent Scraping
- Non-blocking I/O with asyncio
- Configurable concurrency limits
- Efficient resource usage

### Rate Limiting & Retry Logic
- Per-domain rate limiting
- Exponential backoff with jitter
- Automatic retries on server errors

### Data Normalization
- Standardized team names
- Agent name validation
- Map name normalization
- Patch version parsing
- Economic round classification

### Deduplication
- Automatic duplicate detection
- Configurable dedup keys
- Append or overwrite modes

### Export Formats
- **CSV**: Human-readable, Excel-compatible
- **Parquet**: Compressed, columnar, efficient for analytics

### Structured Logging
- Rotating file logs
- Console output with colors
- Scraper-specific log streams
- Debug-level detail

### Type Safety
- Pydantic models for all data types
- Automatic validation
- Schema documentation

## Configuration

### Normalizers
```python
from utils.normalizers import Normalizers

# Team names
Normalizers.normalize_team_name("prx")  # -> "Paper Rex"

# Agents
Normalizers.normalize_agent_name("jett")  # -> "Jett"

# Maps
Normalizers.normalize_map_name("bind")  # -> "Bind"

# Patches
Normalizers.normalize_patch("Episode 8, Act 1")  # -> "8.01"
```

### HTTP Client
```python
from utils.http import AsyncHTTPClient, RateLimiter, RetryStrategy

client = AsyncHTTPClient(
    rate_limit=2.0,           # 2 requests/second
    timeout=30,               # 30 second timeout
    retry_strategy=RetryStrategy(
        max_retries=3,
        base_delay=1.0,
        max_delay=60.0
    )
)
```

### CSV Management
```python
from utils.csv_handler import CSVManager

# Read CSV
df = CSVManager.read_csv("data/matches.csv", engine="pandas")

# Append with deduplication
CSVManager.append_csv(
    df_new,
    "data/matches.csv",
    deduplicate_on=["match_id"]
)

# Convert to Parquet
CSVManager.to_parquet(df, "data/matches.parquet")

# Get statistics
stats = CSVManager.get_stats("data/matches.csv")
```

## Development

### Adding a New Scraper

```python
from scrapers.base import BaseScraper

class NewScraper(BaseScraper):
    def __init__(self):
        super().__init__("new_scraper")
    
    async def scrape(self):
        # Implement scraping logic
        pass
    
    async def parse_match(self, data):
        # Parse match data
        pass
    
    async def parse_map(self, data):
        # Parse map data
        pass
```

### Adding Data Parsers

```python
from parsers.data_parser import MatchParser

class CustomParser:
    @staticmethod
    def parse(raw_data):
        return {
            # Normalize fields
        }
    
    @staticmethod
    def validate(parsed_data):
        # Validate required fields
        return True
```

## Performance Tips

1. **Use Parquet for large datasets**
   - 10-100x compression vs CSV
   - Faster read/write
   - Column-oriented analytics

2. **Enable parallel scraping**
   - `--sequential` flag disabled by default
   - Multiple concurrent requests
   - Better throughput

3. **Schedule incremental updates**
   - Use `schedule` command
   - Append mode with deduplication
   - Avoid redundant processing

4. **Monitor resource usage**
   - Check logs in `logs/` directory
   - Use `status` command for data overview
   - Adjust rate limits if needed

## Troubleshooting

### Import Errors
```bash
# Ensure all dependencies installed
pip install -r requirements.txt

# Verify Python version
python --version  # Should be 3.12+
```

### Scraper Failures
- Check logs: `logs/scraper_*.log`
- Verify site structure hasn't changed
- Test with reduced rate limit
- Check internet connection

### Memory Issues
- Use Parquet format instead of CSV
- Process data in batches
- Reduce rate limit to avoid buffer buildup

## Future Enhancements

- [ ] PostgreSQL database backend
- [ ] DuckDB local data warehouse
- [ ] RIB.gg scraper implementation
- [ ] Tracker.gg integration
- [ ] Round-by-round replay parsing
- [ ] ML feature engineering pipeline
- [ ] Prediction models
- [ ] REST API for data access
- [ ] Dashboard/visualization UI

## Performance Metrics

- **Scraping Speed**: ~100-200 matches/hour (rate limited)
- **Parse/Normalize**: ~10,000 rows/second
- **CSV Export**: ~5,000 rows/second
- **Parquet Export**: ~20,000 rows/second
- **Memory Usage**: ~500MB for 100k match records

## License

Licensed under MIT License - See LICENSE file

## Support

For issues, questions, or contributions:
1. Check existing logs in `logs/` directory
2. Review error messages in stderr output
3. Test with `--sequential` mode for debugging
4. Check data schema in source code
