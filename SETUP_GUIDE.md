# Valolyzer Setup & Development Guide

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers (for future browser automation)
playwright install
```

### 2. First Run

```bash
# Check data status (should be empty initially)
python main.py status

# Run complete pipeline
python main.py full --format csv

# Check exported data
ls -la data/processed/
```

### 3. Verify Installation

```bash
# Test imports
python -c "from scrapers import VLRScraper; from pipelines import ValolyzerPipeline; print('✓ Installation OK')"

# Run examples
python examples.py
```

## Architecture Deep Dive

### Scraper Hierarchy

```
BaseScraper (abstract)
├── VLRScraper (vlr.gg)
├── RIBScraper (rib.gg) - future
└── TrackerScraper (tracker.gg) - future
```

**BaseScraper provides:**
- HTTP client management
- Rate limiting
- Retry logic
- Data export (CSV/Parquet)
- State tracking
- Error handling

### Data Flow

```
Raw HTML/JSON
    ↓
[VLRScraper.scrape()]
    ↓
Raw dictionaries (matches, maps, etc.)
    ↓
[DataPipeline.process_all()]
    ↓
Normalized dictionaries
    ↓
[CSVManager.append_csv()]
    ↓
CSV/Parquet files (data/processed/)
```

### Component Responsibilities

**Scrapers:**
- Fetch HTML/JSON from sources
- Extract raw data
- Return structured dictionaries
- Handle HTTP errors and retries

**Parsers:**
- Normalize raw data
- Validate required fields
- Transform data types
- Apply business rules

**Pipeline:**
- Orchestrate scrapers
- Aggregate results
- Process through parsers
- Export to files

**Utils:**
- Normalize team/agent/map names
- Handle HTTP requests
- Manage CSV/Parquet files
- Provide logging

## Adding New Scrapers

### Step 1: Create Scraper File

```python
# scrapers/rib/rib_scraper.py
from scrapers.base import BaseScraper
from utils.normalizers import Normalizers
from utils.logging import get_logger

logger = get_logger(__name__)

class RIBScraper(BaseScraper):
    """Scraper for rib.gg - Secondary Valorant data source."""

    BASE_URL = "https://www.rib.gg"

    def __init__(self, *args, **kwargs):
        super().__init__("rib_scraper", *args, **kwargs)

    async def scrape(self):
        """Main scraping method."""
        # Implement scraping logic
        return {
            "matches": self.matches,
            "maps": self.maps,
            "compositions": self.compositions,
            "player_stats": self.player_stats,
            "rounds": self.rounds,
        }

    async def parse_match(self, match_data):
        # Implement match parsing
        pass

    async def parse_map(self, map_data):
        # Implement map parsing
        pass
```

### Step 2: Add to Exports

```python
# scrapers/__init__.py
from scrapers.rib.rib_scraper import RIBScraper

__all__ = [..., "RIBScraper"]
```

### Step 3: Test

```python
import asyncio
from scrapers.rib import RIBScraper

async def test():
    scraper = RIBScraper()
    try:
        result = await scraper.run()
        print(f"Result: {result}")
    finally:
        await scraper.cleanup()

asyncio.run(test())
```

## Adding New Parsers

### Example: Custom Match Parser

```python
# parsers/custom_parser.py
from utils.normalizers import Normalizers

class CustomMatchParser:
    @staticmethod
    def parse(raw_match):
        """Transform raw match data."""
        return {
            "match_id": raw_match.get("id"),
            "event": raw_match.get("event_name"),
            "team_a": Normalizers.normalize_team_name(raw_match.get("team1")),
            "team_b": Normalizers.normalize_team_name(raw_match.get("team2")),
            # ... other fields
        }

    @staticmethod
    def validate(parsed_match):
        """Validate required fields."""
        required = ["match_id", "team_a", "team_b"]
        return all(parsed_match.get(field) for field in required)
```

## Normalizer Customization

### Adding Team Aliases

```python
# utils/normalizers.py - Update TEAM_ALIASES dict
TEAM_ALIASES = {
    # ... existing entries ...
    "new_team": "Standard Team Name",
}
```

### Adding Agents

```python
# utils/normalizers.py - Update AGENTS set
AGENTS = {
    # ... existing agents ...
    "new_agent",
}
```

## Performance Optimization

### 1. Batch Processing

```python
# Process large datasets in chunks
CHUNK_SIZE = 1000
for i in range(0, len(raw_data), CHUNK_SIZE):
    chunk = raw_data[i:i+CHUNK_SIZE]
    processed = DataPipeline.process_matches(chunk)
    CSVManager.append_csv(processed, output_file)
```

### 2. Parallel Scraping

```python
# Run multiple scrapers concurrently (already implemented)
result = await pipeline.run(sequential=False)
```

### 3. Parquet Format

```python
# For large datasets, use Parquet
pipeline = ValolyzerPipeline(scrapers, output_format="parquet")
```

### 4. Column Selection

```python
# Read only needed columns
df = CSVManager.read_csv("data/matches.csv", engine="polars")
df = df.select(["match_id", "team_a", "team_b", "winner"])
```

## Debugging Tips

### Enable Debug Logging

```python
from utils.logging import Logger

Logger.configure(level="DEBUG")
```

### Test HTTP Client

```python
import asyncio
from utils.http import AsyncHTTPClient

async def test():
    async with AsyncHTTPClient() as client:
        html = await client.get("https://www.vlr.gg")
        print(f"Got {len(html)} bytes")

asyncio.run(test())
```

### Check Parser Output

```python
from parsers.data_parser import MatchParser

raw = {"match_id": "1", "team_a": "prx", "team_b": "geng"}
parsed = MatchParser.parse(raw)
print(f"Valid: {MatchParser.validate(parsed)}")
print(f"Parsed: {parsed}")
```

### Monitor Data Quality

```python
from utils.csv_handler import CSVManager

stats = CSVManager.get_stats("data/processed/matches.csv")
print(f"Rows: {stats['rows']}")
print(f"Columns: {stats['columns']}")
print(f"Memory: {stats['memory_usage_mb']:.2f} MB")
```

## Testing

### Unit Tests

```python
# test_normalizers.py
def test_team_normalization():
    from utils.normalizers import Normalizers

    assert Normalizers.normalize_team_name("prx") == "Paper Rex"
    assert Normalizers.normalize_team_name("geng") == "Gen.G"

def test_agent_normalization():
    from utils.normalizers import Normalizers

    assert Normalizers.normalize_agent_name("jett") == "Jett"
    assert Normalizers.normalize_agent_name("kay/o") == "KAY/O"
```

### Integration Tests

```python
# test_pipeline.py
import asyncio
from scrapers.vlr import VLRScraper
from pipelines import ValolyzerPipeline

async def test_pipeline():
    scraper = VLRScraper(rate_limit=0.5)
    pipeline = ValolyzerPipeline([scraper])
    result = await pipeline.run()

    assert result["status"] == "success"
    assert result["export_stats"]["matches.csv"] > 0
```

## Common Issues & Solutions

### Issue: Rate Limited (429 errors)
**Solution:** Reduce `rate_limit` parameter or use `--sequential` mode
```python
scraper = VLRScraper(rate_limit=0.5)  # 1 request every 2 seconds
```

### Issue: Memory Usage High
**Solution:** Use Parquet format and process in batches
```python
pipeline = ValolyzerPipeline(scrapers, output_format="parquet")
```

### Issue: Scraper Returns No Data
**Solution:** Check logs, verify site structure hasn't changed
```bash
# Review scraper logs
tail -f logs/scraper_*.log
```

### Issue: Duplicate Data
**Solution:** Ensure deduplication is enabled
```python
CSVManager.append_csv(df, "data.csv", deduplicate_on=["match_id"])
```

## Deployment

### Scheduled Execution (Cron)

```bash
# Add to crontab
0 */6 * * * cd /path/to/valolyzer && python main.py full

# Or with nice/ionice for background priority
0 */6 * * * cd /path/to/valolyzer && nice -n 19 ionice -c3 python main.py full
```

### Docker Deployment

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main.py", "full"]
```

### Cloud Deployment

Store data in cloud storage (S3, GCS):
```python
# Future: Add cloud storage support to CSVManager
```

## Contributing

1. Create feature branch: `git checkout -b feature/new-scraper`
2. Make changes following project structure
3. Test thoroughly: `python -m pytest`
4. Submit PR with description

## Resources

- **VLR.gg**: https://www.vlr.gg
- **Valorant API Docs**: https://developer.riotgames.com
- **Async Python**: https://docs.python.org/3/library/asyncio.html
- **Pandas Docs**: https://pandas.pydata.org/docs/
- **Polars Docs**: https://docs.pola-rs.com/

## License

MIT - See LICENSE file
