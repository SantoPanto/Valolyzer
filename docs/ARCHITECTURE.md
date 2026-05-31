# Valolyzer Architecture & Data Flow

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         VALOLYZER PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      DATA SOURCES                                 │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │   │
│  │  │   VLR.gg     │  │  RIB.gg      │  │  Tracker.gg  │            │   │
│  │  │  (Active)    │  │  (Planned)   │  │  (Planned)   │            │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘            │   │
│  └────────┬─────────────────────────────────────────────────────────┘   │
│           │                                                              │
│           ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │               SCRAPER FRAMEWORK (base.py)                        │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │  • HTTP Client (aiohttp)                                │   │   │
│  │  │  • Rate Limiting (per-domain)                           │   │   │
│  │  │  • Retry Logic (exponential backoff)                    │   │   │
│  │  │  • State Tracking (checkpoints)                         │   │   │
│  │  │  • Error Handling & Logging                             │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │  VLR Scraper Implementation                              │   │   │
│  │  │  • parse_match_page()                                    │   │   │
│  │  │  • parse_map_page()                                      │   │   │
│  │  │  • parse_player_stats()                                  │   │   │
│  │  │  • parse_compositions()                                  │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  └────────┬─────────────────────────────────────────────────────────┘   │
│           │                                                              │
│  Raw Data │                                                              │
│  {matches, maps, compositions, player_stats, rounds}                    │
│           │                                                              │
│           ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │               DATA PIPELINE (main_pipeline.py)                    │   │
│  │  1. Aggregate (from all scrapers)                               │   │
│  │  2. Parse (through DataPipeline)                                │   │
│  │  3. Export (CSV or Parquet)                                     │   │
│  └────────┬─────────────────────────────────────────────────────────┘   │
│           │                                                              │
│  Parsed   │                                                              │
│  Data     │                                                              │
│           ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │          PARSERS & NORMALIZERS (parsers/, utils/)                │   │
│  │  ┌──────────────────┐  ┌──────────────────┐                      │   │
│  │  │   MatchParser    │  │   MapParser      │                      │   │
│  │  └──────────────────┘  └──────────────────┘                      │   │
│  │  ┌──────────────────┐  ┌──────────────────┐                      │   │
│  │  │ CompositionParser│  │PlayerStatsParser │                      │   │
│  │  └──────────────────┘  └──────────────────┘                      │   │
│  │  ┌────────────────────────────────────────┐                      │   │
│  │  │      Normalizers                       │                      │   │
│  │  │  • Team Names (400+ aliases)           │                      │   │
│  │  │  • Agent Names (25 agents)             │                      │   │
│  │  │  • Map Names (10 maps)                 │                      │   │
│  │  │  • Patch Versions                      │                      │   │
│  │  │  • Economic Classifications            │                      │   │
│  │  └────────────────────────────────────────┘                      │   │
│  └────────┬─────────────────────────────────────────────────────────┘   │
│           │                                                              │
│ Normalized│ Validated                                                    │
│ Data      │                                                              │
│           ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │           CSV/PARQUET EXPORT (csv_handler.py)                    │   │
│  │  ┌────────────────┐  ┌────────────────┐                          │   │
│  │  │  CSV Format    │  │ Parquet Format │                          │   │
│  │  │  • Human read  │  │ • Compressed   │                          │   │
│  │  │  • Append mode │  │ • Columnar     │                          │   │
│  │  │  • Dedup keys  │  │ • Analytics    │                          │   │
│  │  └────────────────┘  └────────────────┘                          │   │
│  │  • Automatic deduplication on keys                               │   │
│  │  • Configurable output directory                                 │   │
│  │  • Memory-efficient processing                                   │   │
│  └────────┬─────────────────────────────────────────────────────────┘   │
│           │                                                              │
│           ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    DATA FILES                                     │   │
│  │  data/processed/                                                 │   │
│  │  ├── matches.csv              (Match metadata)                   │   │
│  │  ├── maps.csv                 (Map statistics)                   │   │
│  │  ├── compositions.csv         (5-agent lineups)                  │   │
│  │  ├── player_stats.csv         (Individual performance)           │   │
│  │  └── rounds.csv               (Round-level events)               │   │
│  │                                                                   │   │
│  │  data/parquet/                (Compressed versions)              │   │
│  │  ├── matches.parquet                                            │   │
│  │  └── ... (same files)                                           │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
HTML/JSON Response
      │
      ▼
┌─────────────────────────┐
│  BeautifulSoup/Parse    │
│  Extract Raw Elements   │
└──────────┬──────────────┘
           │
   Raw Dict│ {
           │   "match_id": "542195",
           │   "team_a": "prx",
           │   "team_b": "geng",
           │   ...
           │ }
           │
           ▼
┌─────────────────────────┐
│  DataPipeline.parse()   │
│  • MatchParser          │
│  • Normalizers          │
│  • Validation           │
└──────────┬──────────────┘
           │
 Normalized│ Dict {
     Data  │   "match_id": "542195",
           │   "team_a": "Paper Rex",
           │   "team_b": "Gen.G",
           │   "patch": "11.05",
           │   ...
           │ }
           │
           ▼
┌─────────────────────────┐
│ CSVManager.append_csv() │
│ • Check duplicates      │
│ • Read existing data    │
│ • Combine & deduplicate │
│ • Write back            │
└──────────┬──────────────┘
           │
           ▼
    CSV File
    (Appended with
     deduplication)
```

## Module Dependencies

```
main.py
├── scrapers/
│   ├── base.py
│   │   ├── utils/http.py
│   │   │   └── utils/logging.py
│   │   └── utils/csv_handler.py
│   │       └── utils/logging.py
│   └── vlr/vlr_scraper.py
│       ├── utils/normalizers.py
│       └── utils/logging.py
│
├── pipelines/main_pipeline.py
│   ├── scrapers/base.py
│   ├── parsers/data_parser.py
│   │   ├── utils/normalizers.py
│   │   └── utils/logging.py
│   └── utils/csv_handler.py
│
├── parsers/data_parser.py
│   ├── utils/normalizers.py
│   └── utils/logging.py
│
├── config.py
│   └── (dataclasses)
│
└── utils/
    ├── logging.py
    ├── normalizers.py
    ├── models.py (Pydantic)
    ├── http.py
    └── csv_handler.py
```

## Async Task Graph

```
Pipeline.run()
  │
  ├─► ScraperPipeline.run_parallel()
  │    │
  │    ├─► VLRScraper.run()
  │    │    ├─► scrape_events()
  │    │    ├─► scrape_event_matches()
  │    │    └─► scrape_match_detail()
  │    │
  │    └─► [Future: RIBScraper.run(), TrackerScraper.run()]
  │
  ├─► _aggregate_data()
  │
  ├─► _parse_data()
  │    └─► DataPipeline.process_all()
  │         ├─► process_matches()
  │         ├─► process_maps()
  │         ├─► process_compositions()
  │         ├─► process_player_stats()
  │         └─► process_rounds()
  │
  └─► _export_data()
       ├─► _export_csv()
       │    └─► CSVManager.append_csv() × N
       └─► _export_parquet()
            └─► CSVManager.to_parquet() × N
```

## Rate Limiting Strategy

```
Client → AsyncHTTPClient
           ├── RateLimiter (per domain)
           │   └── Semaphore
           ├── RetryStrategy
           │   ├── 3 retries max
           │   ├── 1s base delay
           │   ├── 2x exponential backoff
           │   └── 60s max delay
           └── HTTP Session
               └── 5 connections per host max
```

## Error Handling Flow

```
Request
  │
  ├─► Success (200)
  │   └─► Return response
  │
  ├─► Rate Limited (429)
  │   └─► Sleep + Retry (exponential backoff)
  │
  ├─► Server Error (5xx)
  │   └─► Retry (exponential backoff)
  │
  ├─► Timeout
  │   └─► Retry (exponential backoff)
  │
  ├─► Client Error (4xx, except 429)
  │   └─► Return None (don't retry)
  │
  └─► Max Retries Exceeded
      └─► Log error + Return None
```

## Data Validation Pipeline

```
Raw Data
  │
  ├─► Parser (type conversion)
  │   └─► Normalizer (standardization)
  │       └─► Validator (field checking)
  │           └─► Deduplicate (key matching)
  │
  ├─ Validation Passed?
  │   │
  │   ├─ Yes: Include in export
  │   └─ No: Log warning + Skip
  │
  └─► Export to CSV/Parquet
```

## Storage Architecture

```
data/
├── raw/                          (Optional: Raw scraper output)
│   ├── vlr_matches_raw.json
│   ├── vlr_maps_raw.json
│   └── ...
│
├── processed/                    (Primary: Normalized CSVs)
│   ├── matches.csv               (500+ KB typical)
│   ├── maps.csv
│   ├── compositions.csv
│   ├── player_stats.csv
│   └── rounds.csv
│
└── parquet/                      (Alternative: Compressed format)
    ├── matches.parquet           (50-100 KB typical)
    ├── maps.parquet
    ├── compositions.parquet
    ├── player_stats.parquet
    └── rounds.parquet

logs/
├── valolyzer_YYYY-MM-DD.log      (Main logs)
└── scraper_YYYY-MM-DD.log        (Debug scraper logs)
```

## Performance Optimization Points

```
1. HTTP Layer
   • Connection pooling (5 per host)
   • Keep-alive headers
   • Compressed responses
   • Parallel requests (rate-limited)

2. Parsing Layer
   • Stream processing where possible
   • Lazy evaluation of data
   • Regex compilation caching

3. Storage Layer
   • Batch inserts (CSV append)
   • Parquet columnar compression
   • Indexed read operations
   • Memory-mapped files (for large datasets)

4. Application Layer
   • Async/await for concurrency
   • Semaphore-limited parallelism
   • Early exit on errors
   • Lazy resource initialization
```

## Extensibility Points

```
1. Add New Scraper
   └── Create class(BaseScraper)
       ├── implement scrape()
       ├── implement parse_match()
       └── implement parse_map()

2. Add New Parser
   └── Create Parser class
       ├── implement parse()
       └── implement validate()

3. Add New Normalizer
   └── Update Normalizers class
       ├── Add alias mappings
       └── Add validation rules

4. Add New Export Format
   └── Extend CSVManager
       ├── implement read_format()
       └── implement write_format()

5. Add Database Backend
   └── Create database module
       ├── Connection management
       ├── Schema creation
       └── ORM models
```

---

## Summary

Valolyzer follows a **modular, layered architecture** with clear separation of concerns:

1. **Data Sources** → Scrapers extract raw data
2. **Framework** → Base scraper provides common functionality
3. **Processing** → Parsers normalize data
4. **Storage** → CSVManager handles export
5. **Orchestration** → Pipeline ties everything together

This design enables:
- Easy addition of new sources
- Reusable components
- Type-safe operations
- Scalable processing
- Comprehensive logging
- Production-ready reliability
