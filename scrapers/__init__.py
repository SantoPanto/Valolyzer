"""Scrapers module."""

from scrapers.base import BaseScraper, ScraperPipeline
from scrapers.vlr.vlr_scraper import VLRScraper

__all__ = ["BaseScraper", "ScraperPipeline", "VLRScraper"]
