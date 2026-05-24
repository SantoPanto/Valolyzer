"""
HTTP utilities for async web scraping.
Includes retry logic, rate limiting, rotating user agents, and session management.
"""

import aiohttp
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import random
from utils.logging import get_logger

logger = get_logger(__name__)


class UserAgent:
    """Rotating user agents to avoid blocking."""

    AGENTS = [
        # Chrome
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        # Firefox
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
        # Safari
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
        # Edge
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    ]

    @classmethod
    def get_random(cls) -> str:
        """Get a random user agent."""
        return random.choice(cls.AGENTS)


class RateLimiter:
    """Rate limiter for async requests."""

    def __init__(self, requests_per_second: float = 2.0, per_domain: bool = True):
        """
        Initialize rate limiter.

        Args:
            requests_per_second: Number of requests allowed per second
            per_domain: Whether to apply rate limit per domain
        """
        self.requests_per_second = requests_per_second
        self.per_domain = per_domain
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time: Dict[str, datetime] = {}

    async def acquire(self, domain: str = "global"):
        """
        Acquire permission to make a request.

        Args:
            domain: Domain to rate limit on
        """
        while True:
            now = datetime.now()
            last_time = self.last_request_time.get(domain)

            if last_time is None:
                self.last_request_time[domain] = now
                return

            elapsed = (now - last_time).total_seconds()
            if elapsed >= self.min_interval:
                self.last_request_time[domain] = now
                return

            wait_time = self.min_interval - elapsed
            await asyncio.sleep(wait_time)


class RetryStrategy:
    """Retry strategy with exponential backoff."""

    def __init__(self, 
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0):
        """
        Initialize retry strategy.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay in seconds
            max_delay: Maximum delay cap
            exponential_base: Exponential backoff multiplier
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

    def get_delay(self, attempt: int) -> float:
        """
        Get delay for retry attempt.

        Args:
            attempt: Attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        delay = self.base_delay * (self.exponential_base ** attempt)
        jitter = delay * 0.1 * random.random()
        return min(delay + jitter, self.max_delay)


class AsyncHTTPClient:
    """Async HTTP client with retry logic and rate limiting."""

    def __init__(self,
                 rate_limit: float = 2.0,
                 timeout: int = 30,
                 retry_strategy: Optional[RetryStrategy] = None,
                 headers: Optional[Dict[str, str]] = None):
        """
        Initialize async HTTP client.

        Args:
            rate_limit: Requests per second
            timeout: Request timeout in seconds
            retry_strategy: Retry strategy instance
            headers: Additional headers for all requests
        """
        self.rate_limiter = RateLimiter(rate_limit)
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.retry_strategy = retry_strategy or RetryStrategy()
        self.headers = headers or {}
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def connect(self):
        """Create aiohttp session."""
        connector = aiohttp.TCPConnector(limit_per_host=5)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=self.timeout
        )
        logger.info("HTTP client connected")

    async def close(self):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
            logger.info("HTTP client closed")

    async def get(self, url: str, **kwargs) -> Optional[str]:
        """
        Make GET request with retry logic.

        Args:
            url: URL to fetch
            **kwargs: Additional arguments for session.get()

        Returns:
            Response text or None on failure
        """
        return await self._request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> Optional[str]:
        """Make POST request with retry logic."""
        return await self._request("POST", url, **kwargs)

    async def _request(self, method: str, url: str, **kwargs) -> Optional[str]:
        """
        Internal request method with retry logic.

        Args:
            method: HTTP method
            url: URL to request
            **kwargs: Additional arguments

        Returns:
            Response text or None
        """
        if not self.session:
            raise RuntimeError("HTTP client not connected. Use 'connect()' first.")

        # Extract domain for rate limiting
        from urllib.parse import urlparse
        domain = urlparse(url).netloc

        # Prepare headers
        headers = {**self.headers, "User-Agent": UserAgent.get_random()}
        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))

        for attempt in range(self.retry_strategy.max_retries + 1):
            try:
                # Rate limiting
                await self.rate_limiter.acquire(domain)

                # Make request
                async with self.session.request(
                    method,
                    url,
                    headers=headers,
                    **kwargs
                ) as response:
                    if response.status == 200:
                        text = await response.text()
                        logger.debug(f"{method} {url} -> {response.status}")
                        return text
                    elif response.status == 429:
                        # Rate limited
                        logger.warning(f"Rate limited (429) on {url}")
                        delay = self.retry_strategy.get_delay(attempt)
                        await asyncio.sleep(delay)
                    elif response.status >= 500:
                        # Server error - retry
                        logger.warning(f"{method} {url} -> {response.status} (retrying)")
                        if attempt < self.retry_strategy.max_retries:
                            delay = self.retry_strategy.get_delay(attempt)
                            await asyncio.sleep(delay)
                    else:
                        # Client error - don't retry
                        logger.error(f"{method} {url} -> {response.status}")
                        return None

            except asyncio.TimeoutError:
                logger.warning(f"Timeout on {url} (attempt {attempt + 1})")
                if attempt < self.retry_strategy.max_retries:
                    delay = self.retry_strategy.get_delay(attempt)
                    await asyncio.sleep(delay)
            except Exception as e:
                logger.error(f"Error requesting {url}: {e}")
                if attempt < self.retry_strategy.max_retries:
                    delay = self.retry_strategy.get_delay(attempt)
                    await asyncio.sleep(delay)

        logger.error(f"Failed to fetch {url} after {self.retry_strategy.max_retries + 1} attempts")
        return None

    async def get_json(self, url: str, **kwargs) -> Optional[Dict]:
        """Get and parse JSON response."""
        import json
        response = await self.get(url, **kwargs)
        if response:
            try:
                return json.loads(response)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from {url}: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    async def main():
        async with AsyncHTTPClient(rate_limit=1.0) as client:
            # Test request
            html = await client.get("https://www.google.com")
            if html:
                print(f"Fetched {len(html)} characters")

    # asyncio.run(main())
