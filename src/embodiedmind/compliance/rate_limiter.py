"""
Async token-bucket rate limiter.
Defaults:
  - Xbotics / HuggingFace web pages: 1 req/s (configurable via .env)
  - GitHub API: strictly <= 4500 req/hr; checks X-RateLimit-Remaining header
  - 429 / Retry-After: automatic back-off, no forced retries
All HTTP requests must go through this limiter.
"""

import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token-bucket rate limiter for async contexts."""

    def __init__(self, requests_per_second: float = 1.0, burst: int = 1):
        self._rate = requests_per_second
        self._burst = burst
        self._tokens: float = burst
        self._last_refill: float = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
            self._last_refill = now

            if self._tokens < 1.0:
                wait_time = (1.0 - self._tokens) / self._rate
                logger.debug("Rate limit: waiting %.2fs before next request", wait_time)
                await asyncio.sleep(wait_time)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


class GitHubRateLimiter:
    """
    Rate limiter for GitHub API that:
    - Enforces <= 4500 req/hr via token bucket
    - Checks X-RateLimit-Remaining response headers
    - Backs off automatically on 429 or when remaining < 100
    """

    MAX_PER_HOUR = 4500
    _REFILL_INTERVAL = 3600.0  # seconds in one hour

    def __init__(self, max_per_hour: int = MAX_PER_HOUR):
        self._max = max_per_hour
        self._tokens: float = max_per_hour
        self._last_refill: float = time.monotonic()
        self._lock = asyncio.Lock()
        self._reset_at: Optional[float] = None  # epoch time from header

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            refill = elapsed * (self._max / self._REFILL_INTERVAL)
            self._tokens = min(float(self._max), self._tokens + refill)
            self._last_refill = now

            if self._tokens < 1.0:
                wait_time = (1.0 - self._tokens) / (self._max / self._REFILL_INTERVAL)
                logger.warning(
                    "GitHub API token bucket exhausted — waiting %.1fs", wait_time
                )
                await asyncio.sleep(wait_time)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0

    async def handle_response_headers(self, headers: dict) -> None:
        """Call this after every GitHub API response to update state."""
        remaining = headers.get("X-RateLimit-Remaining")
        reset = headers.get("X-RateLimit-Reset")

        if remaining is not None:
            remaining_int = int(remaining)
            logger.debug("GitHub API rate limit remaining: %d", remaining_int)

            if remaining_int < 100:
                if reset is not None:
                    import time as _time

                    reset_epoch = int(reset)
                    wait = reset_epoch - _time.time()
                    if wait > 0:
                        logger.warning(
                            "GitHub API remaining=%d < 100 — waiting %ds for reset",
                            remaining_int,
                            int(wait),
                        )
                        await asyncio.sleep(wait)

    async def handle_429(self, retry_after: Optional[str] = None) -> None:
        """Back off on 429 Too Many Requests."""
        if retry_after:
            try:
                wait = float(retry_after)
            except ValueError:
                wait = 60.0
        else:
            wait = 60.0
        logger.warning("Received 429 — backing off for %.1fs", wait)
        await asyncio.sleep(wait)


# Module-level singletons — import and use these in loaders
from embodiedmind.config import settings

web_limiter = RateLimiter(
    requests_per_second=1.0 / settings.crawl_delay_seconds,
    burst=1,
)

github_limiter = GitHubRateLimiter(max_per_hour=settings.github_api_max_per_hour)
