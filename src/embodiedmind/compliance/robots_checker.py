"""
Fetch and parse robots.txt for target sites before ingestion begins.
Checks every URL with is_allowed() before crawling.
Uses stdlib urllib.robotparser.RobotFileParser.
User-Agent: EmbodiedMindBot/1.0 (research; contact: your@email.com)
"""

import logging
from functools import lru_cache
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from embodiedmind.config import settings

logger = logging.getLogger(__name__)


class RobotsChecker:
    def __init__(self, user_agent: str | None = None):
        self._user_agent = user_agent or settings.bot_user_agent
        # Cache parsers keyed by base URL (scheme + netloc)
        self._parsers: dict[str, RobotFileParser] = {}

    def _get_parser(self, base_url: str) -> RobotFileParser:
        if base_url in self._parsers:
            return self._parsers[base_url]

        robots_url = f"{base_url}/robots.txt"
        parser = RobotFileParser(robots_url)
        try:
            # Use httpx with our User-Agent so the fetch itself is compliant
            resp = httpx.get(
                robots_url,
                headers={"User-Agent": self._user_agent},
                follow_redirects=True,
                timeout=10.0,
            )
            if resp.status_code == 200:
                parser.parse(resp.text.splitlines())
                logger.info("Loaded robots.txt from %s", robots_url)
            elif resp.status_code == 404:
                # No robots.txt — allow everything by default
                parser.parse([])
                logger.info("No robots.txt at %s, treating as allow-all", robots_url)
            else:
                # Conservative fallback: disallow on unexpected errors
                parser.parse(["User-agent: *", "Disallow: /"])
                logger.warning(
                    "Unexpected HTTP %d fetching %s — defaulting to disallow-all",
                    resp.status_code,
                    robots_url,
                )
        except Exception as exc:
            parser.parse(["User-agent: *", "Disallow: /"])
            logger.warning(
                "Failed to fetch robots.txt from %s (%s) — defaulting to disallow-all",
                robots_url,
                exc,
            )

        self._parsers[base_url] = parser
        return parser

    @staticmethod
    def _base_url(url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def is_allowed(self, url: str) -> bool:
        base = self._base_url(url)
        parser = self._get_parser(base)
        allowed = parser.can_fetch(self._user_agent, url)
        if not allowed:
            logger.warning("robots.txt disallows crawling: %s — skipping", url)
        return allowed

    def get_crawl_delay(self, base_url: str) -> float:
        parser = self._get_parser(base_url)
        delay = parser.crawl_delay(self._user_agent)
        if delay is None:
            return settings.crawl_delay_seconds
        return float(delay)

    def summarize(self, base_url: str) -> str:
        """Return a human-readable summary of robots.txt rules for logging/reporting."""
        parser = self._get_parser(base_url)
        lines = []
        lines.append(f"robots.txt summary for {base_url}:")
        crawl_delay = parser.crawl_delay(self._user_agent)
        lines.append(f"  Crawl-delay: {crawl_delay or 'not specified'}")
        # Test a few common paths to show allow/disallow
        test_paths = ["/", "/docs", "/api", "/search", "/login"]
        for path in test_paths:
            test_url = base_url + path
            status = "ALLOW" if parser.can_fetch(self._user_agent, test_url) else "DISALLOW"
            lines.append(f"  {status}: {path}")
        return "\n".join(lines)
