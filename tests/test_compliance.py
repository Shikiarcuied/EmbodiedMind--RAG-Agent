"""Tests for compliance modules."""

import asyncio
import pytest
from unittest.mock import MagicMock, patch


class TestRobotsChecker:
    def test_is_allowed_with_permissive_robots(self):
        from embodiedmind.compliance.robots_checker import RobotsChecker

        checker = RobotsChecker(user_agent="TestBot/1.0")

        # Mock httpx.get to return permissive robots.txt
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "User-agent: *\nAllow: /\n"

        with patch("httpx.get", return_value=mock_resp):
            allowed = checker.is_allowed("https://example.com/docs/page")
        assert allowed is True

    def test_is_allowed_with_restrictive_robots(self):
        from embodiedmind.compliance.robots_checker import RobotsChecker

        checker = RobotsChecker(user_agent="TestBot/1.0")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "User-agent: *\nDisallow: /\n"

        with patch("httpx.get", return_value=mock_resp):
            allowed = checker.is_allowed("https://example.com/anything")
        assert allowed is False

    def test_is_allowed_returns_false_on_fetch_error(self):
        from embodiedmind.compliance.robots_checker import RobotsChecker

        checker = RobotsChecker(user_agent="TestBot/1.0")

        with patch("httpx.get", side_effect=Exception("Connection refused")):
            allowed = checker.is_allowed("https://unreachable.example.com/page")
        assert allowed is False

    def test_crawl_delay_returns_default_when_not_specified(self):
        from embodiedmind.compliance.robots_checker import RobotsChecker

        checker = RobotsChecker(user_agent="TestBot/1.0")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "User-agent: *\nAllow: /\n"

        with patch("httpx.get", return_value=mock_resp):
            delay = checker.get_crawl_delay("https://example.com")
        assert delay >= 1.0  # default from settings

    def test_no_robots_txt_allows_all(self):
        from embodiedmind.compliance.robots_checker import RobotsChecker

        checker = RobotsChecker(user_agent="TestBot/1.0")

        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch("httpx.get", return_value=mock_resp):
            allowed = checker.is_allowed("https://example.com/any/path")
        assert allowed is True


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_acquire_does_not_raise(self):
        from embodiedmind.compliance.rate_limiter import RateLimiter

        limiter = RateLimiter(requests_per_second=100.0, burst=10)
        # Should complete without error
        await limiter.acquire()

    @pytest.mark.asyncio
    async def test_github_limiter_acquire(self):
        from embodiedmind.compliance.rate_limiter import GitHubRateLimiter

        limiter = GitHubRateLimiter(max_per_hour=4500)
        await limiter.acquire()

    @pytest.mark.asyncio
    async def test_handle_429_backs_off(self):
        from embodiedmind.compliance.rate_limiter import GitHubRateLimiter

        limiter = GitHubRateLimiter(max_per_hour=4500)
        # Should not raise, should sleep
        with patch("asyncio.sleep") as mock_sleep:
            await limiter.handle_429(retry_after="2")
            mock_sleep.assert_called_once_with(2.0)


class TestAttribution:
    def test_build_metadata_creates_valid_object(self):
        from embodiedmind.compliance.attribution import build_metadata

        meta = build_metadata(
            content="Hello world",
            source_url="https://example.com/page",
            license="MIT",
            source_name="test_source",
        )
        assert meta.source_url == "https://example.com/page"
        assert meta.license == "MIT"
        assert len(meta.content_hash) == 64  # SHA-256 hex
        assert meta.crawl_date  # non-empty ISO date

    def test_compute_hash_is_deterministic(self):
        from embodiedmind.compliance.attribution import compute_hash

        h1 = compute_hash("same content")
        h2 = compute_hash("same content")
        assert h1 == h2

    def test_different_content_different_hash(self):
        from embodiedmind.compliance.attribution import compute_hash

        h1 = compute_hash("content A")
        h2 = compute_hash("content B")
        assert h1 != h2

    def test_format_citation_includes_url(self):
        from embodiedmind.compliance.attribution import format_citation

        meta = {
            "source_url": "https://github.com/example/repo/blob/main/README.md",
            "title": "README.md",
            "source_name": "test_source",
        }
        citation = format_citation(meta)
        assert "https://github.com/example/repo/blob/main/README.md" in citation
