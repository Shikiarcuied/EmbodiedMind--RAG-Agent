"""Tests for ingestion modules."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.documents import Document


class TestChunker:
    def test_chunk_documents_splits_long_text(self):
        from embodiedmind.ingestion.chunker import chunk_documents

        long_text = "This is a sentence about embodied AI. " * 100
        doc = Document(
            page_content=long_text,
            metadata={
                "source_url": "https://example.com",
                "license": "MIT",
                "crawl_date": "2025-01-01T00:00:00+00:00",
                "content_hash": "abc123",
                "title": "test.txt",
            },
        )
        chunks = chunk_documents([doc])
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.metadata.get("source_url") == "https://example.com"

    def test_chunk_documents_preserves_metadata(self, sample_documents):
        from embodiedmind.ingestion.chunker import chunk_documents

        chunks = chunk_documents(sample_documents)
        for chunk in chunks:
            assert "source_url" in chunk.metadata
            assert "license" in chunk.metadata

    def test_chunk_empty_list_returns_empty(self):
        from embodiedmind.ingestion.chunker import chunk_documents

        assert chunk_documents([]) == []

    def test_chunk_markdown_document(self):
        from embodiedmind.ingestion.chunker import chunk_documents

        md_content = (
            "# Chapter 1\n\nIntroduction to embodied AI.\n\n"
            "## Section 1.1\n\nDetails about reinforcement learning.\n\n"
            "## Section 1.2\n\nDetails about imitation learning.\n"
        )
        doc = Document(
            page_content=md_content,
            metadata={
                "source_url": "https://example.com/guide.md",
                "license": "MIT",
                "crawl_date": "2025-01-01T00:00:00+00:00",
                "content_hash": "md123",
                "title": "guide.md",
            },
        )
        chunks = chunk_documents([doc])
        assert len(chunks) >= 1


class TestWebLoader:
    @pytest.mark.asyncio
    async def test_load_web_page_skips_disallowed_url(self):
        from embodiedmind.ingestion.loaders import load_web_page
        from embodiedmind.compliance.robots_checker import RobotsChecker

        checker = MagicMock(spec=RobotsChecker)
        checker.is_allowed.return_value = False

        result = await load_web_page(
            "https://example.com/private",
            checker=checker,
        )
        assert result is None
        checker.is_allowed.assert_called_once_with("https://example.com/private")

    @pytest.mark.asyncio
    async def test_load_web_page_returns_document_on_success(self):
        from embodiedmind.ingestion.loaders import load_web_page
        from embodiedmind.compliance.robots_checker import RobotsChecker

        checker = MagicMock(spec=RobotsChecker)
        checker.is_allowed.return_value = True

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = (
            "<html><head><title>Test Page</title></head>"
            "<body><main><p>Embodied AI content here.</p></main></body></html>"
        )

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            mock_client.get.return_value = mock_response

            result = await load_web_page(
                "https://example.com/docs",
                checker=checker,
                source_name="test",
            )

        assert result is not None
        assert isinstance(result, Document)
        assert "source_url" in result.metadata
        assert result.metadata["source_url"] == "https://example.com/docs"


class TestVectorStoreSchema:
    def test_validate_metadata_catches_missing_fields(self):
        from embodiedmind.vectorstore.schema import validate_metadata

        incomplete = {"source_url": "https://example.com"}
        missing = validate_metadata(incomplete)
        assert "license" in missing
        assert "crawl_date" in missing
        assert "content_hash" in missing

    def test_validate_metadata_passes_complete_doc(self):
        from embodiedmind.vectorstore.schema import validate_metadata

        complete = {
            "source_url": "https://example.com",
            "license": "MIT",
            "crawl_date": "2025-01-01T00:00:00+00:00",
            "content_hash": "abc123",
        }
        missing = validate_metadata(complete)
        assert missing == []
