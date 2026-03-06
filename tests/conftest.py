"""Shared pytest fixtures."""

import os
import sys
import pytest

# Ensure src is importable without poetry install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Set dummy env vars so settings can be loaded without a real .env
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("GITHUB_TOKEN", "ghp_test")
os.environ.setdefault("BOT_CONTACT_EMAIL", "test@example.com")
os.environ.setdefault("CHROMA_PERSIST_DIR", "/tmp/test_chroma")
os.environ.setdefault("CHROMA_COLLECTION_NAME", "test_collection")


@pytest.fixture
def sample_document():
    from langchain_core.documents import Document

    return Document(
        page_content="Diffusion Policy is a method for robot learning.",
        metadata={
            "source_url": "https://github.com/example/repo/blob/main/README.md",
            "license": "MIT",
            "crawl_date": "2025-01-01T00:00:00+00:00",
            "content_hash": "abc123def456",
            "source_name": "test_source",
            "title": "README.md",
        },
    )


@pytest.fixture
def sample_documents(sample_document):
    from langchain_core.documents import Document

    return [
        sample_document,
        Document(
            page_content="LeRobot is a HuggingFace library for robot learning.",
            metadata={
                "source_url": "https://huggingface.co/docs/lerobot/index",
                "license": "Apache-2.0",
                "crawl_date": "2025-01-01T00:00:00+00:00",
                "content_hash": "def456abc789",
                "source_name": "lerobot_docs",
                "title": "LeRobot Docs",
            },
        ),
    ]
