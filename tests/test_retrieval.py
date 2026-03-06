"""Tests for retrieval chains."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


class TestCitationChain:
    def _make_docs(self):
        return [
            Document(
                page_content="Diffusion Policy enables robot learning.",
                metadata={
                    "source_url": "https://github.com/example/repo/blob/main/policy.md",
                    "license": "MIT",
                    "crawl_date": "2025-01-01T00:00:00+00:00",
                    "content_hash": "abc",
                    "source_name": "test",
                    "title": "policy.md",
                },
            )
        ]

    def test_citation_chain_formats_answer_with_sources(self):
        from embodiedmind.chains.citation_chain import CitationChain

        retriever = MagicMock()
        retriever.invoke.return_value = self._make_docs()

        qa_chain = MagicMock()
        qa_chain.invoke.return_value = "Diffusion Policy is a powerful approach."

        chain = CitationChain(retriever, qa_chain)
        result = chain.invoke("What is Diffusion Policy?")

        assert result.answer == "Diffusion Policy is a powerful approach."
        assert len(result.citations) == 1
        assert "github.com" in result.citations[0]["source_url"]

    def test_format_includes_source_url(self):
        from embodiedmind.chains.citation_chain import CitationChain

        retriever = MagicMock()
        retriever.invoke.return_value = self._make_docs()

        qa_chain = MagicMock()
        qa_chain.invoke.return_value = "Answer text."

        chain = CitationChain(retriever, qa_chain)
        formatted = chain.format_response("test question")

        assert "https://github.com/example/repo/blob/main/policy.md" in formatted

    def test_no_duplicate_citations(self):
        from embodiedmind.chains.citation_chain import CitationChain, AnswerWithCitations

        # Two docs with the same URL
        docs = self._make_docs() * 2
        result = AnswerWithCitations(answer="Test answer", citations=[d.metadata for d in docs])
        formatted = result.format()

        # URL should appear only once in the sources section
        sources_section = formatted.split("**Sources:**")[-1]
        url = "https://github.com/example/repo/blob/main/policy.md"
        assert sources_section.count(url) == 1


class TestFormatContext:
    def test_format_context_includes_url(self):
        from embodiedmind.chains.retrieval_qa import _format_context

        docs = [
            Document(
                page_content="Robot arm control.",
                metadata={
                    "source_url": "https://example.com/robot",
                    "title": "robot.md",
                    "source_name": "test",
                },
            )
        ]
        context = _format_context(docs)
        assert "https://example.com/robot" in context
        assert "Robot arm control." in context
