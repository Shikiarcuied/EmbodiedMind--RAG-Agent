"""Tests for the agent module."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


def _make_vector_store():
    """Create a mock vector store."""
    vs = MagicMock()
    vs.similarity_search.return_value = [
        Document(
            page_content="Embodied AI involves robots learning from interaction.",
            metadata={
                "source_url": "https://github.com/TianxingChen/Embodied-AI-Guide/blob/main/README.md",
                "license": "unknown",
                "crawl_date": "2025-01-01T00:00:00+00:00",
                "content_hash": "abc123",
                "source_name": "lumina_embodied_ai_guide",
                "title": "README.md",
            },
        )
    ]
    vs.as_retriever.return_value = MagicMock(
        invoke=lambda q: vs.similarity_search(q)
    )
    return vs


class TestEmbodiedMindAgent:
    @patch("embodiedmind.agent.executor.ChatOpenAI")
    @patch("embodiedmind.agent.tools.make_web_search_tool", return_value=None)
    def test_ask_with_citations_returns_answer(self, _mock_web, mock_llm_cls):
        from embodiedmind.agent.executor import EmbodiedMindAgent

        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        # Mock the chain invocation
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Embodied AI is about robots in the physical world."

        vs = _make_vector_store()
        agent = EmbodiedMindAgent(vs)

        # Patch the citation chain's qa_chain
        agent._citation_chain._qa_chain = mock_chain

        result = agent.ask_with_citations("What is embodied AI?")
        assert result.answer == "Embodied AI is about robots in the physical world."
        assert len(result.citations) >= 1

    @patch("embodiedmind.agent.executor.ChatOpenAI")
    @patch("embodiedmind.agent.tools.make_web_search_tool", return_value=None)
    def test_ask_without_agent_uses_rag(self, _mock_web, mock_llm_cls):
        from embodiedmind.agent.executor import EmbodiedMindAgent

        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Direct RAG answer."

        vs = _make_vector_store()
        agent = EmbodiedMindAgent(vs)
        agent._citation_chain._qa_chain = mock_chain

        answer = agent.ask("What is LeRobot?", use_agent=False)
        assert "Direct RAG answer." in answer
