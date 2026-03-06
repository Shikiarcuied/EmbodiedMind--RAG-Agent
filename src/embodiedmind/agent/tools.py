"""
LangChain tools for the EmbodiedMind agent.
- KnowledgeBaseTool: search the local vector store
- WebSearchTool: Tavily search for up-to-date information
- ArxivTool: search arXiv papers on embodied AI topics
"""

import logging
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def make_knowledge_base_tool(vector_store):
    """Create a tool that searches the local ChromaDB knowledge base."""

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the EmbodiedMind knowledge base for information about embodied AI, \
robotics, LeRobot, diffusion policy, and related topics. Returns relevant passages with sources."""
        docs = vector_store.similarity_search(query, k=5)
        if not docs:
            return "No relevant documents found in the knowledge base."

        parts = []
        for doc in docs:
            url = doc.metadata.get("source_url", "unknown")
            title = doc.metadata.get("title", "")
            parts.append(f"[Source: {url}]\n{doc.page_content[:800]}")
        return "\n\n---\n\n".join(parts)

    return search_knowledge_base


def make_web_search_tool(api_key: Optional[str] = None):
    """Create a Tavily web search tool for up-to-date information."""
    from langchain_community.tools.tavily_search import TavilySearchResults
    from embodiedmind.config import settings

    key = api_key or settings.tavily_api_key
    if not key:
        logger.warning("TAVILY_API_KEY not set — web search tool disabled")
        return None

    return TavilySearchResults(
        api_key=key,
        max_results=5,
        description=(
            "Search the web for up-to-date information about embodied AI, robotics research, "
            "and related topics not covered in the local knowledge base."
        ),
    )


def make_arxiv_tool():
    """Create an arXiv search tool for finding academic papers."""
    from langchain_community.tools.arxiv.tool import ArxivQueryRun
    from langchain_community.utilities.arxiv import ArxivAPIWrapper

    wrapper = ArxivAPIWrapper(
        top_k_results=3,
        doc_content_chars_max=2000,
    )
    return ArxivQueryRun(
        api_wrapper=wrapper,
        description=(
            "Search arXiv for academic papers on embodied AI, robot learning, "
            "diffusion policy, imitation learning, and related research topics."
        ),
    )


def build_tools(vector_store) -> list:
    tools = [make_knowledge_base_tool(vector_store)]

    web_tool = make_web_search_tool()
    if web_tool:
        tools.append(web_tool)

    tools.append(make_arxiv_tool())

    return tools
