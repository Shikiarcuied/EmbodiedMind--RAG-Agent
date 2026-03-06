"""
EmbodiedMind Agent executor.
Uses LangChain's ReAct agent with the knowledge base, web search, and arXiv tools.
"""

import logging
from typing import Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from embodiedmind.agent.tools import build_tools
from embodiedmind.chains.citation_chain import CitationChain
from embodiedmind.chains.retrieval_qa import build_retrieval_qa_chain
from embodiedmind.config import settings

logger = logging.getLogger(__name__)

_REACT_TEMPLATE = """You are EmbodiedMind, an expert assistant in embodied AI and robotics.
You have access to tools to search a curated knowledge base, the web, and arXiv papers.
Always cite sources in your final answer by including the source URL.
Respond in the same language as the user's question.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question, with source citations

Begin!

Question: {input}
Thought: {agent_scratchpad}"""


class EmbodiedMindAgent:
    def __init__(self, vector_store):
        self._vector_store = vector_store
        self._tools = build_tools(vector_store)
        self._llm = ChatOpenAI(
            model=settings.openai_chat_model,
            openai_api_key=settings.openai_api_key,
            temperature=0.1,
        )

        prompt = PromptTemplate.from_template(_REACT_TEMPLATE)
        react_agent = create_react_agent(self._llm, self._tools, prompt)

        self._executor = AgentExecutor(
            agent=react_agent,
            tools=self._tools,
            verbose=True,
            max_iterations=6,
            handle_parsing_errors=True,
        )

        # Also expose a simpler direct RAG chain for fast queries
        retriever = vector_store.as_retriever(k=5)
        qa_chain = build_retrieval_qa_chain(retriever)
        self._citation_chain = CitationChain(retriever, qa_chain)

    def ask(self, question: str, use_agent: bool = True) -> str:
        """
        Answer a question.
        use_agent=True: full ReAct agent (uses all tools, slower)
        use_agent=False: direct RAG with citations (faster)
        """
        if use_agent:
            try:
                result = self._executor.invoke({"input": question})
                answer = result.get("output", "")
                return answer
            except Exception as exc:
                logger.warning("Agent failed: %s — falling back to RAG chain", exc)

        return self._citation_chain.format_response(question)

    def ask_with_citations(self, question: str):
        """Return AnswerWithCitations for structured access to sources."""
        return self._citation_chain.invoke(question)
