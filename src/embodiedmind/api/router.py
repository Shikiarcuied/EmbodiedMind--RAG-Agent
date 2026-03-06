"""
FastAPI router exposing EmbodiedMind as a REST API.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from embodiedmind.agent.executor import EmbodiedMindAgent
from embodiedmind.vectorstore import get_vector_store

router = APIRouter(prefix="/api/v1", tags=["embodiedmind"])

# Lazy-initialized agent
_agent: EmbodiedMindAgent | None = None


def get_agent() -> EmbodiedMindAgent:
    global _agent
    if _agent is None:
        vs = get_vector_store()
        _agent = EmbodiedMindAgent(vs)
    return _agent


class QueryRequest(BaseModel):
    question: str
    use_agent: bool = False  # default to fast RAG; set True for full ReAct agent


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[dict]


class StatsResponse(BaseModel):
    total_chunks: int
    by_source: dict[str, int]


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    agent = get_agent()
    result = agent.ask_with_citations(request.question)
    return QueryResponse(
        question=request.question,
        answer=result.answer,
        citations=result.citations,
    )


@router.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    vs = get_vector_store()
    data = vs.collection_stats()
    return StatsResponse(
        total_chunks=data.get("total_chunks", 0),
        by_source=data.get("by_source", {}),
    )


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "EmbodiedMind"}
