from pydantic import BaseModel
from typing import List, Optional
from pydantic import Field


class QuestionRequest(BaseModel):
    question: str


class DocumentResult(BaseModel):
    title: str
    score: float
    url: Optional[str] = None
    id: Optional[str] = None
    rank: int


class SearchResponse(BaseModel):
    query: str
    total_results: int
    documents: List[DocumentResult]
    search_time_ms: float
    timestamp: str


class SearchRequest(BaseModel):
    query: str = Field(
        ..., description="Search query text", min_length=1, max_length=500
    )
    top_k: int = Field(
        default=3, description="Number of top results to return", ge=1, le=10
    )
