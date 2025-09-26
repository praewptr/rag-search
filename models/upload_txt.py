from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentItem(BaseModel):
    """Pydantic model for individual document items."""

    content: str = Field(..., description="The text content to be processed")
    source: str = Field(..., description="Source identifier for the document")
    timestamp: Optional[str] = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="ISO timestamp",
    )
    id: Optional[int] = Field(None, description="Document ID from database")
    # Add support for Oracle fields
    ROWID: Optional[str] = Field(None, description="Oracle ROWID")

    class Config:
        # Allow extra fields for Oracle columns
        extra = "allow"


class BulkUploadRequest(BaseModel):
    """Pydantic model for bulk upload requests."""

    value: List[DocumentItem] = Field(..., description="List of documents to upload")


class UploadResponse(BaseModel):
    """Pydantic model for upload responses."""

    document_count: int
    chunk_count: int
    message: str
    uploaded_documents: List[Dict[str, Any]]


class DocumentResponse(BaseModel):
    value: List[Dict[str, Any]]  # Changed to support Oracle data structure


class UploadRequest(BaseModel):
    value: List[DocumentItem]


class DocumentsPayload(BaseModel):
    value: List[DocumentItem]
    index: Optional[str] = None  # Optional index name for upload


class SearchResult(BaseModel):
    """Pydantic model for search results."""

    id: str
    content: str
    source: str
    timestamp: str
    chunk_id: Optional[str] = None
    score: Optional[float] = None
    title: Optional[str] = None  # Add title field


class SearchResponse(BaseModel):
    """Pydantic model for search response."""

    total_count: int
    results: List[SearchResult]
    query: str
    status: str


class TitleListResponse(BaseModel):
    """Pydantic model for title list response."""

    total_count: int
    titles: List[str]
    query: str
    status: str


class IndexStats(BaseModel):
    """Pydantic model for index statistics."""

    total_documents: int
    index_name: str
    status: str
