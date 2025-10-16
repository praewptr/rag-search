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


class DocumentResponse(BaseModel):
    value: List[Dict[str, Any]]  # Changed to support Oracle data structure


# class UploadRequest(BaseModel):
#     value: List[DocumentItem]


class DocumentsPayload(BaseModel):
    value: List[DocumentItem] = Field(..., description="List of text items to upload")
    index: Optional[str] = Field(None, description="Index name for upload")


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


class UploadResponse(BaseModel):
    message: str
    container: str
    overwrite: bool


class ListContainersResponse(BaseModel):
    containers: list
    count: int
    status: str


class DeleteResponse(BaseModel):
    message: str
    container: str | None = None


class MarkUploaded(BaseModel):
    document_ids: List[int] = Field(
        ..., description="List of document IDs to mark as uploaded"
    )


class UploadTextResponse(BaseModel):
    message: str
    uploaded_count: int
    status: str
    failed_count: int
    target_index: str


class MarkUploadedResponse(BaseModel):
    message: str
    updated_count: int
    failed_count: int
    status: str


class MessageResponse(BaseModel):
    message: str


class ListFilesResponse(BaseModel):
    files: list[str]
    count: int
    container: str
    status: str


# Response model for listing documents in the server folder
class ListDocumentsResponse(BaseModel):
    files: list[str]
    count: int
    status: str
