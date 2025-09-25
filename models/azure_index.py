from pydantic import BaseModel, Field


class CreateIndexRequest(BaseModel):
    """Request model for creating a new Azure Search index."""

    name: str = Field(..., description="Name of the index to create")
    force_recreate: bool = Field(
        False, description="Delete existing index if it exists"
    )


class CreateIndexResponse(BaseModel):
    """Response model for index creation."""

    status: str
    message: str
    index_name: str
    created: bool
