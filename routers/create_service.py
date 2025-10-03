import logging

import requests
from azure.search.documents.indexes.models import SearchIndex
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import (
    azure_search_endpoint,
    azure_search_key,
)
from models.azure_index import CreateIndexResponse
from services.client import blob_service_client
from services.index_pdf import (
    create_datasource,
    create_index,
    create_indexer,
    create_skillset,
)
from utils.azure_index import (
    create_search_fields,
    create_vector_search_config,
    get_index_client,
)

router = APIRouter()

headers = {
    "Content-Type": "application/json",
    "api-key": azure_search_key,
}


class CreateDataSourceRequest(BaseModel):
    name: str
    container_name: str = "documents"


class CreateSkillsetRequest(BaseModel):
    name: str
    target_index_name: str = None
    description: str = "Document processing skillset"


class CreateIndexRequest(BaseModel):
    name: str
    analyzer_name: str


class CreateIndexerRequest(BaseModel):
    name: str
    datasource_name: str
    target_index_name: str
    skillset_name: str


@router.get("/list-indexers")
async def list_indexers():
    """List all available indexers from Azure Search."""
    try:
        response = requests.get(
            f"{azure_search_endpoint}/indexers?api-version=2023-10-01-Preview",
            headers=headers,
        )

        if response.status_code == 200:
            indexers_data = response.json()
            indexers = indexers_data.get("value", [])
            return {
                "status": "success",
                "indexers": [
                    {"name": indexer["name"], "status": indexer.get("status")}
                    for indexer in indexers
                ],
                "count": len(indexers),
            }
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to list indexers: {response.text}",
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while listing indexers: {str(e)}",
        )


@router.get("/list-containers")
async def list_containers():
    """List all blob containers in the Azure Storage account."""
    try:
        containers = []
        for container in blob_service_client.list_containers():
            containers.append(
                {
                    "name": container.name,
                    "last_modified": container.last_modified.isoformat()
                    if container.last_modified
                    else None,
                }
            )

        return {"containers": containers, "count": len(containers), "status": "success"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list containers: {str(e)}"
        )


@router.get("/list-skillsets")
async def list_skillsets():
    """List all available skillsets from Azure Search."""
    try:
        response = requests.get(
            f"{azure_search_endpoint}/skillsets?api-version=2023-10-01-Preview",
            headers=headers,
        )

        if response.status_code == 200:
            skillsets_data = response.json()
            skillsets = skillsets_data.get("value", [])
            return {
                "status": "success",
                "skillsets": [
                    {
                        "name": skillset["name"],
                        "description": skillset.get("description", "No description"),
                        "skills_count": len(skillset.get("skills", [])),
                        "cognitive_services_account": "configured"
                        if skillset.get("cognitiveServices")
                        else "not configured",
                    }
                    for skillset in skillsets
                ],
                "count": len(skillsets),
            }
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to list skillsets: {response.text}",
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while listing skillsets: {str(e)}",
        )


@router.get("/list-indexes")
async def list_indexes():
    """List all available indexes from Azure Search."""
    try:
        response = requests.get(
            f"{azure_search_endpoint}/indexes?api-version=2023-10-01-Preview",
            headers=headers,
        )

        if response.status_code == 200:
            indexes_data = response.json()
            indexes = indexes_data.get("value", [])
            return {
                "status": "success",
                "indexes": [
                    {
                        "name": index["name"],
                        "fields_count": len(index.get("fields", [])),
                        "vector_search_configured": bool(
                            index.get("vectorSearch", {}).get("profiles", [])
                        ),
                        "analyzers_count": len(index.get("analyzers", [])),
                        "cors_enabled": bool(index.get("corsOptions")),
                    }
                    for index in indexes
                ],
                "count": len(indexes),
            }
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to list indexes: {response.text}",
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while listing indexes: {str(e)}",
        )


@router.get("/list-datasources")
async def list_datasources():
    """List all available datasources from Azure Search."""
    try:
        response = requests.get(
            f"{azure_search_endpoint}/datasources?api-version=2023-10-01-Preview",
            headers=headers,
        )
        if response.status_code == 200:
            datasources_data = response.json()
            datasources = datasources_data.get("value", [])
            return {"items": [ds["name"] for ds in datasources]}
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to list datasources: {response.text}",
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while listing datasources: {str(e)}",
        )


@router.post("/skillset")
async def create_skillset_endpoint(request: CreateSkillsetRequest):
    """Create a new Azure Search skillset."""
    try:
        response = create_skillset(
            name=request.name,
            target_index_name=request.target_index_name,
        )

        if response.status_code in [200, 201]:
            return {
                "message": f"Skillset '{request.name}' created successfully.",
                "status": "success",
            }
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to create skillset: {response.text}",
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@router.post("/index-pdf")
async def create_index_endpoint(request: CreateIndexRequest):
    """Create a new Azure Search index."""
    try:
        # Use custom fields/vector_search if provided, else backend defaults
        fields = create_search_fields(
            name=request.name, analyzer_name=request.analyzer_name, include_vision=True
        )
        vector_search = create_vector_search_config(
            name=request.name, include_vision=True
        )

        create_index(name=request.name, fields=fields, vector_search=vector_search)

        message = f"Index '{request.name}' created successfully."
        return {
            "message": message,
            "status": "success",
            "vector_search_enabled": bool(vector_search),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@router.post("/indexer")
async def create_indexer_endpoint(request: CreateIndexerRequest):
    """Create a new Azure Search indexer."""
    try:
        response = create_indexer(
            name=request.name,
            datasource_name=request.datasource_name,
            target_index_name=request.target_index_name,
            skillset_name=request.skillset_name,
        )

        if response.status_code in [200, 201]:
            return {
                "message": f"Indexer '{request.name}' created successfully.",
                "status": "success",
            }
        else:
            # Try to get detailed error information
            try:
                error_details = response.json()
                error_message = error_details.get("error", {}).get(
                    "message", response.text
                )
            except Exception:
                error_message = response.text

            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to create indexer. Status: {response.status_code}. Error: {error_message}",
            )
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while creating indexer: {str(e)}",
        )


@router.post("/datasource")
async def create_datasource_endpoint(request: CreateDataSourceRequest):
    """Create a new Azure Search data source."""
    try:
        response = create_datasource(
            name=request.name, container_name=request.container_name
        )

        if response.status_code in [200, 201]:
            return {
                "message": f"Data source '{request.name}' created successfully.",
                "status": "success",
            }
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to create data source: {response.text}",
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


class CreateTextIndexRequest(BaseModel):
    index_name: str


@router.post("/create-text-index")
async def create_text_index(request: CreateTextIndexRequest):
    """Create a search index specifically for text documents (RAG)."""
    try:
        # Get the search client
        search_client = get_index_client()

        # Create search fields for text documents
        fields = create_search_fields()

        # Create vector search configuration
        vector_search_config = create_vector_search_config()

        # Create the index
        index = SearchIndex(
            name=request.index_name,
            fields=fields,
            vector_search=vector_search_config,
        )

        # Create the index in Azure Search
        result = search_client.create_index(index)

        return CreateIndexResponse(
            index_name=result.name,
            message=f"Text index '{result.name}' created successfully.",
            status="success",
        )

    except Exception as e:
        logging.error(f"Error creating text index: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create text index: {str(e)}"
        )
