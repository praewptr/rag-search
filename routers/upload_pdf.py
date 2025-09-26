from typing import List

import requests
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from config import (
    azure_search_endpoint,
    azure_search_key,
)
from services.client import blob_container_client, blob_service_client

router = APIRouter()


headers = {
    "Content-Type": "application/json",
    "api-key": azure_search_key,
}


def get_blob_client(filename: str, container_name: str = None):
    """Helper function to get the blob client for a given filename."""
    if container_name:
        container_client = blob_service_client.get_container_client(container_name)
    else:
        container_client = blob_service_client.get_container_client(
            blob_container_client.container_name
        )
    return container_client.get_blob_client(filename)


@router.post("/upload")
async def upload_to_blob(
    file: UploadFile = File(...),
    overwrite: bool = False,
    container_name: str = Query(None, description="Container name to upload to"),
):
    try:
        # Ensure the uploaded file is a PDF
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

        # Use provided container or default
        target_container = container_name or blob_container_client.container_name

        # Get blob client
        blob_client = get_blob_client(file.filename, target_container)

        # Check if the file already exists
        if blob_client.exists() and not overwrite:
            raise HTTPException(
                status_code=409,
                detail=f"File '{file.filename}' already exists in container '{target_container}'. Use overwrite=true to replace it.",
            )

        # Upload file to blob
        blob_client.upload_blob(file.file, overwrite=overwrite)

        return {
            "message": f"PDF file '{file.filename}' uploaded successfully to container '{target_container}'.",
            "container": target_container,
            "overwrite": overwrite,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to upload PDF file: {str(e)}"
        )


@router.get("/list")
async def list_blobs() -> List[str]:
    try:
        # Retrieve and list blobs in the container
        container_client = blob_service_client.get_container_client(
            blob_container_client.container_name
        )
        blobs = [blob.name for blob in container_client.list_blobs()]
        return blobs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list blobs: {str(e)}")


@router.get("/list-files/{container_name}")
async def list_files_from_container(container_name: str) -> List[str]:
    """List all files from a specific container."""
    try:
        container_client = blob_service_client.get_container_client(container_name)
        blobs = [blob.name for blob in container_client.list_blobs()]
        return blobs
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list files from container '{container_name}': {str(e)}",
        )


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
