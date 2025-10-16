import logging
import os
from pathlib import Path as FilePath

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import FileResponse

from config import azure_search_key, doc_storage_path
from models.upload import (
    DeleteResponse,
    ListContainersResponse,
    ListDocumentsResponse,
    ListFilesResponse,
    UploadResponse,
)
from services.client import blob_container_client, blob_service_client

router = APIRouter()
logger = logging.getLogger(__name__)


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


@router.get("/download-document")
async def download_document(
    filepath: str = Path(..., description="Relative path to the document file"),
):
    """
    Download a document file from the server folder.
    """

    rel_path = filepath.lstrip("/")

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    file_path = FilePath(os.path.join(root_dir, rel_path))

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(
        str(file_path), filename=file_path.name, media_type="application/pdf"
    )


@router.post("/upload-server-doc", response_model=UploadResponse)
async def upload_server_doc(
    filename: str = Query(..., description="Name of the file in the server folder"),
    container_name: str = Query(..., description="Container name to upload to"),
    target_filename: str = Query(None, description="Target name for upload (optional)"),
    overwrite: bool = Query(False, description="Overwrite if file exists"),
):
    """
    Upload a file from the server folder to Azure Blob Storage.
    """
    DOCUMENTS_PATH = FilePath(
        os.path.abspath(os.path.join(os.path.dirname(__file__), doc_storage_path))
    )
    file_path = DOCUMENTS_PATH / filename
    upload_name = target_filename if target_filename else filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=404, detail=f"File '{filename}' not found in server folder."
        )
    blob_client = get_blob_client(upload_name, container_name)
    if blob_client.exists() and not overwrite:
        raise HTTPException(
            status_code=409,
            detail=f"File '{upload_name}' already exists in container '{container_name}'. Use overwrite=true to replace it.",
        )
    try:
        with open(file_path, "rb") as f:
            blob_client.upload_blob(f, overwrite=overwrite)
            logger.info(
                f"File '{upload_name}' uploaded successfully to container '{container_name}'."
            )
        return UploadResponse(
            message=f"File '{upload_name}' uploaded successfully to container '{container_name}'.",
            container=container_name,
            overwrite=overwrite,
        )
    except Exception as e:
        logger.error(f"Failed to upload file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


@router.get("/list-files/{container_name}", response_model=ListFilesResponse)
async def list_files_from_container(
    container_name: str = Path(
        ..., description="Name of the container to list files from"
    ),
) -> ListFilesResponse:
    """List all files from a specific container."""
    try:
        container_client = blob_service_client.get_container_client(container_name)
        blobs = [blob.name for blob in container_client.list_blobs()]
        return ListFilesResponse(
            files=blobs, count=len(blobs), container=container_name, status="success"
        )
    except Exception as e:
        logger.error(f"Failed to list files from container '{container_name}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list files from container '{container_name}': {str(e)}",
        )


@router.get("/list-containers", response_model=ListContainersResponse)
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
        return ListContainersResponse(
            containers=containers, count=len(containers), status="success"
        )
    except Exception as e:
        logger.error(f"Failed to list containers: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list containers: {str(e)}"
        )


@router.get("/list-documents", response_model=ListDocumentsResponse)
def list_documents() -> ListDocumentsResponse:
    """
    List all document filenames in the documents folder.
    """
    DOCUMENTS_PATH = FilePath(
        os.path.abspath(os.path.join(os.path.dirname(__file__), doc_storage_path))
    )
    if not DOCUMENTS_PATH.exists():
        return ListDocumentsResponse(files=[], count=0, status="success")
    try:
        file_list = [
            file.name
            for file in DOCUMENTS_PATH.iterdir()
            if file.is_file() and file.suffix.lower() == ".pdf"
        ]
        return ListDocumentsResponse(
            files=file_list, count=len(file_list), status="success"
        )
    except Exception as e:
        logger.error(f"Failed to list documents in server folder: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list documents: {str(e)}"
        )


@router.delete("/delete-file-from-blob", response_model=DeleteResponse)
async def delete_doc_container(
    filename: str = Query(..., description="Name of the file to delete"),
    container_name: str = Query(None, description="Container name to upload to"),
):
    try:
        blob_client = get_blob_client(filename, container_name)
        if not blob_client.exists():
            raise HTTPException(
                status_code=404,
                detail=f"File '{filename}' does not exist in container '{container_name}'.",
            )
        blob_client.delete_blob()
        logger.info(
            f"File '{filename}' deleted successfully from container '{container_name}'."
        )
        return DeleteResponse(
            message=f"File '{filename}' deleted successfully from container '{container_name}'.",
            container=container_name,
        )
    except Exception as e:
        logger.error(f"Failed to delete file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")
