import os
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from config import (
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


@router.get("/download-document")
async def download_document(filepath: str):
    """
    Download a document file from the server folder.
    """

    rel_path = filepath.lstrip("/")

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    file_path = Path(os.path.join(root_dir, rel_path))


    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(
        str(file_path), filename=file_path.name, media_type="application/pdf"
    )


@router.post("/upload-server-doc")
async def upload_server_doc(
    filename: str = Query(..., description="Name of the file in the server folder"),
    container_name: str = Query(..., description="Container name to upload to"),
    target_filename: str = Query(None, description="Target name for upload (optional)"),
    overwrite: bool = Query(False, description="Overwrite if file exists"),
):
    """
    Upload a file from the server folder to Azure Blob Storage.
    """

    DOCUMENTS_PATH =  Path(os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../../../storage/app/public/chat_files"
        )
    ))
    # DOCUMENTS_PATH = Path("C:/Users/PANTHIRA/mock_folder")
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
        return {
            "message": f"File '{upload_name}' uploaded successfully to container '{container_name}'.",
            "container": container_name,
            "overwrite": overwrite,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


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


@router.get("/list-documents", response_model=List[str])
def list_documents():
    """
    List all document filenames in the documents folder.
    """
    # DOCUMENTS_PATH = Path("C:/Users/PANTHIRA/mock_folder")

    DOCUMENTS_PATH = Path(os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../../../storage/app/public/chat_files"
        )
    ))
    if not DOCUMENTS_PATH.exists():
        return []

    # List only PDF files (not directories)
    file_list = [file.name for file in DOCUMENTS_PATH.iterdir() if file.is_file() and file.suffix.lower() == ".pdf"]
    return file_list


@router.delete("/delete-document/{filename}")
def delete_document(filename: str):
    """
    Delete a document by filename from the documents folder.
    """

    DOCUMENTS_PATH = Path("C:/Users/PANTHIRA/mock_folder")
    file_path = DOCUMENTS_PATH / filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    try:
        file_path.unlink()
        return {"message": f"{filename} deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete-file-from-blob")
async def delete_doc_container(
    filename: str = Query(..., description="Name of the file to delete"),
    container_name: str = Query(None, description="Container name to upload to"),
):
    try:
        # Get blob client
        blob_client = get_blob_client(filename, container_name)
        # Ensure the uploaded file is a PDF
        if not blob_client.exists():
            raise HTTPException(
                status_code=404,
                detail=f"File '{filename}' does not exist in container '{container_name}'.",
            )
        blob_client.delete_blob()
        return {
            "message": f"File '{filename}' deleted successfully from container '{container_name}'.",
            "container": container_name,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")
