import json
import os

from fastapi import APIRouter, HTTPException

from models.upload_txt import (
    DocumentItem,
    DocumentResponse,
    DocumentsPayload,
)
from services.client import search_client_text
from services.upload_text import process_and_upload, process_and_upload_batch

router = APIRouter()

MOCK_DB_FILE = "rag_text.json"


# Utility functions
def load_mock_data() -> dict:
    """Load data from the mock database file."""
    try:
        if not os.path.exists(MOCK_DB_FILE):
            return {"value": []}

        with open(MOCK_DB_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Ensure the data has the correct structure
        if isinstance(data, list):
            # If it's a list, wrap it in the expected structure
            data = {"value": data}
        elif not isinstance(data, dict) or "value" not in data:
            # If it's not the expected structure, create empty
            data = {"value": []}

        # Add IDs if they don't exist
        for i, item in enumerate(data["value"]):
            if "id" not in item or item["id"] is None:
                item["id"] = i + 1

        return data
    except Exception as e:
        print(f"Error loading mock data: {e}")
        return {"value": []}


def save_mock_data(data: dict):
    """Save data to the mock database file."""
    try:
        with open(MOCK_DB_FILE, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving mock data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save data: {str(e)}")


# API Endpoints
@router.get("/text", response_model=DocumentResponse)
async def get_documents():
    """
    Retrieve all documents from the mock database.
    """
    try:
        data = load_mock_data()
        return data
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load documents: {str(e)}"
        )


@router.delete("/text/{doc_id}")
async def delete_document(doc_id: int):
    """
    Delete a document by its ID from the mock database.
    """
    try:
        data = load_mock_data()
        documents = data["value"]

        # Find and remove the document with the specified ID
        original_count = len(documents)
        data["value"] = [doc for doc in documents if doc.get("id") != doc_id]

        if len(data["value"]) == original_count:
            raise HTTPException(status_code=404, detail="Document not found")

        save_mock_data(data)
        return {"message": f"Document with ID {doc_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete document: {str(e)}"
        )


@router.post("/upload_batch")
async def upload_batch(payload: DocumentsPayload):
    try:
        result = process_and_upload_batch([doc.model_dump() for doc in payload.value])
        return {
            "status": "success",
            "uploaded_chunks": len(result),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-text")
async def upload_text_documents(payload: DocumentsPayload) -> dict:
    """
    Upload text documents to Azure Search.
    Handles JSON with 'value' wrapper format: {"value": [...]}
    """
    try:
        if not payload.value:
            raise HTTPException(
                status_code=400, detail="No documents provided for upload"
            )

        print(f"🚀 Starting Azure upload for {len(payload.value)} documents...")

        successful_uploads = 0
        failed_uploads = []

        for i, doc_item in enumerate(payload.value):
            try:
                # Validate individual document
                if not doc_item.content.strip():
                    failed_uploads.append(f"Document {i + 1}: Empty content")
                    continue

                if not doc_item.source.strip():
                    failed_uploads.append(f"Document {i + 1}: Empty source")
                    continue

                # Prepare data for processing
                json_data = {
                    "content": doc_item.content,
                    "source": doc_item.source,
                    "timestamp": doc_item.timestamp,
                }

                print(f"📄 Processing document {i + 1}: {doc_item.source}")

                # Process and upload using the service
                result = process_and_upload(json_data)

                if result:
                    successful_uploads += 1
                    print(f"✅ Successfully uploaded document {i + 1}")
                else:
                    failed_uploads.append(
                        f"Document {i + 1} ({doc_item.source}): Upload failed"
                    )

            except Exception as doc_error:
                error_msg = f"Document {i + 1} ({doc_item.source}): {str(doc_error)}"
                failed_uploads.append(error_msg)
                print(f"❌ Error processing document {i + 1}: {doc_error}")

        # Determine response
        if successful_uploads == len(payload.value):
            message = f"🎉 Successfully uploaded all {successful_uploads} documents to Azure AI Search!"
            status = "success"
        elif successful_uploads > 0:
            message = f"⚠️ Uploaded {successful_uploads}/{len(payload.value)} documents. {len(failed_uploads)} failed."
            status = "partial_success"
        else:
            message = f"❌ Failed to upload any documents. All {len(payload.value)} documents failed."
            status = "failed"

        print(
            f"📊 Final result: {successful_uploads} uploaded, {len(failed_uploads)} failed"
        )

        return {
            "message": message,
            "uploaded_count": successful_uploads,
            "status": status,
            "failed_count": len(failed_uploads),
            "errors": failed_uploads[
                :5
            ],  # Return first 5 errors to avoid large responses
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error in upload_text_documents: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to upload documents: {str(e)}"
        )


@router.post("/upload-single")
async def upload_single_document(document: DocumentItem) -> dict:
    """
    Upload a single text document to Azure Search.
    """
    try:
        # Validate input data
        if not document.content.strip():
            raise HTTPException(
                status_code=400, detail="Document content cannot be empty"
            )

        if not document.source.strip():
            raise HTTPException(
                status_code=400, detail="Document source cannot be empty"
            )

        # Prepare data for processing
        json_data = {
            "content": document.content,
            "source": document.source,
            "timestamp": document.timestamp,
        }

        print(f"🔄 Processing single document from source: {document.source}")

        # Process and upload using the service
        result = process_and_upload(json_data)

        if result:
            return {
                "message": f"Successfully uploaded document from {document.source}",
                "uploaded_count": 1,
                "status": "success",
            }
        else:
            return {
                "message": f"Failed to upload document from {document.source}",
                "uploaded_count": 0,
                "status": "failed",
            }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error uploading single document: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to upload document: {str(e)}"
        )


@router.get("/index-text")
def get_text_index():
    results = search_client_text.search(search_text="*")
    all_contents = []
    for doc in results:
        # Assuming 'content' is the field name in your index
        content = doc.get("content")

        if content:
            all_contents.append(content)

    return {"documents": all_contents}
