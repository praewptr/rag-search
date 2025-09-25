from fastapi import APIRouter, HTTPException

from models.upload_txt import (
    DocumentItem,
    DocumentResponse,
    DocumentsPayload,
)
from services.azure_upload import process_and_upload, process_and_upload_batch
from services.client import search_client_text
from utils.text_manage import (
    load_mock_data,
    process_documents,
    save_mock_data,
    validate_document,
)

router = APIRouter()

MOCK_DB_FILE = "rag_text.json"


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
    Handles JSON with 'value' wrapper format: {"value": [...]}.
    """
    try:
        if not payload.value:
            raise HTTPException(
                status_code=400, detail="No documents provided for upload"
            )

        successful_uploads, failed_uploads = process_documents(payload.value)

        if successful_uploads == len(payload.value):
            message = f"🎉 Successfully uploaded all {successful_uploads} documents to Azure AI Search!"
            status = "success"
        elif successful_uploads > 0:
            message = f"⚠️ Uploaded {successful_uploads}/{len(payload.value)} documents. {len(failed_uploads)} failed."
            status = "partial_success"
        else:
            message = f"❌ Failed to upload any documents. All {len(payload.value)} documents failed."
            status = "failed"

        return {
            "message": message,
            "uploaded_count": successful_uploads,
            "status": status,
            "failed_count": len(failed_uploads),
            "errors": failed_uploads[:5],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to upload documents: {str(e)}"
        )


@router.post("/upload-single")
async def upload_single_document(document: DocumentItem) -> dict:
    """
    Upload a single text document to Azure Search.
    """
    try:
        validate_document(document)

        json_data = {
            "content": document.content,
            "source": document.source,
            "timestamp": document.timestamp,
        }

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
