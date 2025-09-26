from fastapi import APIRouter, HTTPException

from models.upload_txt import (
    DocumentItem,
    DocumentResponse,
    DocumentsPayload,
)
from services.azure_upload import process_and_upload, process_and_upload_batch
from services.client import search_client_text
from services.oracle_db import oracle_service
from utils.text_manage import (
    validate_document,
)

router = APIRouter()


# Debug endpoint to check Oracle table structure
@router.get("/text-debug")
async def debug_oracle_structure():
    """
    Debug endpoint to check the structure of Oracle data.
    """
    try:
        data = oracle_service.fetch_knowledge_data()

        # Get sample record structure
        sample_structure = {}
        if data.get("value") and len(data["value"]) > 0:
            sample_record = data["value"][0]
            for key, value in sample_record.items():
                sample_structure[key] = {
                    "type": type(value).__name__,
                    "sample_value": str(value)[:100] + "..."
                    if len(str(value)) > 100
                    else str(value),
                }

        return {
            "total_records": len(data.get("value", [])),
            "columns": list(sample_structure.keys()),
            "sample_structure": sample_structure,
            "first_record": data.get("value", [{}])[0] if data.get("value") else {},
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to debug Oracle structure: {str(e)}"
        )


# API Endpoints
@router.get("/text", response_model=DocumentResponse)
async def get_documents():
    """
    Retrieve all documents from the Oracle database.
    """
    try:
        data = oracle_service.fetch_knowledge_data()
        return data
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load documents: {str(e)}"
        )


@router.delete("/text/bulk")
async def delete_multiple_documents(doc_ids: list[int]):
    """
    Delete multiple documents by their IDs from the Oracle database.
    """
    try:
        deleted_count = 0
        failed_deletions = []

        for doc_id in doc_ids:
            try:
                # Check if document exists
                existing_doc = oracle_service.get_record_by_id(doc_id)
                if not existing_doc:
                    failed_deletions.append(f"Document ID {doc_id}: Not found")
                    continue

                # Delete the document
                success = oracle_service.delete_knowledge_record(doc_id)

                if success:
                    deleted_count += 1
                else:
                    failed_deletions.append(
                        f"Document ID {doc_id}: Could not be deleted"
                    )

            except Exception as e:
                failed_deletions.append(f"Document ID {doc_id}: {str(e)}")

        response = {
            "message": f"Successfully deleted {deleted_count} of {len(doc_ids)} documents",
            "deleted_count": deleted_count,
            "total_requested": len(doc_ids),
            "status": "success" if deleted_count > 0 else "partial_success",
        }

        if failed_deletions:
            response["failed_deletions"] = failed_deletions

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete documents: {str(e)}"
        )


@router.delete("/text/{doc_id}")
async def delete_document(doc_id: int):
    """
    Delete a document by its ID from the Oracle database.
    """
    try:
        # Check if document exists
        existing_doc = oracle_service.get_record_by_id(doc_id)
        if not existing_doc:
            raise HTTPException(status_code=404, detail="Document not found")

        # Delete the document
        success = oracle_service.delete_knowledge_record(doc_id)

        if not success:
            raise HTTPException(
                status_code=404, detail="Document not found or could not be deleted"
            )

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
    Handles JSON with 'value' wrapper format: {"value": [...], "index": "optional-index-name"}
    """
    try:
        if not payload.value:
            raise HTTPException(
                status_code=400, detail="No documents provided for upload"
            )

        # Convert Pydantic models to dictionaries for batch processing
        documents_data = [doc.model_dump() for doc in payload.value]

        # Get index from payload
        index_name = payload.index

        # Use batch upload function with optional index specification
        result = process_and_upload_batch(documents_data, index_name)

        if result and hasattr(result, "__iter__"):
            # Handle Azure Search result format
            uploaded_count = len(
                [r for r in result if hasattr(r, "succeeded") and r.succeeded]
            )
            failed_count = len(
                [r for r in result if hasattr(r, "succeeded") and not r.succeeded]
            )
            total_docs = len(result)

            target_index_msg = (
                f" to index '{index_name}'" if index_name else " to default index"
            )

            if failed_count == 0:
                message = f"🎉 Successfully uploaded all {uploaded_count} document chunks to Azure AI Search{target_index_msg}!"
                status = "success"
            else:
                message = f"⚠️ Uploaded {uploaded_count}/{total_docs} document chunks{target_index_msg}, {failed_count} failed."
                status = "partial_success"
        elif result:
            # Handle other result formats
            target_index_msg = (
                f" to index '{index_name}'" if index_name else " to default index"
            )
            message = f"🎉 Successfully processed and uploaded {len(documents_data)} documents to Azure AI Search{target_index_msg}!"
            status = "success"
            uploaded_count = len(documents_data)
            failed_count = 0
        else:
            message = "❌ Failed to upload documents to Azure AI Search"
            status = "failed"
            uploaded_count = 0
            failed_count = len(payload.value)

        return {
            "message": message,
            "uploaded_count": uploaded_count,
            "status": status,
            "failed_count": failed_count,
            "target_index": index_name or "default",
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
