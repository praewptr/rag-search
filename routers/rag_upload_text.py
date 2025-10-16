import logging

from fastapi import APIRouter, HTTPException, Path, Query

from models.upload import (
    DocumentResponse,
    DocumentsPayload,
    MarkUploaded,
    MarkUploadedResponse,
    MessageResponse,
    UploadTextResponse,
)
from services.azure_upload import process_and_upload_batch
from services.oracle_db import oracle_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/text", response_model=DocumentResponse)
async def get_documents(
    status: str = Query("all", enum=["all", "pending", "uploaded"]),
) -> DocumentResponse:
    """
    Retrieve documents from the Oracle database, filtered by upload status.
    - status: "all" (default), "pending" (ADDED=0), or "uploaded" (ADDED=1)
    """
    try:
        if status == "pending":
            data = oracle_service.fetch_knowledge_data(added=0)
        elif status == "uploaded":
            data = oracle_service.fetch_knowledge_data(added=1)
        else:
            data = oracle_service.fetch_knowledge_data()
        return data
    except Exception as e:
        logger.error(f"Failed to load documents: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to load documents: {str(e)}"
        )


@router.delete("/text/{doc_id}", response_model=MessageResponse)
async def delete_document(
    doc_id: int = Path(..., description="ID of the document to delete"),
) -> MessageResponse:
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

        return MessageResponse(
            message=f"Document with ID {doc_id} deleted successfully"
        )
    except Exception as e:
        logger.error(f"Failed to delete document {doc_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete document: {str(e)}"
        )


@router.post("/upload-text", response_model=UploadTextResponse)
async def upload_text_documents(payload: DocumentsPayload) -> UploadTextResponse:
    """
    Upload text from database to Azure Search.
    Handles JSON with 'value' wrapper format
    """
    try:
        if not payload.value:
            raise HTTPException(
                status_code=400, detail="No documents provided for upload"
            )

        documents_data = [doc.model_dump() for doc in payload.value]

        index_name = payload.index

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
                message = f"Successfully uploaded all {uploaded_count} document chunks to Azure AI Search{target_index_msg}!"
                status = "success"
            else:
                message = f"Uploaded {uploaded_count}/{total_docs} document chunks{target_index_msg}, {failed_count} failed."
                status = "partial_success"
        elif result:
            target_index_msg = (
                f" to index '{index_name}'" if index_name else " to default index"
            )
            message = f"Successfully processed and uploaded {len(documents_data)} documents to Azure AI Search{target_index_msg}!"
            status = "success"
            uploaded_count = len(documents_data)
            failed_count = 0
        else:
            message = "Failed to upload documents to Azure AI Search"
            status = "failed"
            uploaded_count = 0
            failed_count = len(payload.value)

        return UploadTextResponse(
            message=message,
            uploaded_count=uploaded_count,
            status=status,
            failed_count=failed_count,
            target_index=index_name or "default",
        )

    except Exception as e:
        logger.error(f"Failed to upload documents: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to upload documents: {str(e)}"
        )


@router.post("/mark-uploaded", response_model=MarkUploadedResponse)
async def mark_documents_as_uploaded(request: MarkUploaded) -> MarkUploadedResponse:
    """
    Mark documents as uploaded to Azure Search (set ADDED = 1)
    """
    try:
        document_ids = request.document_ids
        if not document_ids:
            raise HTTPException(status_code=400, detail="No document IDs provided")

        updated_count = 0
        failed_count = 0

        for doc_id in document_ids:
            try:
                success = oracle_service.mark_document_uploaded(doc_id)
                if success:
                    updated_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                logger.error(f"Error updating document {doc_id}: {e}")
                failed_count += 1

        return MarkUploadedResponse(
            message=f"Updated {updated_count} documents, {failed_count} failed",
            updated_count=updated_count,
            failed_count=failed_count,
            status="success" if failed_count == 0 else "partial_success",
        )

    except Exception as e:
        logger.error(f"Failed to mark documents as uploaded: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to mark documents as uploaded: {str(e)}"
        )
